# recognition_display_service.py
# Servicio para mostrar en un display secundario:
#   1. Los ROIs de los autobuses reconocidos con su ID.
#   2. La distancia del LiDAR al autobús más cercano.

from controller import Display
import numpy as np
import cv2

# IDs fijos de los autobuses en el mundo de Webots.
BUS_IDS = {12009, 11185}


class RecognitionDisplayService:
    """
    Muestra en un display secundario los ROIs de los autobuses y
    la distancia del LiDAR, usando OpenCV para dibujar texto.
    """

    def __init__(self, robot, display_name, camera_width, camera_height):
        self.display = robot.getDevice(display_name)
        self.width   = camera_width
        self.height  = camera_height

        if self.display is None:
            print(f"WARNING: No se encontró el display '{display_name}'.")
        else:
            print(f"Display de reconocimiento '{display_name}' habilitado.")

    def update(self, camera_device, camera_image_array, lidar_distance,
               d_front=0.0, d_mid=0.0, d_rear=0.0,
               state="SEGUIMIENTO_LINEA", frame_counter=0,
               recognized_objects=None,
               pedestrian_detected=False,
               brake_active=False):
        """
        Dibuja ROIs de autobuses, distancia LiDAR, valores de sensores
        laterales e indicador de estado en el display secundario.

        Args:
            camera_device:
                Objeto Camera de Webots con recognition habilitado.

            camera_image_array:
                Imagen actual de la cámara como array numpy (H, W, 4).

            lidar_distance:
                Distancia frontal del LiDAR en metros (float).

            d_front, d_mid, d_rear:
                Valores de los sensores laterales derechos (float 0.0-1.0).

            state:
                Estado actual del controlador (string).

            frame_counter:
                Contador de frames para el parpadeo del LED.

            pedestrian_detected:
                True si el detector de peatones reporta presencia.

            brake_active:
                True si el freno esta activo.
        """

        if self.display is None or camera_image_array is None:
            return

        # Convertir BGRA -> BGR para OpenCV.
        img = camera_image_array[:, :, :3].copy()

        if recognized_objects is None:
            try:
                objects = camera_device.getRecognitionObjects()
            except Exception:
                objects = []
        else:
            objects = recognized_objects

        for obj in objects:
            obj_id = obj.getId()

            # Filtrar: solo autobuses conocidos.
            if obj_id not in BUS_IDS:
                continue

            pos  = obj.getPositionOnImage()
            size = obj.getSizeOnImage()

            cx = int(pos[0])
            cy = int(pos[1])
            hw = max(1, int(size[0] / 2))
            hh = max(1, int(size[1] / 2))

            x1 = max(0, cx - hw)
            y1 = max(0, cy - hh)
            x2 = min(self.width  - 1, cx + hw)
            y2 = min(self.height - 1, cy + hh)

            # Rectángulo rojo alrededor del ROI.
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)

            # ID del autobús encima del rectángulo.
            cv2.putText(img, f"ID:{obj_id}",
                        (x1, max(y1 - 2, 6)),
                        cv2.FONT_HERSHEY_PLAIN, 0.6,
                        (0, 0, 255), 1)

        # Distancia del LiDAR.
        if lidar_distance == float('inf'):
            dist_text = "LiDAR: --"
        else:
            dist_text = f"LiDAR: {lidar_distance:.1f}m"

        cv2.putText(img, dist_text,
                    (2, self.height - 3),
                    cv2.FONT_HERSHEY_PLAIN, 0.7,
                    (0, 255, 0), 1)

        # Valores de los tres sensores laterales derechos.
        cv2.putText(img, f"F:{d_front:.2f}",
                    (2, 8),
                    cv2.FONT_HERSHEY_PLAIN, 0.6,
                    (255, 255, 0), 1)
        cv2.putText(img, f"M:{d_mid:.2f}",
                    (2, 16),
                    cv2.FONT_HERSHEY_PLAIN, 0.6,
                    (255, 255, 0), 1)
        cv2.putText(img, f"R:{d_rear:.2f}",
                    (2, 24),
                    cv2.FONT_HERSHEY_PLAIN, 0.6,
                    (255, 255, 0), 1)

        # LED de estado: verde parpadeante = reconociendo, rojo = esquivando.
        active   = (state == "SEGUIMIENTO_LINEA")
        blink_on = (frame_counter // 15) % 2 == 0

        if active:
            led_color = (0, 255, 0) if blink_on else (0, 100, 0)
            label     = "RECONOCIENDO"
        else:
            led_color = (0, 255, 255) if blink_on else (0, 100, 100)
            label     = "ESQUIVANDO"

        h, w = img.shape[:2]
        cv2.circle(img, (w - 8, 8), 3, led_color, -1)
        cv2.putText(img, label, (w - 60, 10),
                    cv2.FONT_HERSHEY_PLAIN, 0.4, led_color, 1)

        # LEDs auxiliares: amarillo = peaton, rojo = freno activo.
        pedestrian_led_color = (0, 255, 255) if pedestrian_detected else (0, 70, 70)
        brake_led_color = (0, 0, 255) if brake_active else (0, 0, 70)
        aux_x = max(48, w - 42)
        ped_y = max(36, h - 24)
        brake_y = max(48, h - 12)

        cv2.circle(img, (aux_x, ped_y), 4, pedestrian_led_color, -1)
        cv2.putText(img, "PED", (aux_x + 6, ped_y + 3),
                    cv2.FONT_HERSHEY_PLAIN, 0.5, pedestrian_led_color, 1)

        cv2.circle(img, (aux_x, brake_y), 4, brake_led_color, -1)
        cv2.putText(img, "BRK", (aux_x + 6, brake_y + 3),
                    cv2.FONT_HERSHEY_PLAIN, 0.5, brake_led_color, 1)

        # Convertir BGR -> RGB para Webots Display.
        img_rgb = img[:, :, ::-1].copy()

        image_ref = self.display.imageNew(
            img_rgb.tobytes(),
            Display.RGB,
            width=self.width,
            height=self.height,
        )
        self.display.imagePaste(image_ref, 0, 0, False)
        self.display.imageDelete(image_ref)
