# pedestrian_detector.py
# Detector de peatones usando HOG + SVM.
#
# Pipeline:
#   imagen Webots BGRA/BGR
#       -> BGR
#       -> escala opcional
#       -> sliding window
#       -> resize interno a 64x128
#       -> HOG
#       -> SVM
#       -> threshold por score
#       -> NMS
#       -> debug image
#
# También incluye:
#   - draw_status_panel()
#     para dibujar LEDs de PEDESTRIAN / BARREL y estado general.

import cv2
import joblib
import numpy as np
from pathlib import Path
from skimage.feature import hog


class PedestrianSVMDetector:
    """
    Detector de peatones basado en HOG + SVM.
    """

    def __init__(
        self,
        model_path,
        window_size=(64, 128),
        scale_factor=1,
        window_sizes=((24, 48), (32, 64), (40, 80), (48, 96)),
        step_size=8,
        roi_y_start_ratio=0.30,
        roi_y_end_ratio=0.95,
        decision_threshold=4.00,
        nms_threshold=0.20,
        draw_roi=True,
        draw_all_windows=False,
        max_draw_windows=250,
    ):
        self.model_path = Path(model_path)

        # Tamaño usado durante entrenamiento.
        self.window_size = tuple(window_size)

        # Parámetros de inferencia.
        self.scale_factor = scale_factor
        self.window_sizes = tuple(window_sizes)
        self.step_size = step_size
        self.roi_y_start_ratio = roi_y_start_ratio
        self.roi_y_end_ratio = roi_y_end_ratio
        self.decision_threshold = decision_threshold
        self.nms_threshold = nms_threshold

        # Debug.
        self.draw_roi = draw_roi
        self.draw_all_windows = draw_all_windows
        self.max_draw_windows = max_draw_windows

        self.model = None
        self.scaler = None

        self.hog_params = {
            "orientations": 9,
            "pixels_per_cell": (8, 8),
            "cells_per_block": (2, 2),
            "block_norm": "L2-Hys",
            "transform_sqrt": True,
        }

        if not self.model_path.exists():
            raise FileNotFoundError(f"No existe el modelo SVM: {self.model_path}")

        self._load_model()

    def _load_model(self):
        """
        Carga el modelo .joblib exportado desde Colab.

        Soporta:
            - diccionario con model/scaler/window_size/hog_params
            - modelo directo
        """

        loaded_data = joblib.load(self.model_path)

        if isinstance(loaded_data, dict):
            print("[SVM] Archivo cargado como diccionario.")
            print(f"[SVM] Llaves disponibles: {list(loaded_data.keys())}")

            self.model = (
                loaded_data.get("model")
                or loaded_data.get("svm")
                or loaded_data.get("classifier")
                or loaded_data.get("svm_model")
            )

            self.scaler = (
                loaded_data.get("scaler")
                or loaded_data.get("standard_scaler")
            )

            if "window_size" in loaded_data:
                self.window_size = tuple(loaded_data["window_size"])

            if "hog_params" in loaded_data and isinstance(loaded_data["hog_params"], dict):
                self.hog_params.update(loaded_data["hog_params"])

            if self.model is None:
                raise ValueError(
                    "El archivo .joblib es un diccionario, pero no contiene "
                    "una llave válida para el modelo. Se esperaban llaves como "
                    "'model', 'svm', 'classifier' o 'svm_model'. "
                    f"Llaves encontradas: {list(loaded_data.keys())}"
                )

        else:
            print("[SVM] Archivo cargado como modelo directo.")
            self.model = loaded_data
            self.scaler = None

        print("[SVM] Modelo cargado correctamente.")
        print(f"[SVM] Ruta: {self.model_path}")
        print(f"[SVM] Usa scaler: {self.scaler is not None}")
        print(f"[SVM] Training window size: {self.window_size}")
        print(f"[SVM] Scale factor: {self.scale_factor}")
        print(f"[SVM] Window sizes: {self.window_sizes}")
        print(f"[SVM] Step size: {self.step_size}")
        print(f"[SVM] ROI Y: {self.roi_y_start_ratio} to {self.roi_y_end_ratio}")
        print(f"[SVM] Decision threshold: {self.decision_threshold}")
        print(f"[SVM] NMS threshold: {self.nms_threshold}")
        print(f"[SVM] HOG params: {self.hog_params}")

    def _to_bgr(self, image):
        """
        Convierte imagen de Webots a BGR.

        Webots normalmente entrega BGRA.
        """

        if image is None:
            return None

        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        if len(image.shape) == 3:
            if image.shape[2] == 4:
                return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

            if image.shape[2] == 3:
                return image.copy()

        return None

    def _extract_hog(self, bgr_window):
        """
        Redimensiona la ventana al tamaño de entrenamiento y extrae HOG.
        """

        train_w, train_h = self.window_size

        resized = cv2.resize(
            bgr_window,
            (train_w, train_h),
            interpolation=cv2.INTER_AREA,
        )

        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        features = hog(
            gray,
            orientations=self.hog_params.get("orientations", 9),
            pixels_per_cell=tuple(self.hog_params.get("pixels_per_cell", (8, 8))),
            cells_per_block=tuple(self.hog_params.get("cells_per_block", (2, 2))),
            block_norm=self.hog_params.get("block_norm", "L2-Hys"),
            transform_sqrt=self.hog_params.get("transform_sqrt", True),
            visualize=False,
            feature_vector=True,
        )

        return features.astype(np.float64)

    def _predict_window(self, bgr_window):
        """
        Clasifica una ventana usando HOG + SVM.

        Returns:
            prediction, score
        """

        features = self._extract_hog(bgr_window)
        features = features.reshape(1, -1)

        if self.scaler is not None:
            features = self.scaler.transform(features)

        prediction = int(self.model.predict(features)[0])

        score = 0.0

        if hasattr(self.model, "decision_function"):
            decision = self.model.decision_function(features)
            score = float(decision[0])

        return prediction, score

    def _apply_nms(self, detections):
        """
        Aplica Non-Maximum Suppression.

        detections:
            lista de (x1, y1, x2, y2, score, prediction)
        """

        boxes = []
        scores = []

        for x1, y1, x2, y2, score, prediction in detections:
            boxes.append([
                int(x1),
                int(y1),
                int(x2 - x1),
                int(y2 - y1),
            ])
            scores.append(float(score))

        if len(boxes) == 0:
            return boxes, scores, []

        indices = cv2.dnn.NMSBoxes(
            bboxes=boxes,
            scores=scores,
            score_threshold=self.decision_threshold,
            nms_threshold=self.nms_threshold,
        )

        if len(indices) == 0:
            return boxes, scores, []

        final_indices = np.array(indices).flatten().tolist()

        return boxes, scores, final_indices

    def draw_status_panel(
        self,
        image,
        pedestrian_detected=False,
        barrel_detected=False,
        obstacle_distance=None,
        mode="LANE",
        hazard_on=False,
        frame_counter=0,
    ):
        """
        Dibuja un panel compacto para imágenes pequeñas 256x128.

        Muestra:
            - LED P: peatón
            - LED B: barril
            - LED H: luces intermitentes (parpadea cuando estan ON)
            - estado general
            - distancia LiDAR
            - estado de luces (HAZ ON / HAZ OFF)
            - modo actual

        hazard_on:
            True  -> luces intermitentes encendidas (caso barril)
            False -> luces intermitentes apagadas (caso peaton / sin obstaculo)

        frame_counter:
            Se usa para hacer parpadear el LED de luces, igual que un
            intermitente real.
        """

        if image is None:
            return None

        img = image.copy()

        if len(img.shape) == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # Colores BGR
        color_panel = (25, 25, 25)
        color_border = (160, 160, 160)
        color_text = (255, 255, 255)

        color_led_off = (55, 55, 55)
        color_pedestrian = (0, 255, 0)
        color_barrel = (0, 180, 255)
        color_warning = (0, 0, 255)
        color_hazard = (0, 200, 255)

        # Panel pequeño
        x0, y0 = 3, 3
        x1, y1 = 200, 42

        cv2.rectangle(img, (x0, y0), (x1, y1), color_panel, -1)
        cv2.rectangle(img, (x0, y0), (x1, y1), color_border, 1)

        # LEDs compactos
        pedestrian_led_color = color_pedestrian if pedestrian_detected else color_led_off
        barrel_led_color = color_barrel if barrel_detected else color_led_off

        cv2.circle(img, (12, 13), 5, pedestrian_led_color, -1)
        cv2.circle(img, (12, 28), 5, barrel_led_color, -1)

        cv2.putText(
            img,
            "P",
            (22, 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            color_text,
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            img,
            "B",
            (22, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            color_text,
            1,
            cv2.LINE_AA,
        )

        # Estado corto
        if pedestrian_detected:
            status_text = "PED"
            status_color = color_pedestrian
        elif barrel_detected:
            status_text = "BARREL"
            status_color = color_barrel
        else:
            status_text = "NONE"
            status_color = color_warning

        cv2.putText(
            img,
            status_text,
            (42, 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            status_color,
            1,
            cv2.LINE_AA,
        )

        # Distancia compacta
        if obstacle_distance is not None:
            lidar_text = f"L:{obstacle_distance:.1f}m"
        else:
            lidar_text = "L:--"

        cv2.putText(
            img,
            lidar_text,
            (42, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            color_text,
            1,
            cv2.LINE_AA,
        )

        # Modo compacto
        mode_short = str(mode)[:4]

        cv2.putText(
            img,
            mode_short,
            (105, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.30,
            color_text,
            1,
            cv2.LINE_AA,
        )

        # ------------------------------------------------------------
        # Indicador de luces intermitentes (hazard flashers)
        # ------------------------------------------------------------
        # Se muestra el estado real de las luces:
        #   - LED que parpadea cuando estan encendidas (como un
        #     intermitente real),
        #   - texto "HAZ ON" / "HAZ OFF".
        #
        # El parpadeo se logra alternando el LED segun frame_counter.
        # ------------------------------------------------------------

        if hazard_on:
            # Parpadeo: encendido en frames pares, apagado en impares.
            blink = (int(frame_counter) // 5) % 2 == 0
            hazard_led_color = color_hazard if blink else color_led_off
            hazard_text = "HAZ ON"
            hazard_text_color = color_hazard
        else:
            hazard_led_color = color_led_off
            hazard_text = "HAZ OFF"
            hazard_text_color = color_text

        cv2.circle(img, (110, 13), 5, hazard_led_color, -1)

        cv2.putText(
            img,
            hazard_text,
            (120, 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.30,
            hazard_text_color,
            1,
            cv2.LINE_AA,
        )

        return img

    def detect(self, image):
        """
        Ejecuta detección de peatones.

        Returns:
            dict con:
                detected
                best_score
                best_box
                debug_image
                detected_boxes
                total_windows
                raw_detections
                final_detections
        """

        bgr_original = self._to_bgr(image)

        if bgr_original is None:
            return {
                "detected": False,
                "best_score": None,
                "best_box": None,
                "debug_image": None,
                "detected_boxes": [],
                "total_windows": 0,
                "raw_detections": 0,
                "final_detections": 0,
            }

        # ------------------------------------------------------------
        # Escala
        # ------------------------------------------------------------

        if self.scale_factor == 1:
            bgr = bgr_original.copy()
        else:
            bgr = cv2.resize(
                bgr_original,
                None,
                fx=self.scale_factor,
                fy=self.scale_factor,
                interpolation=cv2.INTER_LINEAR,
            )

        debug_image = bgr.copy()

        image_h, image_w = bgr.shape[:2]

        y_start = int(image_h * self.roi_y_start_ratio)
        y_end = int(image_h * self.roi_y_end_ratio)

        y_start = max(0, y_start)
        y_end = min(image_h, y_end)

        # ------------------------------------------------------------
        # Dibujar ROI
        # ------------------------------------------------------------

        if self.draw_roi:
            cv2.rectangle(
                debug_image,
                (0, y_start),
                (image_w - 1, y_end - 1),
                (255, 0, 0),
                1,
            )

            cv2.putText(
                debug_image,
                "search ROI",
                (5, max(15, y_start + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (255, 0, 0),
                1,
                cv2.LINE_AA,
            )

        # ------------------------------------------------------------
        # Sliding window
        # ------------------------------------------------------------

        detections = []
        total_windows = 0
        drawn_windows = 0

        best_score = -999999.0
        best_box = None
        best_prediction = None

        for window_w, window_h in self.window_sizes:
            window_w = int(window_w)
            window_h = int(window_h)

            if window_w <= 0 or window_h <= 0:
                continue

            if window_w > image_w or window_h > image_h:
                continue

            if y_end - y_start < window_h:
                continue

            for y in range(y_start, y_end - window_h + 1, self.step_size):
                for x in range(0, image_w - window_w + 1, self.step_size):
                    total_windows += 1

                    window = bgr[y:y + window_h, x:x + window_w]

                    if window.shape[0] != window_h or window.shape[1] != window_w:
                        continue

                    if self.draw_all_windows and drawn_windows < self.max_draw_windows:
                        cv2.rectangle(
                            debug_image,
                            (x, y),
                            (x + window_w, y + window_h),
                            (80, 80, 80),
                            1,
                        )
                        drawn_windows += 1

                    prediction, score = self._predict_window(window)

                    if score > best_score:
                        best_score = score
                        best_prediction = prediction
                        best_box = (
                            x,
                            y,
                            window_w,
                            window_h,
                            score,
                            prediction,
                        )

                    if score >= self.decision_threshold:
                        detections.append(
                            (
                                x,
                                y,
                                x + window_w,
                                y + window_h,
                                score,
                                prediction,
                            )
                        )

        # ------------------------------------------------------------
        # NMS
        # ------------------------------------------------------------

        boxes, scores, final_indices = self._apply_nms(detections)

        detected = len(final_indices) > 0

        detected_boxes = []

        for i in final_indices:
            x, y, w, h = boxes[i]
            score = scores[i]

            detected_boxes.append((x, y, w, h, score))

            cv2.rectangle(
                debug_image,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2,
            )

            cv2.putText(
                debug_image,
                f"score={score:.2f}",
                (x, max(y - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        # ------------------------------------------------------------
        # Textos de debug del SVM
        # ------------------------------------------------------------

        status_text = "SVM: PEDESTRIAN" if detected else "SVM: NO PEDESTRIAN"
        status_color = (0, 255, 0) if detected else (0, 0, 255)

        cv2.putText(
            debug_image,
            status_text,
            (5, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            status_color,
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            debug_image,
            f"best={best_score:.2f}",
            (5, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            status_color,
            1,
            cv2.LINE_AA,
        )

        if best_box is not None:
            _, _, bw, bh, _, _ = best_box

            cv2.putText(
                debug_image,
                f"best box={bw}x{bh}",
                (5, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                status_color,
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            debug_image,
            f"raw={len(detections)} final={len(final_indices)}",
            (5, 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            debug_image,
            f"windows={total_windows}",
            (5, 98),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        return {
            "detected": detected,
            "best_score": best_score,
            "best_box": best_box,
            "debug_image": debug_image,
            "detected_boxes": detected_boxes,
            "total_windows": total_windows,
            "raw_detections": len(detections),
            "final_detections": len(final_indices),
        }