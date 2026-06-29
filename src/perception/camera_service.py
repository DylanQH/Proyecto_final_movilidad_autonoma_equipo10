# camera_service.py
# Servicio encargado de obtener imágenes desde la cámara de Webots.
#
# Este módulo separa la lógica de adquisición de imagen del resto del proyecto.
# De esta forma, el controlador principal no necesita conocer directamente
# cómo Webots entrega los datos de la cámara.

import numpy as np


class CameraService:
    """
    Servicio para manejar la cámara de Webots.

    Responsabilidades:
        - Buscar la cámara por nombre dentro del robot/driver.
        - Habilitar la cámara con el timestep de simulación.
        - Obtener la imagen actual como arreglo de NumPy.
        - Consultar ancho y alto de la cámara.
        - Guardar imágenes directamente desde Webots si se requiere.
    """

    def __init__(self, robot, camera_name, timestep):
        """
        Inicializa el servicio de cámara.

        Args:
            robot:
                Instancia del robot o Driver de Webots.

            camera_name:
                Nombre de la cámara dentro del mundo de Webots.
                Debe coincidir exactamente con el nombre configurado
                en el archivo .wbt.

            timestep:
                Paso de simulación usado para habilitar la cámara.
        """

        self.robot = robot
        self.camera_name = camera_name
        self.timestep = timestep

        # Obtener el dispositivo de cámara desde Webots.
        self.camera = self.robot.getDevice(self.camera_name)

        if self.camera is None:
            raise RuntimeError(f"No se encontró la cámara con nombre: {self.camera_name}")

        # Habilitar la cámara para que Webots empiece a entregar imágenes.
        self.camera.enable(self.timestep)

    def get_image(self):
        """
        Obtiene la imagen actual de la cámara.

        Webots entrega la imagen como un buffer de bytes. Para poder procesarla
        con OpenCV y NumPy, el buffer se convierte a un arreglo con forma:

            (height, width, 4)

        Los 4 canales corresponden normalmente a BGRA/RGBA, dependiendo del
        formato interno usado por Webots.

        Returns:
            np.ndarray:
                Imagen con forma (height, width, 4).

            None:
                Si Webots todavía no entrega una imagen válida.
        """

        raw_image = self.camera.getImage()

        if raw_image is None:
            return None

        width = self.camera.getWidth()
        height = self.camera.getHeight()

        # np.frombuffer convierte el buffer de Webots a un arreglo NumPy.
        # reshape organiza los datos como imagen: alto, ancho y 4 canales.
        image = np.frombuffer(raw_image, dtype=np.uint8).reshape((height, width, 4))

        return image

    def get_width(self):
        """
        Obtiene el ancho de la imagen entregada por la cámara.

        Returns:
            int: ancho de la cámara en pixeles.
        """

        return self.camera.getWidth()

    def get_height(self):
        """
        Obtiene el alto de la imagen entregada por la cámara.

        Returns:
            int: alto de la cámara en pixeles.
        """

        return self.camera.getHeight()

    def save_image(self, path, quality=100):
        """
        Guarda una imagen directamente usando la función de Webots.

        Args:
            path:
                Ruta donde se guardará la imagen.

            quality:
                Calidad de guardado. El valor 100 indica máxima calidad.
        """

        self.camera.saveImage(path, quality)