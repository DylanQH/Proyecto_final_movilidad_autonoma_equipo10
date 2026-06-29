"""
Preprocesamiento para inferencia de Behavioral Cloning.

El notebook de entrenamiento decodifica imagenes en RGB, convierte los pixeles
a float32 en el rango [0, 1] y redimensiona a 66x200 con interpolacion de area.
No se aplica recorte/ROI en el pipeline base de entrenamiento.
"""

import cv2
import numpy as np


BC_IMAGE_HEIGHT = 66
BC_IMAGE_WIDTH = 200
BC_IMAGE_CHANNELS = 3
BC_INPUT_SHAPE = (BC_IMAGE_HEIGHT, BC_IMAGE_WIDTH, BC_IMAGE_CHANNELS)


def webots_image_to_rgb(image):
    """
    Convierte una imagen de Webots a RGB.

    CameraService entrega normalmente imagenes BGRA con forma
    (height, width, 4). La red se entreno con imagenes RGB decodificadas desde
    JPEG, por lo que la conversion de canales debe hacerse antes de inferir.
    """

    if image is None:
        raise ValueError("La imagen de entrada no puede ser None.")

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    if image.ndim != 3:
        raise ValueError(f"Forma de imagen no soportada: {image.shape}")

    channels = image.shape[2]

    if channels == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

    if channels == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    raise ValueError(f"Numero de canales no soportado: {channels}")


def preprocess_for_behavioral_cloning(image):
    """
    Prepara una imagen de camara para el modelo Keras.

    Returns:
        np.ndarray con forma (1, 66, 200, 3), dtype float32 y valores [0, 1].
    """

    rgb_image = webots_image_to_rgb(image)
    normalized = rgb_image.astype(np.float32) / 255.0

    resized = cv2.resize(
        normalized,
        (BC_IMAGE_WIDTH, BC_IMAGE_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )

    resized = np.clip(resized, 0.0, 1.0).astype(np.float32)

    return np.expand_dims(resized, axis=0)
