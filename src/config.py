# config.py
# Archivo de configuración general para el controlador autónomo en Webots.
#
# Este módulo centraliza todos los parámetros usados por el sistema:
#   - nombres de dispositivos de Webots,
#   - velocidad y límites del vehículo,
#   - parámetros de procesamiento de imagen,
#   - región de interés,
#   - filtro de líneas,
#   - ganancias del PID,
#   - opciones de guardado para depuración.
#
# Mantener estos valores en un archivo separado facilita realizar pruebas,
# ajustar parámetros y documentar el comportamiento del sistema.

import math


# ============================================================
# DISPOSITIVOS WEBOTS
# ============================================================
# Estos nombres deben coincidir exactamente con los nombres definidos
# en el mundo de Webots para la cámara y el display.

CAMERA_NAME = "camera"
DISPLAY_NAME = "display_image"


# ============================================================
# VEHÍCULO
# ============================================================
# CRUISING_SPEED_KMH define la velocidad constante del vehículo.
# La actividad solicita validar el funcionamiento a una velocidad mínima
# de 50 km/h.
#
# MAX_STEERING_ANGLE limita el ángulo máximo de dirección para evitar
# giros demasiado bruscos.
#
# DEFAULT_STEERING_ANGLE se usa cuando no se detecta ninguna línea válida;
# en ese caso, el vehículo intenta continuar recto.

CRUISING_SPEED_KMH = 50.0
MAX_STEERING_ANGLE = 0.25
DEFAULT_STEERING_ANGLE = 0.0


# ============================================================
# CANNY
# ============================================================
# Parámetros para la detección de bordes mediante el algoritmo de Canny.
#
# La imagen de la cámara primero se convierte a escala de grises y después
# se aplica Canny para obtener los bordes principales de la carretera y de
# la línea amarilla.
#
# CANNY_LOW_THRESHOLD:
#   umbral inferior para la detección de bordes.
#
# CANNY_HIGH_THRESHOLD:
#   umbral superior para la detección de bordes.

CANNY_LOW_THRESHOLD = 98
CANNY_HIGH_THRESHOLD = 222


# ============================================================
# HOUGHLINESP
# ============================================================
# Parámetros para la Transformada Probabilística de Hough.
#
# HoughLinesP toma como entrada la imagen de bordes después de aplicar
# la región de interés y devuelve segmentos de líneas rectas.
#
# HOUGH_RHO:
#   resolución en pixeles para la distancia rho.
#
# HOUGH_THETA:
#   resolución angular. math.pi / 180 equivale a 1 grado.
#
# HOUGH_THRESHOLD:
#   número mínimo de votos requerido para aceptar una línea.
#
# HOUGH_MIN_LINE_LENGTH:
#   longitud mínima en pixeles para aceptar un segmento de línea.
#
# HOUGH_MAX_LINE_GAP:
#   separación máxima permitida entre segmentos para unirlos como una línea.

HOUGH_RHO = 1
HOUGH_THETA = math.pi / 180

HOUGH_THRESHOLD = 12
HOUGH_MIN_LINE_LENGTH = 7
HOUGH_MAX_LINE_GAP = 8


# ============================================================
# ROI CON 6 PUNTOS CONFIGURABLES
# ============================================================
# Región de Interés, ROI.
#
# La ROI limita la zona de la imagen donde se buscan líneas. Esto ayuda a
# ignorar regiones no relevantes como cielo, edificios, sombras lejanas,
# banquetas o zonas superiores de la imagen.
#
# La ROI se define con seis puntos:
#   1. inferior izquierdo
#   2. medio izquierdo
#   3. superior izquierdo
#   4. superior derecho
#   5. medio derecho
#   6. inferior derecho
#
# Cada coordenada está expresada como proporción del ancho o alto de la imagen.
#
# Ejemplo:
#   ROI_LEFT_TOP_X = 0.24 significa:
#       x = 0.24 * image_width
#
#   ROI_LEFT_TOP_Y = 0.55 significa:
#       y = 0.55 * image_height
#
# En imágenes digitales:
#   - (0, 0) está en la esquina superior izquierda.
#   - X aumenta hacia la derecha.
#   - Y aumenta hacia abajo.
#
# Esta ROI se utiliza posteriormente con cv2.fillPoly() para crear una máscara.

ROI_LEFT_BOTTOM_X = 0.09
ROI_LEFT_BOTTOM_Y = 1.00

ROI_LEFT_MIDDLE_X = 0.15
ROI_LEFT_MIDDLE_Y = 0.82

ROI_LEFT_TOP_X = 0.24
ROI_LEFT_TOP_Y = 0.55

ROI_RIGHT_TOP_X = 0.65
ROI_RIGHT_TOP_Y = 0.55

ROI_RIGHT_MIDDLE_X = 0.74
ROI_RIGHT_MIDDLE_Y = 0.82

ROI_RIGHT_BOTTOM_X = 0.8
ROI_RIGHT_BOTTOM_Y = 1.00


# ============================================================
# FILTRO DE LÍNEAS
# ============================================================
# Después de HoughLinesP, se filtran líneas mayormente horizontales.
#
# Esto es importante porque en intersecciones, sombras o bordes de carretera
# pueden aparecer líneas que no representan la línea amarilla central.
#
# La pendiente se calcula como:
#   slope = dy / dx
#
# MIN_ABS_SLOPE define la pendiente mínima absoluta para considerar válida
# una línea. Si abs(slope) es menor que este valor, la línea se descarta
# por ser demasiado horizontal.

MIN_ABS_SLOPE = 0.11


# ============================================================
# BLUR
# ============================================================
# Nivel de desenfoque aplicado antes de Canny.
#
# El blur ayuda a reducir ruido en la imagen antes de detectar bordes.
#
# En el módulo de detección:
#   0 = sin blur
#   1 = kernel 3x3
#   2 = kernel 5x5
#   3 = kernel 7x7

BLUR_LEVEL = 3


# ============================================================
# PID
# ============================================================
# Ganancias del controlador PID.
#
# El controlador recibe como entrada el error horizontal entre:
#   - el centro de la imagen, usado como setpoint,
#   - el centro de la línea seleccionada por el detector.
#
# El error se calcula en pixeles y la salida del PID se usa como ángulo
# de dirección del vehículo.
#
# PID_KP:
#   término proporcional. Corrige en función del error actual.
#
# PID_KI:
#   término integral. Corrige errores acumulados en el tiempo.
#   En esta configuración se mantiene en cero para evitar acumulación
#   excesiva cuando se pierde la línea en intersecciones.
#
# PID_KD:
#   término derivativo. Ayuda a suavizar cambios bruscos en el error
#   y reduce oscilaciones.

PID_KP = 0.008
PID_KI = 0.00
PID_KD = 0.0015


# ============================================================
# DEBUG IMAGES
# ============================================================
# Parámetros para guardar imágenes de depuración.
#
# Estas imágenes sirven para revisar visualmente las etapas del pipeline:
#   - imagen original,
#   - escala de grises,
#   - bordes Canny,
#   - bordes dentro de la ROI,
#   - overlay con ROI y líneas detectadas.
#
# SAVE_DEBUG_IMAGES:
#   activa o desactiva el guardado.
#
# DEBUG_MAX_IMAGES:
#   si es None, no hay límite de imágenes guardadas.
#
# DEBUG_SAVE_EVERY_N_FRAMES:
#   define cada cuántos frames se guarda una imagen.
#
# SAVE_DEBUG_OVERLAY_IMAGE:
#   guarda la imagen final con ROI, líneas de Hough, líneas válidas,
#   línea seleccionada y centro de imagen.

SAVE_DEBUG_IMAGES = False

DEBUG_MAX_IMAGES = None
DEBUG_SAVE_EVERY_N_FRAMES = 1
DEBUG_OUTPUT_DIR = "debug_outputs"

SAVE_GRAY_IMAGE = False
SAVE_EDGES_IMAGE = False
SAVE_ROI_EDGES_IMAGE = False
SAVE_DEBUG_OVERLAY_IMAGE = True

# ============================================================
# LIDAR
# ============================================================
# El mundo usa un sensor SickLms291 en la parte frontal.
# El nombre exacto puede variar dependiendo de Webots, por eso se prueban
# varios nombres candidatos.

LIDAR_NAME_CANDIDATES = [
    "Sick LMS 291",
    "SickLms291",
    "lidar",
    "Lidar",
    "front lidar",
]

LIDAR_FRONT_ANGLE_DEGREES = 30.0
LIDAR_MAX_DETECTION_DISTANCE = 30.0


# ============================================================
# PEDESTRIAN SVM
# ============================================================

PEDESTRIAN_SVM_MODEL_PATH = "models/pedestrian_svm_model_v2.joblib"

PEDESTRIAN_WINDOW_SIZE = (64, 128)

# Configuración que mejor funcionó en Colab.
PEDESTRIAN_SCALE_FACTOR = 1


PEDESTRIAN_WINDOW_SIZES = (
    (12, 24), (16, 32), (20, 40), (24, 48),
    (28, 56), (32, 64), (36, 72), (40, 80),
)


PEDESTRIAN_STEP_SIZE = 16

PEDESTRIAN_ROI_Y_START_RATIO = 0.30
PEDESTRIAN_ROI_Y_END_RATIO = 0.95

PEDESTRIAN_DECISION_THRESHOLD = 8.0
PEDESTRIAN_NMS_THRESHOLD = 0.20

# Intervalo del hilo SVM.
# 0.3 significa que la detección corre aproximadamente cada 300 ms.
PEDESTRIAN_DETECTION_INTERVAL = 0.2

PEDESTRIAN_DRAW_ROI = True
PEDESTRIAN_DRAW_ALL_WINDOWS = True
PEDESTRIAN_MAX_DRAW_WINDOWS = 250