# Proyecto final - Vehiculo autonomo en Webots

Este repositorio contiene un controlador para un vehiculo autonomo en Webots. El sistema combina conduccion por **Behavioral Cloning** con capas de seguridad basadas en sensores para operar en un entorno urbano simulado.

El controlador activo es `src/main_bc.py`. Este archivo carga un modelo Keras entrenado con una arquitectura tipo NVIDIA para predecir el angulo de direccion a partir de la imagen de la camara, y despues aplica reglas de seguridad para peatones, obstaculos, distancia de seguimiento y maniobras de evasion.

## Caracteristicas principales

- Conduccion normal mediante Behavioral Cloning.
- Preprocesamiento de imagen compatible con el entrenamiento del modelo: RGB, normalizacion a `[0, 1]` y redimensionado a `200x66`.
- Deteccion de peatones con HOG + SVM en un hilo independiente.
- Deteccion frontal de obstaculos con LiDAR.
- Control de distancia segura con el objeto frontal.
- Maniobra de evasion para obstaculos no peatonales usando sensores laterales derechos y giroscopio.
- Frenado de emergencia ante peatones.
- Servicios opcionales para `Recognition`, radar y displays de depuracion en Webots.
- Dataset y notebook de entrenamiento para el modelo de Behavioral Cloning.

## Estructura del proyecto

```text
.
|-- README.md
|-- environment.yml
|-- Modelado/
|   |-- entrenamiento_behavioral_cloning_nvidia.ipynb
|   `-- Dataset/
|       |-- Annotations/      # CSV con image_name, steering_angle, nav_command
|       `-- JPEGImages/       # Imagenes de entrenamiento por recorrido/color
|-- resources/
|   `-- world/
|       |-- city_traffic_2025_01.wbt
|       |-- city_traffic_2025_02.wbt
|       `-- city_traffic_2025_02_net/  # Archivos SUMO del mundo
`-- src/
    |-- main_bc.py
    |-- config.py
    |-- models/
    |   |-- behavioral_cloning_nvidia.keras
    |   `-- pedestrian_svm_model_v2.joblib
    |-- perception/
    |   |-- bc_preprocessing.py
    |   |-- camera_service.py
    |   |-- lidar_service.py
    |   |-- pedestrian_detector.py
    |   `-- pedestrian_detection_thread.py
    |-- control/
    |   |-- bc_safety.py
    |   `-- vehicle_service.py
    |-- services/
    |   |-- gyro_service.py
    |   |-- radar_service.py
    |   |-- recognition_service.py
    |   `-- wall_sensors_service.py
    `-- display/
        |-- recognition_display_service.py
        `-- vehicle_overlay_display_service.py
```

## Flujo de funcionamiento

La prioridad de decision del controlador se define en `src/main_bc.py`:

```text
1. Frenado de emergencia por peaton.
2. Espera por vehiculo frontal via radar, si esta habilitada.
3. Maniobra de evasion ante obstaculos no peatonales.
4. Control de distancia segura con vehiculo frontal.
5. Conduccion normal usando Behavioral Cloning.
```

En conduccion normal, la camara entrega una imagen Webots en BGRA. `bc_preprocessing.py` la convierte a RGB, la normaliza, la redimensiona a `200x66` y la envia al modelo `behavioral_cloning_nvidia.keras`. La salida del modelo se interpreta como angulo de direccion y se limita con `BC_MAX_STEERING_ANGLE`.

En paralelo, `PedestrianDetectionThread` procesa el frame mas reciente cada cierto intervalo con `PedestrianSVMDetector`. El detector usa ventanas deslizantes, HOG, SVM, umbral de decision y Non-Maximum Suppression para determinar si hay peatones en la region de interes.

## Modelos incluidos

Los modelos usados por el controlador estan en `src/models/`:

- `behavioral_cloning_nvidia.keras`: modelo Keras para prediccion de direccion.
- `pedestrian_svm_model_v2.joblib`: modelo HOG + SVM para deteccion de peatones.

La ruta del modelo SVM se configura en `src/config.py`:

```python
PEDESTRIAN_SVM_MODEL_PATH = "models/pedestrian_svm_model_v2.joblib"
```

El modelo de Behavioral Cloning se carga desde `src/main_bc.py`:

```python
BC_MODEL_PATH = SRC_DIR / "models" / "behavioral_cloning_nvidia.keras"
```

## Dataset y entrenamiento

El directorio `Modelado/Dataset/` contiene los datos usados para entrenar el modelo de Behavioral Cloning:

- `Annotations/`: 19 archivos CSV.
- `JPEGImages/`: mas de 17,000 imagenes organizadas por recorridos.

Cada CSV usa columnas como:

```text
image_name,steering_angle,nav_command
```

El notebook `Modelado/entrenamiento_behavioral_cloning_nvidia.ipynb` prepara los datos, asocia imagenes con angulos de direccion, explora la distribucion de `steering_angle`, construye un pipeline `tf.data`, define data augmentation seguro para regresion y declara una red convolucional basada en NVIDIA. El entrenamiento y guardado del modelo se ejecutan manualmente desde el notebook cuando el dataset ya fue validado.

## Requisitos

- Webots con soporte para controlador externo de vehiculo.
- Conda o Miniconda.
- Python 3.10.
- Dependencias de vision y machine learning definidas en `environment.yml`.
- TensorFlow/Keras disponible para cargar el modelo `.keras`.

Dependencias principales:

- `numpy`
- `opencv-python`
- `scikit-image`
- `scikit-learn`
- `scipy`
- `joblib`
- `pillow`
- `imageio`

## Instalacion

Desde la raiz del repositorio:

```bash
conda env create -f environment.yml
conda activate webots_env
```

Para verificar algunas dependencias:

```bash
python --version
python -c "import cv2; print(cv2.__version__)"
python -c "import sklearn; print(sklearn.__version__)"
```

Si el entorno no incluye TensorFlow/Keras, instala la version compatible con tu plataforma antes de ejecutar `main_bc.py`.

## Ejecucion en Webots

1. Abre Webots.
2. Carga uno de los mundos disponibles en `resources/world/`, por ejemplo:

   ```text
   resources/world/city_traffic_2025_02.wbt
   ```

3. Verifica que el vehiculo use controlador externo.
4. Activa el ambiente:

   ```bash
   conda activate webots_env
   ```

5. Ejecuta el controlador desde `src/` con el comando de controlador externo de tu instalacion de Webots:

   ```bash
   cd src
   webots-controller main_bc.py
   ```

En Windows, el ejecutable puede llamarse `webots-controller.exe`.

## Parametros importantes

En `src/main_bc.py`:

- `BC_CRUISING_SPEED_KMH = 30.0`
- `BC_MAX_STEERING_ANGLE = 0.25`
- `BC_PREDICTION_INTERVAL_FRAMES = 10`
- `SAFE_FOLLOWING_DISTANCE_M = 10.0`
- `FOLLOWING_STOP_DISTANCE_M = 6.0`
- `PEDESTRIAN_STOP_DISTANCE_M = 18.0`
- `OBSTACLE_AVOIDANCE_TRIGGER_DISTANCE_M = 12.5`
- `OBSTACLE_AVOIDANCE_SPEED_KMH = 10.0`

En `src/config.py`:

- `CAMERA_NAME = "camera"`
- `DISPLAY_NAME = "display_image"`
- `LIDAR_MAX_DETECTION_DISTANCE = 30.0`
- `PEDESTRIAN_DECISION_THRESHOLD = 8.0`
- `PEDESTRIAN_DETECTION_INTERVAL = 0.2`
- `PEDESTRIAN_WINDOW_SIZES = ((12, 24), ..., (40, 80))`

Los nombres de dispositivos deben coincidir con los definidos en el mundo Webots. Los servicios opcionales se desactivan de forma defensiva si el mundo no incluye el dispositivo esperado.

## Modulos principales

- `src/main_bc.py`: controlador principal y logica de prioridades.
- `src/perception/bc_preprocessing.py`: conversion y preparacion de imagen para el modelo Keras.
- `src/perception/pedestrian_detector.py`: detector HOG + SVM.
- `src/perception/pedestrian_detection_thread.py`: inferencia de peatones sin bloquear el loop principal.
- `src/perception/lidar_service.py`: lectura y filtrado del LiDAR frontal.
- `src/control/bc_safety.py`: control de velocidad, distancia segura y evasion.
- `src/control/vehicle_service.py`: comandos de velocidad, direccion, freno e intermitentes.
- `src/services/wall_sensors_service.py`: sensores laterales derechos para la evasion.
- `src/services/gyro_service.py`: seguimiento de orientacion durante retorno de maniobra.
- `src/display/`: overlays de depuracion para Webots.

## Notas

El archivo `src/config.py` conserva parametros del pipeline anterior de carril/PID, pero la ejecucion actual del proyecto se concentra en `src/main_bc.py` y los modelos de Behavioral Cloning y peatones incluidos en `src/models/`.
# Proyecto_final_movilidad_autonoma_equipo10
