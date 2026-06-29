"""
Controlador Webots con Behavioral Cloning y capas de seguridad.

Prioridad de decision:
    1. Frenado de emergencia por peaton.
    2. Espera por vehiculo frontal via radar si esta habilitada.
    3. Maniobra de evasion ante obstaculos no peatonales.
    4. Control de distancia segura con vehiculo frontal.
    5. Conduccion normal usando Behavioral Cloning.
"""

from pathlib import Path
import math
import traceback

import numpy as np
from vehicle import Driver

from perception.bc_preprocessing import (
    BC_INPUT_SHAPE,
    preprocess_for_behavioral_cloning,
)
from control.bc_safety import (
    FollowingDistanceController,
    ObstacleAvoidanceController,
    SpeedGovernor,
)
from perception.camera_service import CameraService
from config import (
    CAMERA_NAME,
    DISPLAY_NAME,
    LIDAR_FRONT_ANGLE_DEGREES,
    LIDAR_MAX_DETECTION_DISTANCE,
    LIDAR_NAME_CANDIDATES,
    PEDESTRIAN_DECISION_THRESHOLD,
    PEDESTRIAN_DETECTION_INTERVAL,
    PEDESTRIAN_DRAW_ALL_WINDOWS,
    PEDESTRIAN_DRAW_ROI,
    PEDESTRIAN_MAX_DRAW_WINDOWS,
    PEDESTRIAN_NMS_THRESHOLD,
    PEDESTRIAN_ROI_Y_END_RATIO,
    PEDESTRIAN_ROI_Y_START_RATIO,
    PEDESTRIAN_SCALE_FACTOR,
    PEDESTRIAN_STEP_SIZE,
    PEDESTRIAN_SVM_MODEL_PATH,
    PEDESTRIAN_WINDOW_SIZE,
    PEDESTRIAN_WINDOW_SIZES,
)
from perception.lidar_service import LidarService
from perception.pedestrian_detection_thread import PedestrianDetectionThread
from perception.pedestrian_detector import PedestrianSVMDetector
from control.vehicle_service import VehicleService

from services.gyro_service import GyroService
from display.recognition_display_service import RecognitionDisplayService
from display.vehicle_overlay_display_service import VehicleOverlayDisplayService
from services.radar_service import RadarService
from services.recognition_service import RecognitionService
from services.wall_sensors_service import WallSensorsService


SRC_DIR = Path(__file__).resolve().parent
BC_MODEL_PATH = SRC_DIR / "models" / "behavioral_cloning_nvidia.keras"

# Vehiculo.
BC_CRUISING_SPEED_KMH = 30.0
BC_MAX_STEERING_ANGLE = 0.25
BC_DEFAULT_STEERING_ANGLE = 0.0
BC_PREDICTION_INTERVAL_FRAMES = 10

# Control de distancia segura.
SAFE_FOLLOWING_DISTANCE_M = 10.0
FOLLOWING_STOP_DISTANCE_M = 6.0
SPEED_ACCELERATION_STEP_KMH = 0.6
SPEED_DECELERATION_STEP_KMH = 2.0

# Radar frontal para vehiculos.
RADAR_VEHICLE_WAIT_ENABLED = False
RADAR_NAME = "radar"
RADAR_FRONT_ANGLE_DEGREES = 30.0
RADAR_VEHICLE_STOP_DISTANCE_M = 20.0
RADAR_VEHICLE_RELEASE_DISTANCE_M = 22.0
RADAR_VEHICLE_RELEASE_FRAMES = 5

# Peatones.
PEDESTRIAN_RESULT_MAX_AGE_S = 0.75
PEDESTRIAN_STOP_DISTANCE_M = 18.0
PEDESTRIAN_RELEASE_FRAMES = 10

# Maniobra base de evasion ante obstaculos no peatonales.
OBSTACLE_AVOIDANCE_TRIGGER_DISTANCE_M = 12.5
OBSTACLE_AVOIDANCE_SPEED_KMH = 10.0
OBSTACLE_TURN_OUT_ANGLE = -0.15
OBSTACLE_TURN_OUT_FRAMES = 20
OBSTACLE_MIN_FOLLOW_FRAMES = 20
OBSTACLE_MAX_FOLLOW_FRAMES = 60
OBSTACLE_RETURN_ANGLE = 0.15
OBSTACLE_RETURN_TOLERANCE_RAD = 0.035
OBSTACLE_RETURN_TIMEOUT_FRAMES = 90

# Dispositivos nuevos. Los nombres deben coincidir con el mundo Webots.
GYRO_NAME_CANDIDATES = ("gyro", "Gyro")
RECOGNITION_FRONT_CENTER_RATIO = 0.70
RECOGNITION_DEBUG_DISPLAY_ENABLED = True

# Debug.
DEBUG_PRINT_EVERY_N_FRAMES = 20


def load_behavioral_cloning_model(model_path):
    """
    Carga el modelo Keras en modo inferencia.

    Se usa compile=False porque el controlador no entrena ni evalua metricas; de
    esta forma se evitan problemas por versiones distintas de optimizadores.
    """

    model_path = Path(model_path).resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"No se encontro el modelo: {model_path}")

    load_errors = []

    try:
        from tensorflow import keras as tf_keras

        return tf_keras.models.load_model(str(model_path), compile=False)
    except Exception as exc:
        load_errors.append(f"tensorflow.keras: {exc}")

    try:
        import keras

        return keras.models.load_model(str(model_path), compile=False)
    except Exception as exc:
        load_errors.append(f"keras: {exc}")

    details = "\n".join(load_errors)
    raise RuntimeError(f"No se pudo cargar el modelo Keras.\n{details}")


def resolve_src_path(path_value):
    path = Path(path_value)

    if path.is_absolute():
        return path

    return SRC_DIR / path


def clip_steering_angle(angle, max_abs_angle):
    """Limita el angulo de direccion a un rango seguro para Webots."""

    return float(np.clip(float(angle), -max_abs_angle, max_abs_angle))


def extract_scalar_prediction(prediction):
    """Convierte la salida del modelo a un float escalar."""

    values = np.asarray(prediction, dtype=np.float32).reshape(-1)

    if values.size == 0:
        raise ValueError("El modelo no devolvio ninguna prediccion.")

    return float(values[0])


def predict_steering_angle(model, image):
    """Preprocesa la imagen y ejecuta una inferencia del modelo BC."""

    input_batch = preprocess_for_behavioral_cloning(image)

    try:
        prediction = model(input_batch, training=False)

        if hasattr(prediction, "numpy"):
            prediction = prediction.numpy()
    except TypeError:
        prediction = model.predict(input_batch, verbose=0)

    return extract_scalar_prediction(prediction), input_batch.shape


class BehavioralCloningPredictionCache:
    """
    Ejecuta el modelo BC solo cada N frames y reutiliza la ultima prediccion.
    """

    def __init__(self, interval_frames, default_angle=0.0):
        self.interval_frames = max(1, int(interval_frames))
        self.default_angle = float(default_angle)

        self.last_evaluation_frame = None
        self.last_predicted_angle = self.default_angle
        self.last_input_shape = None

    def should_evaluate(self, frame_counter):
        if self.last_evaluation_frame is None:
            return True

        return frame_counter - self.last_evaluation_frame >= self.interval_frames

    def get_prediction(self, model, image, frame_counter):
        evaluated = False

        if self.should_evaluate(frame_counter):
            predicted_angle, input_shape = predict_steering_angle(model, image)
            self.last_predicted_angle = predicted_angle
            self.last_input_shape = input_shape
            self.last_evaluation_frame = frame_counter
            evaluated = True

        return {
            "predicted_angle": self.last_predicted_angle,
            "input_shape": self.last_input_shape,
            "evaluated": evaluated,
            "last_evaluation_frame": self.last_evaluation_frame,
            "interval_frames": self.interval_frames,
        }


def initialize_optional_service(name, factory):
    try:
        return factory()
    except Exception as exc:
        print(f"[BC][WARNING] {name} no disponible: {exc}")
        return None


def initialize_gyro(driver, timestep):
    last_error = None

    for gyro_name in GYRO_NAME_CANDIDATES:
        try:
            return GyroService(
                robot=driver,
                gyro_name=gyro_name,
                timestep=timestep,
            )
        except Exception as exc:
            last_error = exc

    print(f"[BC][WARNING] Giroscopio no disponible: {last_error}")
    return None


def initialize_pedestrian_thread():
    pedestrian_detector = PedestrianSVMDetector(
        model_path=resolve_src_path(PEDESTRIAN_SVM_MODEL_PATH),
        window_size=PEDESTRIAN_WINDOW_SIZE,
        scale_factor=PEDESTRIAN_SCALE_FACTOR,
        window_sizes=PEDESTRIAN_WINDOW_SIZES,
        step_size=PEDESTRIAN_STEP_SIZE,
        roi_y_start_ratio=PEDESTRIAN_ROI_Y_START_RATIO,
        roi_y_end_ratio=PEDESTRIAN_ROI_Y_END_RATIO,
        decision_threshold=PEDESTRIAN_DECISION_THRESHOLD,
        nms_threshold=PEDESTRIAN_NMS_THRESHOLD,
        draw_roi=PEDESTRIAN_DRAW_ROI,
        draw_all_windows=PEDESTRIAN_DRAW_ALL_WINDOWS,
        max_draw_windows=PEDESTRIAN_MAX_DRAW_WINDOWS,
    )

    pedestrian_thread = PedestrianDetectionThread(
        detector=pedestrian_detector,
        detection_interval=PEDESTRIAN_DETECTION_INTERVAL,
    )
    pedestrian_thread.start()

    return pedestrian_thread


def get_front_obstacle(lidar_service):
    obstacle_detected, front_distance = lidar_service.detect_obstacle()

    if front_distance is not None and not math.isfinite(front_distance):
        front_distance = None

    return obstacle_detected, front_distance


def detect_front_recognition_object(recognition_service, camera_width):
    """
    Detecta si hay un objeto reconocido en la zona central de la camara.

    En el mundo usado por los nuevos modulos, los objetos reconocidos son los
    vehiculos/autobuses con color de recognition configurado.
    """

    if recognition_service is None or not recognition_service.available:
        return False, None

    try:
        objects = recognition_service.get_objects()
    except Exception as exc:
        print(f"[BC][WARNING] Error leyendo recognition: {exc}")
        return False, None

    if not objects:
        return False, None

    center_x = camera_width / 2.0
    max_offset = camera_width * RECOGNITION_FRONT_CENTER_RATIO / 2.0

    for obj in objects:
        obj_id = None

        try:
            obj_id = obj.getId()
            position_on_image = obj.getPositionOnImage()
            obj_x = float(position_on_image[0])
        except Exception:
            return True, obj_id

        if abs(obj_x - center_x) <= max_offset:
            return True, obj_id

    return False, None


def is_pedestrian_in_path(pedestrian_result, front_distance):
    if pedestrian_result is None:
        return False

    if not pedestrian_result.get("detected", False):
        return False

    if front_distance is None:
        return True

    return front_distance <= PEDESTRIAN_STOP_DISTANCE_M


def should_control_following_distance(
    following_controller,
    front_distance,
    recognized_front_vehicle,
):
    if not following_controller.is_too_close(front_distance):
        return False

    if recognized_front_vehicle:
        return True

    return front_distance > OBSTACLE_AVOIDANCE_TRIGGER_DISTANCE_M


def apply_non_emergency_command(
    vehicle_service,
    speed_kmh,
    steering_angle,
    hazard_flashers=False,
):
    vehicle_service.release_brake()
    vehicle_service.set_hazard_flashers(hazard_flashers)
    vehicle_service.set_steering_angle(steering_angle)
    vehicle_service.set_speed(speed_kmh)


def update_recognition_display(
    recognition_display,
    recognition_service,
    image,
    front_distance,
    wall_sensors,
    state,
    frame_counter,
    pedestrian_detected=False,
    brake_active=False,
):
    if (
        recognition_display is None
        or recognition_service is None
        or not recognition_service.available
    ):
        return

    d_front = 0.0
    d_mid = 0.0
    d_rear = 0.0

    if wall_sensors is not None:
        try:
            d_front, d_mid, d_rear = wall_sensors.read()
        except Exception:
            pass

    try:
        recognized_objects = recognition_service.get_objects()

        recognition_display.update(
            camera_device=recognition_service.camera,
            camera_image_array=image,
            lidar_distance=front_distance if front_distance is not None else float("inf"),
            d_front=d_front,
            d_mid=d_mid,
            d_rear=d_rear,
            state=state,
            frame_counter=frame_counter,
            recognized_objects=recognized_objects,
            pedestrian_detected=pedestrian_detected,
            brake_active=brake_active,
        )
    except Exception as exc:
        print(f"[BC][WARNING] Error actualizando display de recognition: {exc}")


def print_controller_debug(
    frame_counter,
    state,
    front_distance,
    recognized_vehicle_id,
    pedestrian_result,
    obstacle_detected,
    speed_kmh,
    steering_angle,
    extra="",
):
    pedestrian_detected = (
        pedestrian_result is not None
        and pedestrian_result.get("detected", False)
    )
    pedestrian_score = None

    if pedestrian_result is not None:
        pedestrian_score = pedestrian_result.get("best_score")

    print("------------------------------------------")
    print(f"[BC] Frame: {frame_counter}")
    print(f"[BC] State: {state}")
    print(f"[BC] Pedestrian detected: {pedestrian_detected}")
    print(f"[BC] Pedestrian score: {pedestrian_score}")
    print(f"[BC] Front distance: {front_distance}")
    print(f"[BC] Recognized front vehicle id: {recognized_vehicle_id}")
    print(f"[BC] Obstacle detected: {obstacle_detected}")
    print(f"[BC] Applied speed: {speed_kmh:.3f} km/h")
    print(f"[BC] Applied steering angle: {steering_angle:.6f}")

    if extra:
        print(f"[BC] Detail: {extra}")

    print("------------------------------------------")


def main():
    print("==========================================")
    print("Starting Behavioral Cloning safety controller")
    print("==========================================")

    driver = Driver()
    timestep = int(driver.getBasicTimeStep())
    timestep_s = timestep / 1000.0
    print(f"Timestep: {timestep}")

    camera_service = CameraService(
        robot=driver,
        camera_name=CAMERA_NAME,
        timestep=timestep,
    )

    image_width = camera_service.get_width()
    image_height = camera_service.get_height()

    print("------------------------------------------")
    print(f"Camera name: {CAMERA_NAME}")
    print(f"Camera width: {image_width}")
    print(f"Camera height: {image_height}")
    print("------------------------------------------")

    model = load_behavioral_cloning_model(BC_MODEL_PATH)
    print("[BC] Modelo cargado correctamente.")
    print(f"[BC] Model path: {BC_MODEL_PATH}")
    print(f"[BC] Expected input shape: {BC_INPUT_SHAPE}")
    print(f"[BC] Model input shape: {getattr(model, 'input_shape', 'unknown')}")
    print(f"[BC] Model output shape: {getattr(model, 'output_shape', 'unknown')}")

    vehicle_service = VehicleService(
        driver=driver,
        cruising_speed=BC_CRUISING_SPEED_KMH,
        max_steering_angle=BC_MAX_STEERING_ANGLE,
    )

    lidar_service = LidarService(
        robot=driver,
        timestep=timestep,
        lidar_name_candidates=LIDAR_NAME_CANDIDATES,
        front_angle_degrees=LIDAR_FRONT_ANGLE_DEGREES,
        max_detection_distance=max(
            LIDAR_MAX_DETECTION_DISTANCE,
            SAFE_FOLLOWING_DISTANCE_M,
            RADAR_VEHICLE_RELEASE_DISTANCE_M,
            PEDESTRIAN_STOP_DISTANCE_M,
            OBSTACLE_AVOIDANCE_TRIGGER_DISTANCE_M,
        ),
    )

    radar_service = None
    if RADAR_VEHICLE_WAIT_ENABLED:
        radar_service = initialize_optional_service(
            "RadarService",
            lambda: RadarService(
                robot=driver,
                timestep=timestep,
                radar_name=RADAR_NAME,
                front_angle_degrees=RADAR_FRONT_ANGLE_DEGREES,
                max_distance_m=RADAR_VEHICLE_RELEASE_DISTANCE_M,
            ),
        )

    recognition_service = initialize_optional_service(
        "RecognitionService",
        lambda: RecognitionService(
            robot=driver,
            camera_name=CAMERA_NAME,
            timestep=timestep,
        ),
    )

    wall_sensors = initialize_optional_service(
        "WallSensorsService",
        lambda: WallSensorsService(
            robot=driver,
            timestep=timestep,
            turn_angle=BC_MAX_STEERING_ANGLE,
        ),
    )

    gyro_service = initialize_gyro(driver, timestep)

    recognition_display = None
    if (
        RECOGNITION_DEBUG_DISPLAY_ENABLED
        and recognition_service is not None
        and recognition_service.available
    ):
        recognition_display = initialize_optional_service(
            "RecognitionDisplayService",
            lambda: RecognitionDisplayService(
                robot=driver,
                display_name=DISPLAY_NAME,
                camera_width=image_width,
                camera_height=image_height,
            ),
        )
    vehicle_overlay_display = initialize_optional_service(
        "VehicleOverlayDisplayService",
        lambda: VehicleOverlayDisplayService(
            robot=driver,
            display_name=DISPLAY_NAME,
        ),
    )

    pedestrian_thread = initialize_pedestrian_thread()

    speed_governor = SpeedGovernor(
        cruise_speed_kmh=BC_CRUISING_SPEED_KMH,
        acceleration_step_kmh=SPEED_ACCELERATION_STEP_KMH,
        deceleration_step_kmh=SPEED_DECELERATION_STEP_KMH,
    )
    bc_prediction_cache = BehavioralCloningPredictionCache(
        interval_frames=BC_PREDICTION_INTERVAL_FRAMES,
        default_angle=BC_DEFAULT_STEERING_ANGLE,
    )
    following_controller = FollowingDistanceController(
        cruise_speed_kmh=BC_CRUISING_SPEED_KMH,
        safe_distance_m=SAFE_FOLLOWING_DISTANCE_M,
        stop_distance_m=FOLLOWING_STOP_DISTANCE_M,
    )
    avoidance_controller = ObstacleAvoidanceController(
        wall_sensors=wall_sensors,
        gyro=gyro_service,
        max_steering_angle=BC_MAX_STEERING_ANGLE,
        avoidance_speed_kmh=OBSTACLE_AVOIDANCE_SPEED_KMH,
        turn_out_angle=OBSTACLE_TURN_OUT_ANGLE,
        turn_out_frames=OBSTACLE_TURN_OUT_FRAMES,
        min_follow_frames=OBSTACLE_MIN_FOLLOW_FRAMES,
        max_follow_frames=OBSTACLE_MAX_FOLLOW_FRAMES,
        return_angle=OBSTACLE_RETURN_ANGLE,
        return_tolerance_rad=OBSTACLE_RETURN_TOLERANCE_RAD,
        return_timeout_frames=OBSTACLE_RETURN_TIMEOUT_FRAMES,
    )

    print("------------------------------------------")
    print(f"[BC] Cruising speed: {BC_CRUISING_SPEED_KMH} km/h")
    print(f"[BC] Safe following distance: {SAFE_FOLLOWING_DISTANCE_M} m")
    print(f"[BC] Radar vehicle wait enabled: {RADAR_VEHICLE_WAIT_ENABLED}")
    print(f"[BC] Radar vehicle stop distance: {RADAR_VEHICLE_STOP_DISTANCE_M} m")
    print(f"[BC] Obstacle avoidance trigger: {OBSTACLE_AVOIDANCE_TRIGGER_DISTANCE_M} m")
    print(f"[BC] Prediction interval: every {BC_PREDICTION_INTERVAL_FRAMES} frames")
    print(f"[BC] Avoidance available: {avoidance_controller.available}")
    print(f"[BC] Max steering angle: +/-{BC_MAX_STEERING_ANGLE}")
    print("Controller initialized successfully.")
    print("==========================================")

    frame_counter = 0
    camera_debug_printed = False
    pedestrian_emergency_active = False
    pedestrian_clear_counter = 0
    radar_vehicle_wait_active = False
    radar_vehicle_release_counter = 0
    last_state = None

    try:
        while driver.step() != -1:
            frame_counter += 1

            if gyro_service is not None:
                gyro_service.update(timestep_s)

            image = camera_service.get_image()

            if image is None:
                state = "CAMERA_WAIT"
                speed_governor.reset(0.0)
                vehicle_service.stop_and_center_steering()

                if frame_counter % DEBUG_PRINT_EVERY_N_FRAMES == 0:
                    print("[BC][WARNING] No image received from camera.")

                continue

            if not camera_debug_printed:
                print(f"[BC] Camera image received. Shape: {image.shape}")
                camera_debug_printed = True

            pedestrian_thread.update_frame(image)

            obstacle_detected, front_distance = get_front_obstacle(lidar_service)
            recognized_front_vehicle, recognized_vehicle_id = detect_front_recognition_object(
                recognition_service,
                image_width,
            )
            radar_vehicle = (
                radar_service.get_front_vehicle()
                if RADAR_VEHICLE_WAIT_ENABLED and radar_service is not None
                else None
            )
            radar_vehicle_distance = (
                radar_vehicle["distance"]
                if radar_vehicle is not None
                else None
            )
            pedestrian_result = pedestrian_thread.get_result(
                max_age=PEDESTRIAN_RESULT_MAX_AGE_S,
            )
            pedestrian_detected_for_display = (
                pedestrian_result is not None
                and pedestrian_result.get("detected", False)
            )
            pedestrian_now = is_pedestrian_in_path(
                pedestrian_result,
                front_distance,
            )

            if pedestrian_now:
                pedestrian_emergency_active = True
                pedestrian_clear_counter = 0
            elif pedestrian_emergency_active:
                if pedestrian_result is None:
                    pedestrian_clear_counter = 0
                else:
                    pedestrian_clear_counter += 1

                if pedestrian_clear_counter >= PEDESTRIAN_RELEASE_FRAMES:
                    pedestrian_emergency_active = False
                    pedestrian_clear_counter = 0
                    vehicle_service.release_brake()

            obstacle_close = (
                obstacle_detected
                and front_distance is not None
                and front_distance <= OBSTACLE_AVOIDANCE_TRIGGER_DISTANCE_M
            )
            radar_vehicle_close = (
                radar_vehicle_distance is not None
                and radar_vehicle_distance <= RADAR_VEHICLE_STOP_DISTANCE_M
            )

            if RADAR_VEHICLE_WAIT_ENABLED:
                if radar_vehicle_close and not avoidance_controller.active:
                    radar_vehicle_wait_active = True
                    radar_vehicle_release_counter = 0
                elif radar_vehicle_wait_active and not avoidance_controller.active:
                    radar_path_clear = (
                        radar_vehicle_distance is None
                        or radar_vehicle_distance >= RADAR_VEHICLE_RELEASE_DISTANCE_M
                    )

                    if radar_path_clear:
                        radar_vehicle_release_counter += 1
                    else:
                        radar_vehicle_release_counter = 0

                    if radar_vehicle_release_counter >= RADAR_VEHICLE_RELEASE_FRAMES:
                        radar_vehicle_wait_active = False
                        radar_vehicle_release_counter = 0
            else:
                radar_vehicle_wait_active = False
                radar_vehicle_release_counter = 0

            if pedestrian_emergency_active:
                if avoidance_controller.active:
                    avoidance_controller.cancel()

                state = "PEDESTRIAN_EMERGENCY_STOP"
                speed_governor.reset(0.0)
                vehicle_service.set_hazard_flashers(False)
                vehicle_service.stop_and_center_steering()

                applied_speed = 0.0
                applied_angle = 0.0
                extra = "priority=pedestrian emergency stop"

            elif (
                RADAR_VEHICLE_WAIT_ENABLED
                and radar_vehicle_wait_active
                and not avoidance_controller.active
            ):
                state = "RADAR_VEHICLE_WAIT"
                speed_governor.reset(0.0)
                vehicle_service.set_hazard_flashers(False)
                vehicle_service.stop_and_center_steering()

                applied_speed = 0.0
                applied_angle = 0.0
                extra = (
                    f"radar_vehicle_distance={radar_vehicle_distance} "
                    f"release_counter={radar_vehicle_release_counter}/"
                    f"{RADAR_VEHICLE_RELEASE_FRAMES} "
                    "priority=radar vehicle wait"
                )

            elif avoidance_controller.active or obstacle_close:
                if not avoidance_controller.active:
                    started = avoidance_controller.start()

                    if not started:
                        state = "OBSTACLE_AVOIDANCE_UNAVAILABLE"
                        applied_angle = clip_steering_angle(
                            OBSTACLE_TURN_OUT_ANGLE,
                            BC_MAX_STEERING_ANGLE,
                        )
                        applied_speed = speed_governor.approach(
                            OBSTACLE_AVOIDANCE_SPEED_KMH,
                        )
                        apply_non_emergency_command(
                            vehicle_service=vehicle_service,
                            speed_kmh=applied_speed,
                            steering_angle=applied_angle,
                            hazard_flashers=True,
                        )
                        extra = (
                            "avoidance unavailable; continuing without emergency brake "
                            "priority=obstacle avoidance"
                        )
                    else:
                        command = avoidance_controller.update(
                            front_obstacle_detected=obstacle_close,
                        )
                        state = command.state
                        applied_speed = speed_governor.approach(command.speed_kmh)
                        applied_angle = command.steering_angle
                        apply_non_emergency_command(
                            vehicle_service=vehicle_service,
                            speed_kmh=applied_speed,
                            steering_angle=applied_angle,
                            hazard_flashers=avoidance_controller.active,
                        )
                        extra = f"{command.reason} priority=obstacle avoidance"
                else:
                    command = avoidance_controller.update(
                        front_obstacle_detected=obstacle_close,
                    )
                    state = command.state
                    applied_speed = speed_governor.approach(command.speed_kmh)
                    applied_angle = command.steering_angle

                    apply_non_emergency_command(
                        vehicle_service=vehicle_service,
                        speed_kmh=applied_speed,
                        steering_angle=applied_angle,
                        hazard_flashers=avoidance_controller.active,
                    )
                    extra = f"{command.reason} priority=obstacle avoidance"

            elif should_control_following_distance(
                following_controller=following_controller,
                front_distance=front_distance,
                recognized_front_vehicle=recognized_front_vehicle,
            ):
                state = "SAFE_DISTANCE_CONTROL"
                target_speed = following_controller.target_speed(front_distance)
                applied_speed = speed_governor.approach(target_speed)

                if applied_speed <= 0.1:
                    applied_angle = BC_DEFAULT_STEERING_ANGLE
                    prediction_detail = "bc_prediction=skipped_stopped"
                else:
                    prediction_info = bc_prediction_cache.get_prediction(
                        model=model,
                        image=image,
                        frame_counter=frame_counter,
                    )
                    applied_angle = clip_steering_angle(
                        prediction_info["predicted_angle"],
                        BC_MAX_STEERING_ANGLE,
                    )
                    prediction_detail = (
                        f"bc_predicted_angle={prediction_info['predicted_angle']:.6f} "
                        f"bc_evaluated={prediction_info['evaluated']} "
                        f"last_bc_frame={prediction_info['last_evaluation_frame']} "
                        f"input_shape={prediction_info['input_shape']}"
                    )

                apply_non_emergency_command(
                    vehicle_service=vehicle_service,
                    speed_kmh=applied_speed,
                    steering_angle=applied_angle,
                    hazard_flashers=False,
                )

                extra = (
                    f"target_speed={target_speed:.3f} {prediction_detail} "
                    f"priority=safe distance"
                )

            else:
                state = "BEHAVIORAL_CLONING_NORMAL"
                prediction_info = bc_prediction_cache.get_prediction(
                    model=model,
                    image=image,
                    frame_counter=frame_counter,
                )
                applied_angle = clip_steering_angle(
                    prediction_info["predicted_angle"],
                    BC_MAX_STEERING_ANGLE,
                )
                applied_speed = speed_governor.approach(BC_CRUISING_SPEED_KMH)

                apply_non_emergency_command(
                    vehicle_service=vehicle_service,
                    speed_kmh=applied_speed,
                    steering_angle=applied_angle,
                    hazard_flashers=False,
                )

                extra = (
                    f"predicted_angle={prediction_info['predicted_angle']:.6f} "
                    f"bc_evaluated={prediction_info['evaluated']} "
                    f"last_bc_frame={prediction_info['last_evaluation_frame']} "
                    f"input_shape={prediction_info['input_shape']} "
                    f"priority=behavioral cloning"
                )

            update_recognition_display(
                recognition_display=recognition_display,
                recognition_service=recognition_service,
                image=image,
                front_distance=front_distance,
                wall_sensors=wall_sensors,
                state=state,
                frame_counter=frame_counter,
                pedestrian_detected=pedestrian_detected_for_display,
                brake_active=vehicle_service.brake_intensity > 0.0,
            )

            if vehicle_overlay_display is not None:
                vehicle_overlay_display.update(
                    camera_image=image,
                    state=state,
                    lidar_distance=front_distance,
                    pedestrian_detected=pedestrian_detected_for_display,
                    pedestrian_alert=pedestrian_emergency_active,
                    brake_active=vehicle_service.brake_intensity > 0.0,
                )

            should_print_debug = (
                frame_counter == 1
                or frame_counter % DEBUG_PRINT_EVERY_N_FRAMES == 0
                or state != last_state
            )

            if should_print_debug:
                print_controller_debug(
                    frame_counter=frame_counter,
                    state=state,
                    front_distance=front_distance,
                    recognized_vehicle_id=recognized_vehicle_id,
                    pedestrian_result=pedestrian_result,
                    obstacle_detected=obstacle_detected,
                    speed_kmh=applied_speed,
                    steering_angle=applied_angle,
                    extra=extra,
                )

            last_state = state

    finally:
        pedestrian_thread.stop()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("Behavioral Cloning safety controller crashed with exception:")
        traceback.print_exc()
        try:
            input("Press ENTER to close...")
        except EOFError:
            pass
