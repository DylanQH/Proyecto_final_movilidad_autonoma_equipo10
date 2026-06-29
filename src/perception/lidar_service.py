# lidar_service.py
# Servicio para manejar el LiDAR frontal del vehículo en Webots.
#
# El LiDAR se usa para detectar si existe un obstáculo frente al vehículo.
# Según la actividad, la detección se limita a un sector frontal de 20 o 30 grados
# y a una distancia máxima de 20 metros.

import math
import numpy as np


class LidarService:
    """
    Servicio para lectura del LiDAR frontal.

    Responsabilidades:
        - Buscar y habilitar el LiDAR.
        - Leer el rango frontal.
        - Limitar la lectura a un sector angular.
        - Detectar si existe un obstáculo dentro de la distancia máxima.
    """

    def __init__(
        self,
        robot,
        timestep,
        lidar_name_candidates,
        front_angle_degrees=30.0,
        max_detection_distance=20.0,
    ):
        self.robot = robot
        self.timestep = timestep
        self.lidar_name_candidates = lidar_name_candidates
        self.front_angle_degrees = front_angle_degrees
        self.max_detection_distance = max_detection_distance

        self.lidar = None

        for name in self.lidar_name_candidates:
            try:
                device = self.robot.getDevice(name)
                if device is not None:
                    self.lidar = device
                    print(f"[LiDAR] Dispositivo encontrado: {name}")
                    break
            except Exception:
                pass

        if self.lidar is None:
            raise RuntimeError(
                "No se encontró el LiDAR. Revisa el nombre del dispositivo en el mundo Webots."
            )

        self.lidar.enable(self.timestep)

        self.horizontal_resolution = self.lidar.getHorizontalResolution()
        self.fov = self.lidar.getFov()

        print("[LiDAR] Habilitado correctamente.")
        print(f"[LiDAR] Resolución horizontal: {self.horizontal_resolution}")
        print(f"[LiDAR] FOV rad: {self.fov}")
        print(f"[LiDAR] FOV deg: {math.degrees(self.fov):.2f}")

    def get_front_ranges(self):
        """
        Obtiene las mediciones del sector frontal del LiDAR.

        Returns:
            np.ndarray: distancias válidas dentro del sector frontal.
        """

        ranges = self.lidar.getRangeImage()

        if ranges is None:
            return np.array([])

        ranges = np.array(ranges, dtype=np.float32)

        if ranges.size == 0:
            return np.array([])

        center_index = len(ranges) // 2

        angle_per_point = self.fov / max(1, self.horizontal_resolution - 1)
        half_angle_rad = math.radians(self.front_angle_degrees / 2.0)
        half_window = int(half_angle_rad / angle_per_point)

        start = max(0, center_index - half_window)
        end = min(len(ranges), center_index + half_window + 1)

        front_ranges = ranges[start:end]

        # Quitar valores infinitos, NaN y valores no válidos.
        front_ranges = front_ranges[np.isfinite(front_ranges)]
        front_ranges = front_ranges[front_ranges > 0.0]

        return front_ranges

    def detect_obstacle(self):
        """
        Detecta si hay un obstáculo frente al vehículo.

        Returns:
            tuple:
                obstacle_detected (bool)
                min_distance (float | None)
        """

        front_ranges = self.get_front_ranges()

        if front_ranges.size == 0:
            return False, None

        min_distance = float(np.min(front_ranges))

        obstacle_detected = min_distance <= self.max_detection_distance

        return obstacle_detected, min_distance