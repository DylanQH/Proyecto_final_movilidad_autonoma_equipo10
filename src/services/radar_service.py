import math


class RadarService:
    """
    Reads the Webots Radar device and reports the closest frontal vehicle target.
    """

    def __init__(
        self,
        robot,
        timestep,
        radar_name="radar",
        front_angle_degrees=30.0,
        max_distance_m=25.0,
    ):
        self.robot = robot
        self.timestep = timestep
        self.radar_name = radar_name
        self.front_half_angle_rad = math.radians(front_angle_degrees) / 2.0
        self.max_distance_m = float(max_distance_m)

        self.radar = self.robot.getDevice(self.radar_name)
        if self.radar is None:
            raise RuntimeError(
                f"No se encontro el radar '{self.radar_name}'. "
                "Verifica el nombre del dispositivo en Webots."
            )

        self.radar.enable(timestep)
        print(
            f"Radar '{self.radar_name}' habilitado "
            f"(front_angle={front_angle_degrees} deg, max={self.max_distance_m} m)."
        )

    def get_front_vehicle(self):
        targets = self._get_targets()
        front_targets = []

        for target in targets:
            distance = self._target_value(target, "distance", "getDistance")
            if distance is None or not math.isfinite(distance):
                continue

            distance = float(distance)
            if distance <= 0.0 or distance > self.max_distance_m:
                continue

            azimuth = self._target_value(target, "azimuth", "getAzimuth", default=0.0)
            if azimuth is not None and abs(float(azimuth)) > self.front_half_angle_rad:
                continue

            front_targets.append(
                {
                    "distance": distance,
                    "azimuth": float(azimuth) if azimuth is not None else 0.0,
                    "speed": self._optional_float(
                        self._target_value(target, "speed", "getSpeed")
                    ),
                    "received_power": self._optional_float(
                        self._target_value(
                            target,
                            "received_power",
                            "getReceivedPower",
                            fallback_attr_name="receivedPower",
                        )
                    ),
                }
            )

        if not front_targets:
            return None

        return min(front_targets, key=lambda item: item["distance"])

    def _get_targets(self):
        try:
            return self.radar.getTargets()
        except Exception as exc:
            print(f"[Radar][WARNING] No se pudieron leer targets: {exc}")
            return []

    def _target_value(
        self,
        target,
        attr_name,
        method_name,
        default=None,
        fallback_attr_name=None,
    ):
        if hasattr(target, attr_name):
            return getattr(target, attr_name)

        if fallback_attr_name is not None and hasattr(target, fallback_attr_name):
            return getattr(target, fallback_attr_name)

        if hasattr(target, method_name):
            return getattr(target, method_name)()

        return default

    def _optional_float(self, value):
        if value is None:
            return None

        try:
            return float(value)
        except (TypeError, ValueError):
            return None
