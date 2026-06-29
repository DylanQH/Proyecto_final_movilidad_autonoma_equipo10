"""
Capas de seguridad para el controlador de Behavioral Cloning.

Este modulo no depende directamente de Webots. Recibe servicios ya
inicializados y produce comandos de velocidad/direccion que main_bc.py aplica
al vehiculo segun la prioridad del controlador.
"""

from dataclasses import dataclass
import math

import numpy as np


@dataclass
class ControlCommand:
    """Comando final que se aplicara al vehiculo."""

    state: str
    speed_kmh: float
    steering_angle: float
    reason: str = ""


class SpeedGovernor:
    """
    Suaviza cambios de velocidad salvo cuando se solicita parada inmediata.
    """

    def __init__(self, cruise_speed_kmh, acceleration_step_kmh, deceleration_step_kmh):
        self.cruise_speed_kmh = float(cruise_speed_kmh)
        self.acceleration_step_kmh = float(acceleration_step_kmh)
        self.deceleration_step_kmh = float(deceleration_step_kmh)
        self.current_speed_kmh = float(cruise_speed_kmh)

    def reset(self, speed_kmh=0.0):
        self.current_speed_kmh = float(speed_kmh)
        return self.current_speed_kmh

    def approach(self, target_speed_kmh):
        target_speed_kmh = float(np.clip(target_speed_kmh, 0.0, self.cruise_speed_kmh))

        if target_speed_kmh < self.current_speed_kmh:
            self.current_speed_kmh = max(
                target_speed_kmh,
                self.current_speed_kmh - self.deceleration_step_kmh,
            )
        elif target_speed_kmh > self.current_speed_kmh:
            self.current_speed_kmh = min(
                target_speed_kmh,
                self.current_speed_kmh + self.acceleration_step_kmh,
            )

        return self.current_speed_kmh


class FollowingDistanceController:
    """
    Calcula una velocidad objetivo para conservar distancia con el vehiculo
    delantero.
    """

    def __init__(
        self,
        cruise_speed_kmh,
        safe_distance_m=15.0,
        stop_distance_m=3.0,
    ):
        self.cruise_speed_kmh = float(cruise_speed_kmh)
        self.safe_distance_m = float(safe_distance_m)
        self.stop_distance_m = float(stop_distance_m)

    def is_too_close(self, front_distance_m):
        if front_distance_m is None or not math.isfinite(front_distance_m):
            return False

        return 0.0 < front_distance_m < self.safe_distance_m

    def target_speed(self, front_distance_m):
        if front_distance_m is None or not math.isfinite(front_distance_m):
            return self.cruise_speed_kmh

        if front_distance_m <= self.stop_distance_m:
            return 0.0

        if front_distance_m >= self.safe_distance_m:
            return self.cruise_speed_kmh

        usable_range = max(0.1, self.safe_distance_m - self.stop_distance_m)
        ratio = (front_distance_m - self.stop_distance_m) / usable_range

        return float(np.clip(self.cruise_speed_kmh * ratio, 0.0, self.cruise_speed_kmh))


class ObstacleAvoidanceController:
    """
    Maquina de estados para evasiones con sensores laterales derechos y giro.

    Flujo:
        IDLE -> TURN_OUT -> FOLLOW_RIGHT_WALL -> RETURN_TO_HEADING -> IDLE
    """

    IDLE = "IDLE"
    TURN_OUT = "AVOID_TURN_OUT"
    FOLLOW_RIGHT_WALL = "AVOID_FOLLOW_RIGHT_WALL"
    RETURN_TO_HEADING = "AVOID_RETURN_TO_HEADING"

    def __init__(
        self,
        wall_sensors,
        gyro,
        max_steering_angle,
        avoidance_speed_kmh,
        turn_out_angle,
        turn_out_frames,
        min_follow_frames,
        max_follow_frames,
        return_angle,
        return_tolerance_rad,
        return_timeout_frames,
    ):
        self.wall_sensors = wall_sensors
        self.gyro = gyro
        self.max_steering_angle = float(max_steering_angle)
        self.avoidance_speed_kmh = float(avoidance_speed_kmh)
        self.turn_out_angle = float(turn_out_angle)
        self.turn_out_frames = int(turn_out_frames)
        self.min_follow_frames = int(min_follow_frames)
        self.max_follow_frames = int(max_follow_frames)
        self.return_angle = float(return_angle)
        self.return_tolerance_rad = float(return_tolerance_rad)
        self.return_timeout_frames = int(return_timeout_frames)

        self.state = self.IDLE
        self.phase_frames = 0
        self.total_frames = 0
        self.last_detail = ""

    @property
    def available(self):
        return self.wall_sensors is not None and self.gyro is not None

    @property
    def active(self):
        return self.state != self.IDLE

    def start(self):
        if not self.available:
            return False

        self.gyro.save_orientation()
        self.state = self.TURN_OUT
        self.phase_frames = 0
        self.total_frames = 0
        self.last_detail = "starting avoidance"

        return True

    def cancel(self):
        self.state = self.IDLE
        self.phase_frames = 0
        self.total_frames = 0
        self.last_detail = "cancelled"

    def update(self, front_obstacle_detected=False):
        if not self.active:
            return ControlCommand(
                state=self.IDLE,
                speed_kmh=0.0,
                steering_angle=0.0,
            )

        self.phase_frames += 1
        self.total_frames += 1

        if self.state == self.TURN_OUT:
            current_state = self.state
            steering_angle = self._clip_angle(self.turn_out_angle)
            self.last_detail = f"turn_out frame={self.phase_frames}/{self.turn_out_frames}"

            if self.phase_frames >= self.turn_out_frames:
                self._transition(self.FOLLOW_RIGHT_WALL)

            return ControlCommand(
                state=current_state,
                speed_kmh=self.avoidance_speed_kmh,
                steering_angle=steering_angle,
                reason=self.last_detail,
            )

        if self.state == self.FOLLOW_RIGHT_WALL:
            current_state = self.state
            d_front, d_mid, d_rear = self.wall_sensors.read()
            obstacle_cleared = self.wall_sensors.obstacle_cleared()
            steering_angle = 0.0

            self.last_detail = (
                f"straight_follow frame={self.phase_frames}/{self.max_follow_frames} "
                f"sensors=({d_front:.3f},{d_mid:.3f},{d_rear:.3f}) "
                f"obstacle_cleared={obstacle_cleared}"
            )

            if (
                (
                    self.phase_frames >= self.min_follow_frames
                    and obstacle_cleared
                    and not front_obstacle_detected
                )
                or self.phase_frames >= self.max_follow_frames
            ):
                self._transition(self.RETURN_TO_HEADING)

            return ControlCommand(
                state=current_state,
                speed_kmh=self.avoidance_speed_kmh,
                steering_angle=steering_angle,
                reason=self.last_detail,
            )

        if self.state == self.RETURN_TO_HEADING:
            orientation_error = self.gyro.get_orientation_error()
            steering_angle = self._clip_angle(self.return_angle)
            self.last_detail = (
                f"orientation_error={orientation_error:.4f} "
                f"fixed_return_angle={steering_angle:.3f}"
            )

            if (
                abs(orientation_error) <= self.return_tolerance_rad
                or self.phase_frames >= self.return_timeout_frames
            ):
                self.cancel()
                return ControlCommand(
                    state=self.IDLE,
                    speed_kmh=self.avoidance_speed_kmh,
                    steering_angle=0.0,
                    reason="avoidance complete",
                )

            return ControlCommand(
                state=self.state,
                speed_kmh=self.avoidance_speed_kmh,
                steering_angle=steering_angle,
                reason=self.last_detail,
            )

        self.cancel()
        return ControlCommand(
            state=self.IDLE,
            speed_kmh=0.0,
            steering_angle=0.0,
            reason="unknown avoidance state",
        )

    def _transition(self, new_state):
        self.state = new_state
        self.phase_frames = 0

    def _clip_angle(self, angle):
        return float(np.clip(angle, -self.max_steering_angle, self.max_steering_angle))
