# vehicle_service.py
# Servicio para controlar el vehículo de Webots.

class VehicleService:
    """
    Servicio para controlar velocidad, dirección y luces del vehículo.
    """

    def __init__(self, driver, cruising_speed, max_steering_angle):
        self.driver = driver
        self.cruising_speed = cruising_speed
        self.max_steering_angle = max_steering_angle

        self.current_steering_angle = 0.0
        self.current_speed = self.cruising_speed
        self.hazard_flashers_enabled = False
        self.brake_intensity = 0.0

        self.driver.setCruisingSpeed(self.current_speed)
        self.driver.setSteeringAngle(self.current_steering_angle)

        self._set_brake_intensity(0.0)
        self.set_hazard_flashers(False)

    def _set_brake_intensity(self, intensity):
        """
        Aplica intensidad de freno si el Driver lo permite.

        intensity:
            0.0 = sin freno
            1.0 = freno máximo
        """

        intensity = max(0.0, min(1.0, float(intensity)))
        self.brake_intensity = intensity

        try:
            self.driver.setBrakeIntensity(intensity)
        except Exception as e:
            print(f"[WARNING] No se pudo aplicar brake intensity: {e}")

    def set_steering_angle(self, angle):
        """
        Aplica el ángulo de dirección limitado.
        """

        if angle > self.max_steering_angle:
            angle = self.max_steering_angle
        elif angle < -self.max_steering_angle:
            angle = -self.max_steering_angle

        self.current_steering_angle = angle
        self.driver.setSteeringAngle(self.current_steering_angle)

    def keep_constant_speed(self):
        """
        Mantiene la velocidad constante configurada.

        También libera el freno si estaba activo.
        """

        self._set_brake_intensity(0.0)
        self.set_speed(self.cruising_speed)

    def set_speed(self, speed):
        """
        Cambia la velocidad de crucero del vehículo.
        """

        self.current_speed = float(speed)
        self.driver.setCruisingSpeed(self.current_speed)

    def emergency_stop(self):
        """
        Aplica frenado de emergencia real.

        Primero pone velocidad objetivo en 0 y luego aplica freno máximo.
        """

        self.set_speed(0.0)
        self._set_brake_intensity(1.0)

    def release_brake(self):
        """
        Libera el freno manualmente.
        """

        self._set_brake_intensity(0.0)

    def set_hazard_flashers(self, enabled):
        """
        Activa o desactiva luces intermitentes.
        """

        enabled = bool(enabled)

        if self.hazard_flashers_enabled == enabled:
            return

        self.hazard_flashers_enabled = enabled

        try:
            self.driver.setHazardFlashers(enabled)
        except Exception as e:
            print(f"[WARNING] No se pudieron cambiar las luces intermitentes: {e}")

    def stop_and_center_steering(self):
        """
        Detiene el vehículo y centra la dirección.
        """

        self.emergency_stop()
        self.set_steering_angle(0.0)