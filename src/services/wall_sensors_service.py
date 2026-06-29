# wall_sensors_service.py
# Servicio para leer los sensores de distancia del costado derecho del vehículo.
#
# Se usan tres sensores de distancia de rayo único (genérico) colocados en
# slots del costado derecho del vehículo:
#   - dist_front_right: sensor delantero derecho.
#   - dist_mid_right:   sensor central derecho.
#   - dist_rear_right:  sensor trasero derecho.
#
# Lookup table configurada en Webots:
#   0   0 0  -> a 0m retorna 0
#   0.1 0 0  -> a 0.1m retorna 0
#   4   1 0  -> a 4m retorna 1 (obstáculo detectado en rango útil)
#   5   0 0  -> a 5m retorna 0 (sin obstáculo)
#
# Por lo tanto:
#   - Valores altos (cercanos a 1.0) = hay obstáculo en el rango 0.1m~4m.
#   - Valores bajos (cercanos a 0.0) = sin obstáculo o demasiado cerca.
#
# El sensor trasero es el indicador clave de que el autobús ya quedó
# completamente atrás y la evasión puede concluir.

class WallSensorsService:
    """
    Servicio para los tres sensores de distancia del costado derecho.

    Responsabilidades:
        - Inicializar y habilitar los tres sensores.
        - Retornar las lecturas individuales de cada sensor.
        - Determinar si el obstáculo ya quedó atrás (fin de evasión).
        - Determinar el comando de giro según el algoritmo de pared derecha.
    """

    def __init__(self, robot, timestep,
                 name_front="dist_front_right",
                 name_mid="dist_mid_right",
                 name_rear="dist_rear_right",
                 max_distance=0.05,
                 turn_angle=0.2):
        """
        Inicializa los sensores de distancia laterales derechos.

        Args:
            robot:
                Instancia del Driver de Webots.

            timestep:
                Paso de simulación en milisegundos.

            name_front:
                Nombre del sensor delantero derecho en el mundo de Webots.

            name_mid:
                Nombre del sensor central derecho en el mundo de Webots.

            name_rear:
                Nombre del sensor trasero derecho en el mundo de Webots.

            max_distance:
                Umbral por debajo del cual se considera que no hay obstáculo.
                Con la lookup table actual, valores < 0.05 = sin obstáculo.

            turn_angle:
                Magnitud del ángulo de giro durante la maniobra de evasión.
                Positivo = izquierda, negativo = derecha.
        """

        self.sensor_front = robot.getDevice(name_front)
        self.sensor_mid   = robot.getDevice(name_mid)
        self.sensor_rear  = robot.getDevice(name_rear)

        for sensor, name in [
            (self.sensor_front, name_front),
            (self.sensor_mid,   name_mid),
            (self.sensor_rear,  name_rear),
        ]:
            if sensor is None:
                raise RuntimeError(
                    f"No se encontró el sensor de distancia: '{name}'. "
                    f"Verifica que el nombre coincide con el del mundo .wbt."
                )
            sensor.enable(timestep)

        self.max_distance = max_distance
        self.turn_angle   = turn_angle

        print(f"Sensores de pared derecha habilitados: "
              f"{name_front}, {name_mid}, {name_rear}")

    def read(self):
        """
        Lee los tres sensores y retorna sus valores actuales.

        Con la lookup table configurada:
            - Valor alto (~1.0) = hay obstáculo entre 0.1m y 4m.
            - Valor bajo (~0.0) = sin obstáculo o fuera del rango.

        Returns:
            tuple(float, float, float):
                (d_front, d_mid, d_rear)
        """

        d_front = self.sensor_front.getValue()
        d_mid   = self.sensor_mid.getValue()
        d_rear  = self.sensor_rear.getValue()
        return d_front, d_mid, d_rear

    def obstacle_detected_right(self):
        """
        Indica si algún sensor lateral detecta el autobús activamente.

        Returns:
            bool: True si al menos el sensor central detecta obstáculo.
        """

        _, d_mid, _ = self.read()
        return d_mid > self.max_distance

    def obstacle_cleared(self):
        """
        Indica si el autobús ya quedó completamente atrás del vehículo.

        La evasión concluye cuando el sensor trasero deja de detectar
        el autobús, es decir, retorna un valor bajo (< max_distance).

        Returns:
            bool: True si el obstáculo ya pasó, False si todavía está presente.
        """

        _, _, d_rear = self.read()
        return d_rear < self.max_distance

    def compute_wall_steering(self):
        _, d_mid, _ = self.read()

        # Valor objetivo del sensor central (paralelo al autobús)
        target = 0.25

        # Error entre el valor actual y el objetivo
        error = target - d_mid

        # Ganancia proporcional
        kp = 0.8

        # Ángulo proporcional al error
        wall_angle = kp * error

        # Limitar al ángulo máximo
        wall_angle = max(-self.turn_angle, min(self.turn_angle, wall_angle))

        return wall_angle, f"d_mid={d_mid:.3f} target={target} error={error:.3f} angle={wall_angle:.3f}"