# gyro_service.py
# Servicio para leer el giroscopio e integrar la orientación del vehículo.
#
# El giroscopio de Webots retorna la velocidad angular [wx, wy, wz] en rad/s.
# Este módulo integra la componente wz (yaw) a lo largo del tiempo para
# estimar la orientación acumulada del vehículo en el plano horizontal.
#
# La orientación se guarda al inicio de la maniobra de evasión y se usa
# como referencia para recuperar el rumbo original al concluir la evasión.

class GyroService:
    """
    Servicio para manejar el giroscopio de Webots.

    Responsabilidades:
        - Buscar el giroscopio por nombre dentro del Driver.
        - Habilitarlo con el timestep de simulación.
        - Integrar la velocidad angular Z para estimar la orientación acumulada.
        - Guardar y comparar orientaciones para la recuperación de rumbo.
    """

    def __init__(self, robot, gyro_name, timestep):
        """
        Inicializa el servicio del giroscopio.

        Args:
            robot:
                Instancia del Driver de Webots.

            gyro_name:
                Nombre del giroscopio en el mundo de Webots.
                Debe coincidir con el nombre en el archivo .wbt.

            timestep:
                Paso de simulación en milisegundos.
        """

        self.gyro = robot.getDevice(gyro_name)

        if self.gyro is None:
            raise RuntimeError(f"No se encontró el giroscopio con nombre: {gyro_name}")

        self.gyro.enable(timestep)

        # Orientación acumulada en el eje Z (yaw), en radianes.
        # Se actualiza en cada llamada a update().
        self.orientation = 0.0

        # Orientación guardada al inicio de la maniobra de evasión.
        # Se usa como referencia para la recuperación de rumbo.
        self.saved_orientation = 0.0

        print(f"Giroscopio '{gyro_name}' habilitado.")

    def update(self, dt):
        """
        Integra la velocidad angular Z para actualizar la orientación acumulada.

        Se debe llamar en cada iteración del loop principal antes de leer
        la orientación. La integración usa el método de Euler simple:

            orientacion += wz * dt

        Args:
            dt:
                Tiempo transcurrido desde la última actualización, en segundos.
                Se obtiene restando el tiempo de simulación anterior al actual.
        """

        # getValues() retorna [wx, wy, wz] en rad/s.
        # wz es la velocidad angular sobre el eje vertical (yaw).
        values = self.gyro.getValues()
        wz = values[2]

        # Integración de Euler: acumula el ángulo girado en este paso.
        self.orientation += wz * dt

    def get_orientation(self):
        """
        Retorna la orientación acumulada actual en radianes.

        Returns:
            float: Orientación acumulada en el eje Z [rad].
        """

        return self.orientation

    def save_orientation(self):
        """
        Guarda la orientación actual como referencia para la recuperación.

        Se llama justo antes de iniciar la maniobra de evasión, para
        registrar el rumbo que el vehículo tenía en ese momento.
        """

        self.saved_orientation = self.orientation
        return self.saved_orientation

    def get_saved_orientation(self):
        """
        Retorna la orientación guardada al inicio de la evasión.

        Returns:
            float: Orientación de referencia en radianes.
        """

        return self.saved_orientation

    def get_orientation_error(self):
        """
        Calcula la diferencia entre la orientación guardada y la actual.

        Un error positivo indica que el vehículo giró hacia la izquierda
        respecto a su rumbo original; negativo, hacia la derecha.

        Returns:
            float: Error de orientación en radianes.
        """

        return self.saved_orientation - self.orientation
