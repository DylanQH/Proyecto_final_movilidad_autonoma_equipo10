class RecognitionService:
    """
    Servicio defensivo para el nodo Recognition de una Camera de Webots.

    Si el mundo activo no define Recognition dentro de la camara, este servicio
    se desactiva desde la inicializacion para evitar el error:
    wb_camera_recognition_get_number_of_objects() called on a Camera without
    Recognition node.
    """

    def __init__(self, robot, camera_name, timestep):
        self.camera_name = camera_name
        self.camera = robot.getDevice(camera_name)
        self.available = False

        if self.camera is None:
            raise RuntimeError(
                f"No se encontro la camara '{camera_name}' para recognition."
            )

        if not self._camera_has_recognition():
            raise RuntimeError(
                f"La camara '{camera_name}' no tiene nodo Recognition. "
                "La capa de reconocimiento se desactivara para este mundo."
            )

        self.camera.recognitionEnable(timestep)
        self.available = True

        print(f"Nodo Recognition habilitado en camara '{camera_name}'.")

    def _camera_has_recognition(self):
        has_recognition = getattr(self.camera, "hasRecognition", None)

        if callable(has_recognition):
            return bool(has_recognition())

        # Compatibilidad con wrappers antiguos de Webots: si no existe
        # hasRecognition(), se intenta continuar y get_objects() desactiva el
        # servicio si la llamada falla.
        return True

    def get_objects(self):
        if not self.available:
            return []

        try:
            return self.camera.getRecognitionObjects()
        except Exception as exc:
            self.available = False
            print(
                f"[Recognition][WARNING] Recognition se desactivo en "
                f"'{self.camera_name}': {exc}"
            )
            return []

    def detect_bus(self):
        objects = self.get_objects()

        if not objects:
            return False, None

        first_object = objects[0]
        return True, first_object.getId()
