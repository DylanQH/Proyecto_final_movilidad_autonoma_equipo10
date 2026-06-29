# pedestrian_detection_thread.py
# Hilo para ejecutar la detección SVM + HOG sin congelar el loop principal.

import threading
import time


class PedestrianDetectionThread:
    """
    Ejecuta el detector de peatones en un hilo separado.

    La idea es:
        - main.py manda siempre el frame más reciente.
        - el hilo procesa solo cada cierto intervalo.
        - main.py usa el último resultado disponible.
    """

    def __init__(self, detector, detection_interval=0.3):
        self.detector = detector
        self.detection_interval = detection_interval

        self.latest_frame = None
        self.latest_result = None
        # Momento en que se calculo el ultimo resultado. Sirve para saber si
        # el resultado es reciente o esta "viejo" (stale).
        self.latest_result_time = 0.0

        self.running = False
        self.thread = None
        self.lock = threading.Lock()

    def start(self):
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(
            target=self._run,
            daemon=False
        )
        self.thread.start()

        print("[PedestrianThread] Started.")

    def stop(self):
        self.running = False

        if self.thread is not None:
            self.thread.join(timeout=1.0)

        print("[PedestrianThread] Stopped.")

    def update_frame(self, frame):
        """
        Guarda el frame más reciente.

        Se usa copy() para evitar que el frame cambie mientras el hilo lo procesa.
        """
        if frame is None:
            return

        with self.lock:
            self.latest_frame = frame.copy()

    def invalidate_result(self):
        """
        Borra el ultimo resultado guardado.

        Se llama cuando empieza una nueva emergencia, para no reutilizar el
        resultado de la emergencia anterior (por ejemplo, marcar un barril
        como peaton porque justo antes paso una persona).
        """
        with self.lock:
            self.latest_result = None
            self.latest_result_time = 0.0

    def get_result(self, max_age=None):
        """
        Devuelve el último resultado disponible.

        max_age:
            Si se indica (en segundos), solo devuelve el resultado si fue
            calculado hace menos de max_age segundos. Si el resultado es mas
            viejo que eso, devuelve None (se considera stale).
            Si es None, devuelve el ultimo resultado sin importar su edad.
        """
        with self.lock:
            if self.latest_result is None:
                return None

            if max_age is not None:
                age = time.time() - self.latest_result_time
                if age > max_age:
                    return None

            return self.latest_result

    def _run(self):
        """
        Loop interno del hilo.
        """
        while self.running:
            start_time = time.time()

            with self.lock:
                if self.latest_frame is None:
                    frame = None
                else:
                    frame = self.latest_frame.copy()

            if frame is not None:
                try:
                    result = self.detector.detect(frame)

                    with self.lock:
                        self.latest_result = result
                        self.latest_result_time = time.time()

                except Exception as error:
                    print("[PedestrianThread] Error during detection:", error)

            elapsed = time.time() - start_time
            sleep_time = max(0.0, self.detection_interval - elapsed)
            time.sleep(sleep_time)