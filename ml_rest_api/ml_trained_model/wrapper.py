import threading
import logging

log = logging.getLogger(__name__)
_init_lock = threading.Lock()

class TrainedModelWrapper:
    # ... (rest of your class)

    def load(self, module_name: str) -> None:
        try:
            self.module_name, _ = os.path.splitext(module_name)
            self.module = importlib.import_module(
                "ml_rest_api.ml_trained_model." + self.module_name
            )
            self._init = self._find_callable("init")
            self._run = self._find_callable("run")
            self._sample = self._find_callable("sample")
        except ImportError as e:
            log.error(f"Failed to import module '{module_name}': {e}")
            raise

    def multithreaded_init(self) -> None:
        if self._init and not self.initialised:
            Thread(target=self.init).start()

    def init(self) -> None:
        if self._init and not self.initialised:
            with _init_lock:
                if not self.initialised:  # Double-checked locking
                    self._init()
                    self.initialised = True
