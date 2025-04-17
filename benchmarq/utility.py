import subprocess
from threading import Lock


class VLLMServerSingleton:
    _instance = None
    _lock = Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(VLLMServerSingleton, cls).__new__(cls)
                cls._instance._process = None
        return cls._instance

    def start_server(self, model: str):
        if self._process is not None:
            print("Server is already running.")
            return

        # Start the subprocess and capture the output
        self._process = subprocess.Popen(
            ["vllm", "serve", model],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        # Monitor the output line by line
        while True:
            output = self._process.stdout.readline()
            if output == "" and self._process.poll() is not None:
                break
            if "INFO:     Application startup complete." in output:
                print("Server started successfully.")
                return

        self._process = None
        print("Failed to start the server.")

    def stop_server(self):
        if self._process:
            self._process.terminate()  # Gracefully terminate the process
            try:
                self._process.wait(timeout=5)  # Wait for the process to terminate
            except subprocess.TimeoutExpired:
                self._process.kill()  # Forcefully kill the process if it doesn't terminate
            finally:
                self._process = None
                print("Server stopped.")
        else:
            print("No server is running.")
