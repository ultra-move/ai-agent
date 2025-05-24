import multiprocessing
import sys
import traceback
import io
import contextlib
from langchain_core.tools import tool

class Sandbox:
    def __init__(self, timeout=60):
        """
        Initializes the executor with a timeout.

        Args:
            timeout (int): Maximum time (in seconds) to allow code execution. The default is 60 seconds.
        """
        self.timeout = timeout

    def _run_code(self, code, output_queue):
        """
        Internal function to execute code and capture output/errors.
        """
        stdout = io.StringIO()
        stderr = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                try:
                    result = eval(code, {})
                    if result is not None:
                        print(result)
                except SyntaxError:
                    exec(code, {})
            output_queue.put({"output": stdout.getvalue().strip(), "error": stderr.getvalue().strip()})
        except Exception:
            output_queue.put({"error": traceback.format_exc().strip()})
    
    def execute(self, code: str) -> dict:
        """
        Executes the given code in a separate process and returns output or errors.

        Args:
            code (str): Python code to execute.

        Returns:
            dict: {'output': str} or {'error': str}
        """
        output_queue = multiprocessing.Queue()
        process = multiprocessing.Process(target=self._run_code, args=(code, output_queue))
        process.start()
        process.join(timeout=self.timeout)

        if process.is_alive():
            process.terminate()
            return {"error": "Execution timed out."}
        if not output_queue.empty():
            result = output_queue.get()
            if result.get("error"):
                return {"error": result["error"]}
            return {"output": result.get("output", "")}
        return {"error": "No output received."}
