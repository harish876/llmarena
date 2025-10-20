"""This module provides functionality for executing code in a sandboxed environment."""

import time

import modal
from loguru import logger

PY_LIBRARIES = [
    "pandas",
    "numpy",
    "scikit-learn",
    "sympy",
    "gmpy2",
]


def execute_code(code, lang, exec_timeout=120):
    """Executes code in a sandboxed environment and returns the output.

    Args:
        code (str): The code to execute.
        lang (str): The language of the code. Can be "python" or "cpp".

    Returns:
        str: The output of the code execution, including stdout, stderr, and execution time.
    """
    code_runner = CodeRunner()
    if lang == "python":
        output = code_runner.execute_python_code(code, exec_timeout)
    elif lang == "cpp":
        output = code_runner.execute_cpp_code(code, exec_timeout)
    code_runner.terminate()

    if len(output["stdout"]) > 10000:
        output["stdout"] = output["stdout"][:10000] + "\n...<truncated>\n"
    if len(output["stderr"]) > 10000:
        output["stderr"] = output["stderr"][:10000] + "\n...<truncated>\n"
    if output["time"] > exec_timeout:
        info = f"\n\nExecution time exceeded the timeout of {exec_timeout} seconds."
    else:
        info = f"\n\nExecution time: {output['time']} seconds."
    return "stdout:\n" + output["stdout"] + "\nstderr:\n" + output["stderr"] + "\n" + info


class CodeRunner:
    """A class for running code in a sandboxed environment."""

    def __init__(self, sandbox_timeout=3600, n_retries=3):
        """Initializes the CodeRunner.

        Args:
            sandbox_timeout (int, optional): The timeout for the sandbox in seconds. Defaults to 3600.
            n_retries (int, optional): The number of retries for executing code. Defaults to 3.
        """
        self.app = modal.App.lookup("project-euler-matharena", create_if_missing=True)
        self.sandbox = modal.Sandbox.create(
            image=modal.Image.debian_slim(python_version="3.12").pip_install(PY_LIBRARIES),
            app=self.app,
            timeout=sandbox_timeout,
            block_network=True,
        )
        self.n_exec = 0
        self.n_retries = n_retries

    def execute_python_code(self, code, exec_timeout):
        """Writes Python code to a file, executes it and returns the result.

        Args:
            code (str): The Python code to execute.

        Returns:
            dict: A dictionary containing the stdout, stderr, and execution time.

        Raises:
            Exception: If the code fails to execute after multiple retries.
        """
        for _ in range(self.n_retries):
            try:
                filename = f"pycode_{self.n_exec}.py"

                f = self.sandbox.open(filename, "w")
                f.write(code)
                f.close()

                time_start = time.time()
                p = self.sandbox.exec("bash", "-c", f"python {filename}", timeout=exec_timeout)
                self.n_exec += 1

                output = {
                    "stdout": p.stdout.read(),
                    "stderr": p.stderr.read(),
                }
                output["time"] = time.time() - time_start
                return output
            except Exception as e:
                logger.warning(f"Error executing Python code: {e}")
                time.sleep(1)
        raise Exception("Failed to execute code")

    def execute_cpp_code(self, code, exec_timeout):
        """Writes C++ code to a file, compiles it and returns the result.

        Args:
            code (str): The C++ code to execute.

        Returns:
            dict: A dictionary containing the stdout, stderr, and execution time.

        Raises:
            Exception: If the code fails to execute after multiple retries.
        """
        for _ in range(self.n_retries):
            try:
                filename = f"cppcode_{self.n_exec}.cpp"

                f = self.sandbox.open(filename, "w")
                f.write(code)
                f.close()

                time_start = time.time()
                p = self.sandbox.exec(
                    "bash", "-c", f"g++ {filename} -o {filename}.out && ./{filename}.out", timeout=exec_timeout
                )
                time_end = time.time()
                self.n_exec += 1

                output = {
                    "stdout": p.stdout.read(),
                    "stderr": p.stderr.read(),
                }
                output["time"] = time_end - time_start
                return output
            except Exception as e:
                logger.warning(f"Error executing C++ code: {e}")
                time.sleep(1)
        raise Exception("Failed to execute code")

    def terminate(self):
        """Terminates the sandbox."""
        self.sandbox.terminate()
