"""Manages the llama-server process lifecycle for local model inference."""
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


class LocalServerManager:
    """Starts and stops a llama-server subprocess, exposing an OpenAI-compatible base_url."""

    def __init__(self):
        self._process: subprocess.Popen | None = None
        self._port = int(os.getenv("LLAMA_SERVER_PORT", "8080"))
        self._binary = self._resolve_binary()

        self._n_ctx       = int(os.getenv("LLAMA_SERVER_N_CTX",       "32768"))
        self._n_batch     = int(os.getenv("LLAMA_SERVER_N_BATCH",     "2048"))
        self._n_ubatch    = int(os.getenv("LLAMA_SERVER_N_UBATCH",    "2048"))
        self._flash_attn      = os.getenv("LLAMA_SERVER_FLASH_ATTN", "on").lower()
        self._n_gpu_layers    = int(os.getenv("LLAMA_SERVER_N_GPU_LAYERS", "-1"))
        self._cache_type_k    = os.getenv("LLAMA_SERVER_CACHE_TYPE_K", "f16")
        self._cache_type_v    = os.getenv("LLAMA_SERVER_CACHE_TYPE_V", "f16")
        self._verbose         = os.getenv("LLAMA_SERVER_VERBOSE", "false").lower() == "true"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def base_url(self) -> str:
        return f"http://localhost:{self._port}/v1"

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self, model_path: str, startup_timeout: int = 120) -> None:
        """Start llama-server with the given model. Blocks until /health returns 200."""
        if self.is_running:
            raise RuntimeError("Server is already running. Call stop() first.")

        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        cmd = [
            self._binary,
            "--model",        model_path,
            "--port",         str(self._port),
            "--host",         "127.0.0.1",
            "--ctx-size",     str(self._n_ctx),
            "--batch-size",   str(self._n_batch),
            "--ubatch-size",  str(self._n_ubatch),
            "--n-gpu-layers",   str(self._n_gpu_layers),
            "--cache-type-k",   self._cache_type_k,
            "--cache-type-v",   self._cache_type_v,
        ]
        cmd.extend(["--flash-attn", self._flash_attn])
        if self._verbose:
            cmd.append("--verbose")

        print(f"[LocalServer] Starting: {Path(model_path).name}")
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            # verbose=True → inherit stderr so logs stream to terminal in real time
            # verbose=False → pipe stderr so crash output can be included in the exception
            stderr=None if self._verbose else subprocess.PIPE,
        )
        self._wait_for_health(startup_timeout)
        print(f"[LocalServer] Ready at {self.base_url}")

    def stop(self) -> None:
        """Terminate the server and wait for the process to exit."""
        if not self.is_running:
            return
        self._process.terminate()
        try:
            self._process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()
        self._process = None
        print("[LocalServer] Stopped — VRAM released.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_binary(self) -> str:
        """Locate llama-server: LLAMA_SERVER_PATH env var > PATH search."""
        env_path = os.getenv("LLAMA_SERVER_PATH")
        if env_path:
            p = Path(env_path)
            if p.exists():
                return str(p)
            raise FileNotFoundError(
                f"LLAMA_SERVER_PATH is set to '{env_path}' but the file does not exist."
            )

        found = shutil.which("llama-server")
        if found:
            return found

        raise FileNotFoundError(
            "llama-server binary not found. Either:\n"
            "  • Add it to your PATH, or\n"
            "  • Set LLAMA_SERVER_PATH=/full/path/to/llama-server[.exe] in .env"
        )

    def _wait_for_health(self, timeout: int) -> None:
        health_url = f"http://localhost:{self._port}/health"
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._process.poll() is not None:
                stderr_output = ""
                if self._process.stderr:
                    stderr_output = self._process.stderr.read().decode("utf-8", errors="replace").strip()
                msg = "llama-server exited unexpectedly during startup."
                if stderr_output:
                    msg += f"\n\nllama-server stderr:\n{stderr_output}"
                raise RuntimeError(msg)
            try:
                with urllib.request.urlopen(health_url, timeout=2) as resp:
                    if resp.status == 200:
                        return
            except Exception:
                pass
            time.sleep(1)
        stderr_output = ""
        if self._process.stderr:
            stderr_output = self._process.stderr.read().decode("utf-8", errors="replace").strip()
        self.stop()
        msg = (
            f"llama-server did not become healthy within {timeout}s. "
            "Check that the model path is correct and VRAM is sufficient."
        )
        if stderr_output:
            msg += f"\n\nllama-server stderr:\n{stderr_output}"
        raise TimeoutError(msg)
