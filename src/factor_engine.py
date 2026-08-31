"""
factor_engine.py
Safely execute LLM-generated factor computation code, computing factor values
window-by-window for windows [N, seq_length, D] tensor,
outputting a numpy float32 array of length N.

Execution context (variables available in LLM code):
  window      : np.ndarray [seq_length, D]  — current window feature sequence (float32)
  idx         : dict                         — {attribute_name: int_index}, access with window[:, idx['xxx']]
  np          : numpy
"""

import threading
import traceback
import logging
import numpy as np
from typing import Optional

# Global lock: Python's exec()/compile() is not thread-safe, concurrent calls can deadlock
_exec_lock = threading.Lock()

logger = logging.getLogger(__name__)

# Whitelisted builtins for the sandbox
_SAFE_BUILTINS = {
    "abs": abs, "min": min, "max": max, "sum": sum,
    "len": len, "range": range, "enumerate": enumerate,
    "zip": zip, "list": list, "dict": dict, "tuple": tuple,
    "float": float, "int": int, "bool": bool, "str": str,
    "isinstance": isinstance, "hasattr": hasattr,
    "print": print,   # debugging
    "__import__": __import__,  # Required by NumPy C extensions internally (lazy submodule loading)
}


class FactorEngine:
    """
    Execute LLM-generated factor code on windowed data.

    compute_factor(hypothesis, windows, flat_attributes) -> np.ndarray [N] or None

    windows        : np.ndarray [N, seq_length, D]  (float32)
    flat_attributes: list[str]                       (from MBD.feature_indexer.flat_attributes)
    """

    def __init__(self, cfg: dict):
        self.max_error_ratio = cfg.get("factor_engine", {}).get("max_error_ratio", 0.1)
        self.min_valid_ratio = cfg.get("factor_engine", {}).get("min_valid_ratio", 0.5)
        self._compile_cache: dict = {}       # code_hash → compiled function (row)
        self._batch_compile_cache: dict = {}  # code_hash → compiled function (batch)
        self._compile_lock = threading.Lock()  # thread safety

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def compute_factor(
        self,
        hypothesis: dict,
        windows: np.ndarray,
        flat_attributes: list,
    ) -> Optional[np.ndarray]:
        """
        Execute factor code and return a float32 array of length N, or None on failure.

        Parameters
        ----------
        hypothesis      : dict, containing 'name' and 'code' keys
        windows         : np.ndarray [N, seq_length, D], each row is a window of feature sequences
        flat_attributes : list[str], length D, column names for the last dimension of windows
        """
        code = hypothesis["code"]
        factor_name = hypothesis["name"]
        N = windows.shape[0]

        name_to_idx: dict = {name: i for i, name in enumerate(flat_attributes)}

        try:
            func = self._compile_factor_func(code, factor_name)
        except Exception as e:
            logger.warning(f"[FactorEngine] Factor '{factor_name}' compilation failed: {e}")
            return None

        values = np.full(N, np.nan, dtype=np.float32)
        errors = 0
        max_errors = max(10, int(N * self.max_error_ratio))

        for i in range(N):
            try:
                val = func(windows[i], name_to_idx, np)
                if val is not None:
                    values[i] = float(val)
            except Exception:
                errors += 1
                if errors > max_errors:
                    logger.warning(
                        f"[FactorEngine] Factor '{factor_name}' has too many error windows (>{max_errors}), skipping."
                    )
                    return None

        if errors > 0:
            logger.debug(f"[FactorEngine] Factor '{factor_name}' had {errors} error windows (set to nan)")

        values[~np.isfinite(values)] = np.nan

        nan_ratio = np.isnan(values).mean()
        if nan_ratio > (1 - self.min_valid_ratio):
            logger.warning(
                f"[FactorEngine] Factor '{factor_name}' insufficient valid values ({1-nan_ratio:.1%}), skipping."
            )
            return None

        return values

    def compute_factor_batch(
        self,
        hypothesis: dict,
        windows: np.ndarray,
        flat_attributes: list,
    ) -> Optional[np.ndarray]:
        """
        Vectorized version (suitable for LLM-written full-data operations).
        Variables available in LLM code:
          windows : np.ndarray [N, seq_length, D]
          idx     : dict
          np      : numpy
        Code must assign the result to _result (shape [N] or scalar).
        """
        code = hypothesis["code"]
        factor_name = hypothesis["name"]
        name_to_idx: dict = {name: i for i, name in enumerate(flat_attributes)}

        # Compilation cache: exec() same code only once, reuse subsequently
        import hashlib
        code_hash = hashlib.md5(code.encode()).hexdigest()
        with self._compile_lock:
            func = self._batch_compile_cache.get(code_hash)
        if func is None:
            func_code = "def _factor_func(windows, idx, np):\n"
            func_code += "    with np.errstate(invalid='ignore', divide='ignore'):\n"
            for line in code.strip().split("\n"):
                func_code += f"        {line}\n"
            func_code += "        return _result\n"

            namespace = {"__builtins__": _SAFE_BUILTINS}
            with _exec_lock:
                exec(compile(func_code, "<factor_batch>", "exec"), namespace)
            func = namespace["_factor_func"]
            with self._compile_lock:
                self._batch_compile_cache[code_hash] = func

        try:
            result = func(windows, name_to_idx, np)
        except Exception as e:
            logger.warning(f"[FactorEngine] Batch factor '{factor_name}' execution failed: {e}")
            return None

        if result is None:
            return None

        result = np.asarray(result, dtype=np.float32)
        if result.ndim == 0:
            result = np.full(len(windows), float(result), dtype=np.float32)

        result[~np.isfinite(result)] = np.nan

        nan_ratio = np.isnan(result).mean()
        if nan_ratio > (1 - self.min_valid_ratio):
            logger.warning(f"[FactorEngine] Batch factor '{factor_name}' insufficient valid values, skipping.")
            return None

        return result

    # ------------------------------------------------------------------
    # Private: compile factor function
    # ------------------------------------------------------------------
    def _compile_factor_func(self, code: str, factor_name: str):
        """
        Wrap LLM-generated code body into a function and compile, returning a callable.
        Uses caching to avoid recompiling identical code.

        Signature: _factor_func(window: np.ndarray[seq_length, D], idx: dict, np) -> float | None
        feat = window[-1] is a compatibility alias for formula-converted factor code.
        """
        import hashlib
        code_hash = hashlib.md5(code.encode()).hexdigest()

        with self._compile_lock:
            if code_hash in self._compile_cache:
                return self._compile_cache[code_hash]

        func_code = "def _factor_func(window, idx, np):\n"
        func_code += "    feat = window[-1]\n"  # compat: formula-converted factors use feat[idx[...]]
        for line in code.strip().split("\n"):
            func_code += f"    {line}\n"

        namespace = {"__builtins__": _SAFE_BUILTINS}
        with _exec_lock:
            exec(compile(func_code, f"<factor:{factor_name}>", "exec"), namespace)  # noqa: S102
        func = namespace["_factor_func"]

        with self._compile_lock:
            self._compile_cache[code_hash] = func
        return func
