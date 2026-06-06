"""
Sandboxed Python executor.
Runs untrusted code in a subprocess with a hard timeout and (best-effort)
resource limits. Returns stdout, stderr, exit code, and elapsed time.

Isolation provided:
  - Separate subprocess (your interpreter's globals are never exposed)
  - Hard wall-clock timeout (kills the whole process group, not just the parent)
  - Optional network block, CPU-time cap, and memory cap on POSIX (via `resource`)
  - Optional stripped environment

NOT provided (do not run genuinely hostile code without more):
  - Filesystem isolation — use containers / seccomp / a jail for that.
  - On Windows, `resource` is unavailable, so CPU/memory caps are skipped
    (the wall-clock timeout still applies). This is documented, not silent.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path

# `resource` is POSIX-only; absent on Windows. We degrade gracefully.
try:
    import resource  # type: ignore
    _HAS_RESOURCE = True
except ImportError:  # pragma: no cover - platform dependent
    _HAS_RESOURCE = False

_IS_POSIX = os.name == "posix"


@dataclass
class RunResult:
    stdout: str
    stderr: str
    exit_code: int
    elapsed_seconds: float
    timed_out: bool
    # New (additive, defaulted so existing callers/positional construction still work):
    killed_signal: int | None = None
    limits_applied: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def summary(self) -> str:
        status = "PASS" if self.success else ("TIMEOUT" if self.timed_out else "FAIL")
        lines = [f"[{status}] exit={self.exit_code} time={self.elapsed_seconds:.2f}s"]
        if self.killed_signal is not None:
            lines.append(f"(killed by signal {self.killed_signal})")
        if self.limits_applied:
            lines.append(f"limits: {', '.join(self.limits_applied)}")
        if self.stdout.strip():
            lines.append("--- stdout ---")
            lines.append(self.stdout[:2000])
        if self.stderr.strip():
            lines.append("--- stderr ---")
            lines.append(self.stderr[:2000])
        return "\n".join(lines)


def _build_preexec(
    cpu_seconds: int | None,
    memory_mb: int | None,
    limits_applied: list[str],
):
    """
    Return a preexec_fn (POSIX only) that, in the child *before* exec:
      - starts a new session so we can kill the whole process group on timeout
      - applies CPU-time and address-space (memory) rlimits if requested
    Returns None on platforms where this isn't supported.
    """
    if not _IS_POSIX:
        return None

    def _preexec():  # pragma: no cover - runs in child process
        os.setsid()  # new process group → killable as a unit
        if _HAS_RESOURCE:
            if cpu_seconds is not None:
                resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
            if memory_mb is not None:
                nbytes = memory_mb * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (nbytes, nbytes))

    if cpu_seconds is not None:
        limits_applied.append(f"cpu={cpu_seconds}s")
    if memory_mb is not None:
        limits_applied.append(f"mem={memory_mb}MB")
    return _preexec


def _make_env(block_network: bool, clean_env: bool) -> dict:
    """Construct the child environment."""
    env = {} if clean_env else dict(os.environ)
    # Ensure the child can still find Python's own machinery when clean_env is set.
    if clean_env:
        for key in ("PATH", "SYSTEMROOT", "PYTHONPATH", "LANG", "LC_ALL"):
            if key in os.environ:
                env[key] = os.environ[key]
    if block_network:
        # Many libraries honour these; this is a courtesy block, not a firewall.
        env["http_proxy"] = env["https_proxy"] = "http://127.0.0.1:9"
        env["HTTP_PROXY"] = env["HTTPS_PROXY"] = "http://127.0.0.1:9"
        env["no_proxy"] = ""
    return env


def _run_subprocess(
    argv: list[str],
    cwd: str | None,
    timeout: float,
    cpu_seconds: int | None,
    memory_mb: int | None,
    block_network: bool,
    clean_env: bool,
) -> RunResult:
    """Shared subprocess runner with timeout, process-group kill, and limits."""
    limits_applied: list[str] = []
    preexec = _build_preexec(cpu_seconds, memory_mb, limits_applied)
    if block_network:
        limits_applied.append("network=blocked(env)")
    env = _make_env(block_network, clean_env)

    start = time.monotonic()
    proc = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd,
        env=env,
        preexec_fn=preexec,  # None on Windows → ignored
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        elapsed = time.monotonic() - start
        return RunResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=proc.returncode,
            elapsed_seconds=elapsed,
            timed_out=False,
            killed_signal=(-proc.returncode if proc.returncode and proc.returncode < 0 else None),
            limits_applied=limits_applied,
        )
    except subprocess.TimeoutExpired:
        # Kill the whole process group on POSIX (children too); fall back to the
        # single process elsewhere. Then reap so we don't leak a zombie.
        _kill_tree(proc)
        try:
            stdout, stderr = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            stdout, stderr = "", ""
        return RunResult(
            stdout=stdout or "",
            stderr=(stderr or "") + f"\nTimed out after {timeout}s",
            exit_code=-1,
            elapsed_seconds=timeout,
            timed_out=True,
            limits_applied=limits_applied,
        )


def _kill_tree(proc: subprocess.Popen) -> None:
    """Terminate a process and (on POSIX) its whole session group."""
    try:
        if _IS_POSIX:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        else:
            proc.kill()
    except (ProcessLookupError, PermissionError, OSError):
        # Already gone, or we lack permission — nothing more to do.
        pass


def run_code(
    code: str,
    timeout: float = 10.0,
    *,
    cpu_seconds: int | None = None,
    memory_mb: int | None = None,
    block_network: bool = False,
    clean_env: bool = False,
) -> RunResult:
    """
    Execute a Python code string in an isolated subprocess.

    Backward compatible: `run_code(code)` and `run_code(code, timeout)` behave
    exactly as before. The new keyword-only options add stronger isolation:

        cpu_seconds:    hard CPU-time cap (POSIX only; ignored on Windows)
        memory_mb:      hard address-space cap (POSIX only; ignored on Windows)
        block_network:  set proxy env vars to a dead address (best-effort)
        clean_env:      run with a stripped environment (no leaked secrets)
    """
    if not isinstance(code, str) or not code.strip():
        return RunResult("", "run_code: empty or non-string code", -1, 0.0, False)

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False, encoding="utf-8") as f:
            f.write(textwrap.dedent(code))
            tmp_path = f.name

        return _run_subprocess(
            [sys.executable, "-I", tmp_path],  # -I = isolated mode (ignore env/user site)
            cwd=None,
            timeout=timeout,
            cpu_seconds=cpu_seconds,
            memory_mb=memory_mb,
            block_network=block_network,
            clean_env=clean_env,
        )
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


def run_pytest(
    test_code: str,
    source_code: str,
    timeout: float = 30.0,
    *,
    cpu_seconds: int | None = None,
    memory_mb: int | None = None,
    block_network: bool = False,
) -> RunResult:
    """
    Write source + test to a temp directory, run pytest, return results.
      source_code → solution.py
      test_code   → test_solution.py   (a `from solution import *` is prepended)

    Backward compatible with the original two-positional-arg signature. The new
    keyword-only options apply the same isolation as `run_code`.
    """
    if not isinstance(source_code, str) or not isinstance(test_code, str):
        return RunResult("", "run_pytest: source/test must be strings", -1, 0.0, False)

    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "solution.py"
        test_path = Path(tmpdir) / "test_solution.py"
        try:
            src_path.write_text(textwrap.dedent(source_code), encoding="utf-8")
            full_test = "from solution import *  # noqa: F401,F403\n\n" + textwrap.dedent(test_code)
            test_path.write_text(full_test, encoding="utf-8")
        except OSError as e:
            return RunResult("", f"run_pytest: could not write temp files: {e}", -1, 0.0, False)

        return _run_subprocess(
            [sys.executable, "-m", "pytest", str(test_path), "-v", "--tb=short", "-p", "no:cacheprovider"],
            cwd=tmpdir,
            timeout=timeout,
            cpu_seconds=cpu_seconds,
            memory_mb=memory_mb,
            block_network=block_network,
        )


if __name__ == "__main__":
    print(run_code("print('hello from sandbox')").summary())
    print()
    print(run_code("x = 1/0").summary())
    print()
    # Demonstrate the timeout + process-group kill on an infinite loop.
    print(run_code("while True: pass", timeout=2.0, cpu_seconds=2).summary())
