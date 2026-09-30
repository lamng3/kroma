"""Small decorators used by the matching run."""

import time
from functools import wraps

TRANSIENT_ERROR_NAMES = {
    "APIConnectionError",
    "APITimeoutError",
    "RateLimitError",
    "InternalServerError",
    "ServiceUnavailableError",
}


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return True
    return type(exc).__name__ in TRANSIENT_ERROR_NAMES


def retry(attempts: int = 3, backoff: float = 0.5):
    """Retry a call when the provider reports a transient failure."""

    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            delay = backoff
            for attempt in range(attempts):
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:
                    if not _is_transient(exc) or attempt == attempts - 1:
                        raise
                    time.sleep(delay)
                    delay *= 2
        return wrapper

    return decorator


def timed(listener, bucket: str):
    """Record how long a call takes on a metrics listener."""

    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            started = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                listener.add_timing(bucket, time.perf_counter() - started)
        return wrapper

    return decorator
