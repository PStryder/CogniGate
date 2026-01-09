"""Circuit breaker implementation for CogniGate.

Provides circuit breaker pattern for external service calls to prevent
cascade failures and allow for graceful degradation.
"""

import asyncio
import time
from enum import Enum
from functools import wraps
from typing import Any, Callable, TypeVar

from .observability import get_logger
from .metrics import set_circuit_breaker_state, record_circuit_breaker_failure


logger = get_logger(__name__)


class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation, requests pass through
    OPEN = "open"  # Circuit tripped, requests fail fast
    HALF_OPEN = "half-open"  # Testing if service recovered


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""

    def __init__(self, name: str, message: str = "Circuit breaker is open"):
        self.name = name
        super().__init__(f"[{name}] {message}")


class CircuitBreaker:
    """Circuit breaker for protecting external service calls.

    When failures exceed the threshold, the circuit opens and fails fast
    for a recovery period before attempting half-open state.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 3,
        excluded_exceptions: tuple[type[Exception], ...] | None = None
    ):
        """Initialize the circuit breaker.

        Args:
            name: Name of this circuit breaker (for logging/metrics)
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            half_open_max_calls: Max calls allowed in half-open state
            excluded_exceptions: Exceptions that don't count as failures
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.excluded_exceptions = excluded_exceptions or ()

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float | None = None
        self._half_open_calls = 0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state

    @property
    def failure_count(self) -> int:
        """Get current failure count."""
        return self._failure_count

    def _update_metrics(self) -> None:
        """Update Prometheus metrics."""
        set_circuit_breaker_state(self.name, self._state.value)

    async def _should_attempt(self) -> bool:
        """Check if a request should be attempted.

        Returns:
            True if the request should proceed, False if it should fail fast
        """
        async with self._lock:
            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                # Check if recovery timeout has passed
                if (
                    self._last_failure_time
                    and time.time() - self._last_failure_time >= self.recovery_timeout
                ):
                    # Transition to half-open
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
                    self._update_metrics()
                    logger.info(
                        "circuit_breaker_half_open",
                        name=self.name,
                        recovery_timeout=self.recovery_timeout
                    )
                    return True
                return False

            if self._state == CircuitState.HALF_OPEN:
                # Allow limited calls in half-open state
                if self._half_open_calls < self.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False

            return False

    async def _record_success(self) -> None:
        """Record a successful call."""
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                # If enough successes in half-open, close the circuit
                if self._success_count >= self.half_open_max_calls:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
                    self._update_metrics()
                    logger.info(
                        "circuit_breaker_closed",
                        name=self.name,
                        reason="recovery_successful"
                    )
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success
                if self._failure_count > 0:
                    self._failure_count = 0

    async def _record_failure(self, error: Exception) -> None:
        """Record a failed call."""
        # Check if this exception should be excluded
        if isinstance(error, self.excluded_exceptions):
            return

        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            record_circuit_breaker_failure(self.name)

            logger.warning(
                "circuit_breaker_failure",
                name=self.name,
                failure_count=self._failure_count,
                threshold=self.failure_threshold,
                error=str(error)
            )

            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open reopens the circuit
                self._state = CircuitState.OPEN
                self._success_count = 0
                self._update_metrics()
                logger.warning(
                    "circuit_breaker_open",
                    name=self.name,
                    reason="half_open_failure"
                )

            elif (
                self._state == CircuitState.CLOSED
                and self._failure_count >= self.failure_threshold
            ):
                # Trip the circuit
                self._state = CircuitState.OPEN
                self._update_metrics()
                logger.warning(
                    "circuit_breaker_open",
                    name=self.name,
                    reason="threshold_exceeded",
                    failure_count=self._failure_count
                )

    async def call(self, func: Callable[..., Any], *args, **kwargs) -> Any:
        """Execute a function through the circuit breaker.

        Args:
            func: The async function to call
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            The function result

        Raises:
            CircuitBreakerError: If the circuit is open
            Exception: Any exception from the function
        """
        if not await self._should_attempt():
            raise CircuitBreakerError(
                self.name,
                f"Circuit is {self._state.value}, failing fast"
            )

        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            await self._record_success()
            return result

        except Exception as e:
            await self._record_failure(e)
            raise

    def reset(self) -> None:
        """Reset the circuit breaker to closed state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None
        self._half_open_calls = 0
        self._update_metrics()
        logger.info("circuit_breaker_reset", name=self.name)


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""

    def __init__(self):
        self._breakers: dict[str, CircuitBreaker] = {}

    def get_or_create(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        **kwargs
    ) -> CircuitBreaker:
        """Get or create a circuit breaker.

        Args:
            name: Circuit breaker name
            failure_threshold: Failure threshold
            recovery_timeout: Recovery timeout in seconds
            **kwargs: Additional circuit breaker arguments

        Returns:
            The circuit breaker instance
        """
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(
                name=name,
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
                **kwargs
            )
        return self._breakers[name]

    def get(self, name: str) -> CircuitBreaker | None:
        """Get a circuit breaker by name."""
        return self._breakers.get(name)

    def list_breakers(self) -> dict[str, dict[str, Any]]:
        """List all circuit breakers with their status."""
        return {
            name: {
                "state": breaker.state.value,
                "failure_count": breaker.failure_count,
                "failure_threshold": breaker.failure_threshold
            }
            for name, breaker in self._breakers.items()
        }

    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        for breaker in self._breakers.values():
            breaker.reset()


# Global registry instance
_registry = CircuitBreakerRegistry()


def get_circuit_breaker_registry() -> CircuitBreakerRegistry:
    """Get the global circuit breaker registry."""
    return _registry


F = TypeVar("F", bound=Callable[..., Any])


def circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0
) -> Callable[[F], F]:
    """Decorator to apply circuit breaker to a function.

    Args:
        name: Circuit breaker name
        failure_threshold: Number of failures before opening
        recovery_timeout: Seconds before attempting recovery

    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        breaker = _registry.get_or_create(
            name,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout
        )

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await breaker.call(func, *args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For sync functions, run in event loop if available
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(breaker.call(func, *args, **kwargs))

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorator
