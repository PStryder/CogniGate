"""Observability infrastructure for CogniGate.

Provides structured logging, distributed tracing, and centralized
observability configuration.
"""

import logging
import os
import sys
from contextvars import ContextVar
from functools import wraps
from typing import Any, Callable, TypeVar

import structlog
from structlog.types import Processor

# Context variables for request/job correlation
current_task_id: ContextVar[str | None] = ContextVar("current_task_id", default=None)
current_lease_id: ContextVar[str | None] = ContextVar("current_lease_id", default=None)
current_worker_id: ContextVar[str | None] = ContextVar("current_worker_id", default=None)
current_trace_id: ContextVar[str | None] = ContextVar("current_trace_id", default=None)
current_span_id: ContextVar[str | None] = ContextVar("current_span_id", default=None)


def add_context_to_log(
    logger: logging.Logger, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Add context variables to log events."""
    if task_id := current_task_id.get():
        event_dict["task_id"] = task_id
    if lease_id := current_lease_id.get():
        event_dict["lease_id"] = lease_id
    if worker_id := current_worker_id.get():
        event_dict["worker_id"] = worker_id
    if trace_id := current_trace_id.get():
        event_dict["trace_id"] = trace_id
    if span_id := current_span_id.get():
        event_dict["span_id"] = span_id
    return event_dict


def configure_logging(
    log_level: str = "INFO",
    json_logs: bool = True,
    service_name: str = "cognigate"
) -> None:
    """Configure structured logging for the application.

    Args:
        log_level: The logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_logs: If True, output JSON formatted logs; if False, use colored console output
        service_name: The service name to include in logs
    """
    # Set base log level
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper(), logging.INFO)
    )

    # Shared processors for both dev and prod
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        add_context_to_log,
    ]

    if json_logs:
        # Production: JSON logs
        processors: list[Processor] = [
            *shared_processors,
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Development: colored console output
        processors = [
            *shared_processors,
            structlog.dev.ConsoleRenderer(colors=True),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Add service name to all logs
    structlog.contextvars.bind_contextvars(service=service_name)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger for the given module name.

    Args:
        name: The logger name (typically __name__)

    Returns:
        A configured structured logger
    """
    return structlog.get_logger(name)


class JobContext:
    """Context manager for job execution context.

    Sets up context variables for structured logging during job execution.
    """

    def __init__(
        self,
        task_id: str,
        lease_id: str,
        worker_id: str,
        profile: str | None = None
    ):
        self.task_id = task_id
        self.lease_id = lease_id
        self.worker_id = worker_id
        self.profile = profile
        self._tokens: list[Any] = []

    def __enter__(self) -> "JobContext":
        self._tokens.append(current_task_id.set(self.task_id))
        self._tokens.append(current_lease_id.set(self.lease_id))
        self._tokens.append(current_worker_id.set(self.worker_id))

        # Also bind to structlog contextvars
        structlog.contextvars.bind_contextvars(
            task_id=self.task_id,
            lease_id=self.lease_id,
            worker_id=self.worker_id,
        )
        if self.profile:
            structlog.contextvars.bind_contextvars(profile=self.profile)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        for token in reversed(self._tokens):
            token.var.reset(token)
        self._tokens.clear()
        structlog.contextvars.unbind_contextvars(
            "task_id", "lease_id", "worker_id", "profile"
        )


class SpanContext:
    """Context manager for tracing spans.

    Creates nested spans for distributed tracing.
    """

    def __init__(self, name: str, **attributes: Any):
        self.name = name
        self.attributes = attributes
        self._parent_span_id: str | None = None
        self._token: Any = None

    def __enter__(self) -> "SpanContext":
        import uuid

        self._parent_span_id = current_span_id.get()
        new_span_id = str(uuid.uuid4())[:16]
        self._token = current_span_id.set(new_span_id)

        logger = get_logger(__name__)
        logger.debug(
            "span_started",
            span_name=self.name,
            span_id=new_span_id,
            parent_span_id=self._parent_span_id,
            **self.attributes
        )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        logger = get_logger(__name__)
        span_id = current_span_id.get()

        if exc_type is not None:
            logger.debug(
                "span_error",
                span_name=self.name,
                span_id=span_id,
                error_type=exc_type.__name__,
                error_message=str(exc_val)
            )
        else:
            logger.debug(
                "span_completed",
                span_name=self.name,
                span_id=span_id
            )

        if self._token:
            current_span_id.reset(self._token)


F = TypeVar("F", bound=Callable[..., Any])


def traced(name: str | None = None) -> Callable[[F], F]:
    """Decorator to add tracing to a function.

    Args:
        name: Optional span name (defaults to function name)
    """
    def decorator(func: F) -> F:
        span_name = name or func.__name__

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            with SpanContext(span_name):
                return await func(*args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            with SpanContext(span_name):
                return func(*args, **kwargs)

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorator


def log_event(
    event: str,
    level: str = "info",
    **kwargs: Any
) -> None:
    """Log a structured event.

    Args:
        event: The event name/description
        level: Log level (debug, info, warning, error, critical)
        **kwargs: Additional event attributes
    """
    logger = get_logger(__name__)
    log_method = getattr(logger, level.lower(), logger.info)
    log_method(event, **kwargs)
