"""Rate limiting middleware for CogniGate API."""

import asyncio
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Protocol, Tuple

from fastapi import HTTPException, Request, status

from ..observability import get_logger


logger = get_logger(__name__)


class RateLimitBackend(ABC):
    """Abstract base class for rate limit backends."""

    @abstractmethod
    async def check_rate_limit(
        self, key: str, max_calls: int, window_seconds: int
    ) -> Tuple[bool, int, int]:
        """Check rate limit.

        Args:
            key: The rate limit key (e.g., client IP)
            max_calls: Maximum calls allowed in window
            window_seconds: Window size in seconds

        Returns:
            Tuple of (allowed, remaining, reset_time)
        """
        pass


class InMemoryRateLimiter(RateLimitBackend):
    """In-memory rate limiter using sliding window.

    Note: This implementation is per-instance and doesn't work
    across multiple workers. Use RedisRateLimiter for distributed
    rate limiting.
    """

    def __init__(self):
        self._windows: dict[str, list[float]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def check_rate_limit(
        self, key: str, max_calls: int, window_seconds: int
    ) -> Tuple[bool, int, int]:
        """Check rate limit using sliding window."""
        async with self._lock:
            now = time.time()
            window_start = now - window_seconds
            self._windows[key] = [ts for ts in self._windows[key] if ts > window_start]
            current_calls = len(self._windows[key])
            allowed = current_calls < max_calls
            remaining = max(0, max_calls - current_calls - (1 if allowed else 0))
            reset_time = int(self._windows[key][0] + window_seconds) if self._windows[key] else int(now + window_seconds)
            if allowed:
                self._windows[key].append(now)
            return allowed, remaining, reset_time


class RedisRateLimiter(RateLimitBackend):
    """Redis-backed rate limiter for distributed rate limiting.

    Uses a sliding window algorithm with Redis sorted sets.
    """

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self._redis = None
        self._initialized = False

    async def _ensure_connected(self) -> None:
        """Ensure Redis connection is established."""
        if self._initialized:
            return

        try:
            import redis.asyncio as aioredis
            self._redis = aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            # Test connection
            await self._redis.ping()
            self._initialized = True
            logger.info("redis_rate_limiter_connected", url=self.redis_url)
        except ImportError:
            logger.error("redis_not_installed", message="redis package not installed")
            raise RuntimeError("redis package is required for RedisRateLimiter")
        except Exception as e:
            logger.error("redis_connection_failed", error=str(e))
            raise

    async def check_rate_limit(
        self, key: str, max_calls: int, window_seconds: int
    ) -> Tuple[bool, int, int]:
        """Check rate limit using Redis sorted set sliding window."""
        await self._ensure_connected()

        now = time.time()
        window_start = now - window_seconds
        redis_key = f"ratelimit:{key}"

        # Use a Redis pipeline for atomic operations
        async with self._redis.pipeline(transaction=True) as pipe:
            # Remove expired entries
            pipe.zremrangebyscore(redis_key, "-inf", window_start)
            # Count current entries
            pipe.zcard(redis_key)
            # Execute
            results = await pipe.execute()

        current_calls = results[1]
        allowed = current_calls < max_calls

        if allowed:
            # Add new entry with current timestamp as score
            await self._redis.zadd(redis_key, {str(now): now})
            # Set expiry on the key
            await self._redis.expire(redis_key, window_seconds + 1)

        remaining = max(0, max_calls - current_calls - (1 if allowed else 0))

        # Get the oldest entry for reset time calculation
        oldest = await self._redis.zrange(redis_key, 0, 0, withscores=True)
        if oldest:
            reset_time = int(oldest[0][1] + window_seconds)
        else:
            reset_time = int(now + window_seconds)

        return allowed, remaining, reset_time

    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._initialized = False


class RateLimiter:
    """Main rate limiter with configurable backend."""

    def __init__(
        self,
        calls_per_minute: int = 50,
        enabled: bool = True,
        backend: RateLimitBackend | None = None,
        redis_url: str | None = None
    ):
        self.calls_per_minute = calls_per_minute
        self.enabled = enabled

        # Use Redis backend if URL provided, otherwise in-memory
        if redis_url:
            self.backend = RedisRateLimiter(redis_url)
            logger.info("rate_limiter_using_redis")
        elif backend:
            self.backend = backend
        else:
            self.backend = InMemoryRateLimiter()
            logger.info("rate_limiter_using_memory")

    async def check_request(self, request: Request) -> None:
        """Check if request should be rate limited."""
        if not self.enabled:
            return

        client_ip = request.client.host if request.client else "unknown"
        key = f"ip:{client_ip}"

        try:
            allowed, remaining, reset_time = await self.backend.check_rate_limit(
                key, self.calls_per_minute, 60
            )
        except Exception as e:
            # On backend failure, log and allow request (fail open)
            logger.error("rate_limit_check_failed", error=str(e))
            return

        if not allowed:
            retry_after = reset_time - int(time.time())
            logger.warning(
                "rate_limit_exceeded",
                key=key,
                retry_after=retry_after
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Retry after {retry_after} seconds.",
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(self.calls_per_minute),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(reset_time)
                },
            )


_rate_limiter: RateLimiter | None = None


def get_rate_limiter(
    calls_per_minute: int,
    enabled: bool,
    redis_url: str | None = None
) -> RateLimiter:
    """Get rate limiter singleton.

    Args:
        calls_per_minute: Maximum requests per minute
        enabled: Whether rate limiting is enabled
        redis_url: Optional Redis URL for distributed rate limiting

    Returns:
        RateLimiter instance
    """
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter(
            calls_per_minute=calls_per_minute,
            enabled=enabled,
            redis_url=redis_url
        )
    return _rate_limiter


def reset_rate_limiter() -> None:
    """Reset the rate limiter singleton (for testing)."""
    global _rate_limiter
    _rate_limiter = None
