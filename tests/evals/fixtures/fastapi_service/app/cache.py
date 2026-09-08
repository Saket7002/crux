"""Redis is already wired up here and used for short-lived counters."""

import redis.asyncio as redis

client = redis.Redis(host="localhost", port=6379, decode_responses=True)


async def incr(key: str, ttl: int) -> int:
    value = await client.incr(key)
    await client.expire(key, ttl)
    return value
