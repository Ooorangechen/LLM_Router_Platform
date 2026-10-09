#!/usr/bin/env bash
# P2: start Redis locally for the inference response cache (config: inference.cache, localhost:6379).
# Run from the project root: bash scripts/start_redis.sh

cd "$(dirname "$0")/.."

# Reuse existing containers (data kept); only create them on first run.
# If you change docker run flags below, remove the container once by hand: docker rm -f <name>

#    REDIS (6379)
docker start redis-p2 2>/dev/null || \
docker run -d --name redis-p2 -p 6379:6379 \
  redis:7-alpine

# Wait until Redis answers PING
sleep 3
docker exec redis-p2 redis-cli ping
