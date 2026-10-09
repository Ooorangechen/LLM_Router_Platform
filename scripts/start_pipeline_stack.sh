#!/usr/bin/env bash
# P3: start Kafka + ClickHouse locally (see docs/P3.md §5.1).
# Run from the project root: bash scripts/start_pipeline_stack.sh

cd "$(dirname "$0")/.."

# Reuse existing containers (data kept); only create them on first run.
# If you change docker run flags below, remove the container once by hand: docker rm -f <name>

#    KAFKA (9092), data persisted in data/kafka (bitnami runs as uid 1001, hence 777)
mkdir -p data/kafka && chmod -R 777 data/kafka
docker start kafka-p3 2>/dev/null || \
docker run -d --name kafka-p3 -p 9092:9092 \
  -v $(pwd)/data/kafka:/bitnami/kafka \
  -e KAFKA_CFG_NODE_ID=0 \
  -e KAFKA_CFG_PROCESS_ROLES=broker,controller \
  -e KAFKA_CFG_CONTROLLER_QUORUM_VOTERS=0@localhost:9093 \
  -e KAFKA_CFG_LISTENERS=PLAINTEXT://:9092,CONTROLLER://:9093 \
  -e KAFKA_CFG_ADVERTISED_LISTENERS=PLAINTEXT://localhost:9092 \
  -e KAFKA_CFG_LISTENER_SECURITY_PROTOCOL_MAP=CONTROLLER:PLAINTEXT,PLAINTEXT:PLAINTEXT \
  -e KAFKA_CFG_CONTROLLER_LISTENER_NAMES=CONTROLLER \
  -e KAFKA_CFG_AUTO_CREATE_TOPICS_ENABLE=true \
  bitnami/kafka:3.6

#    CLICKHOUSE (8123 HTTP, 9000 native), data persisted in clickhouse/data (already in .gitignore)
mkdir -p clickhouse/data
docker start ch-p3 2>/dev/null || \
docker run -d --name ch-p3 -p 8123:8123 -p 9000:9000 \
  -v $(pwd)/clickhouse/data:/var/lib/clickhouse \
  -e CLICKHOUSE_DB=default \
  -e CLICKHOUSE_USER=default \
  -e CLICKHOUSE_DEFAULT_ACCESS_MANAGEMENT=1 \
  clickhouse/clickhouse-server:23.8-alpine

# Wait 20 seconds
sleep 10
docker ps --filter "name=-p3"
echo "First run only: python main.py init-kafka-topics  (and init ClickHouse schema, see docs/P3.md §5.1)"
