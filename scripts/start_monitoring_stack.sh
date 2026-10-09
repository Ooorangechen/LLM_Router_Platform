#!/usr/bin/env bash
# P4 D9: start Prometheus + Grafana + 3 exporters locally (see docs/P4.md §5.1).
# Run from the project root: bash scripts/start_monitoring_stack.sh

cd "$(dirname "$0")/.."

# Reuse existing containers (data kept); only create them on first run.
# If you change docker run flags below, remove the container once by hand: docker rm -f <name>

#    PROMETHEUS (9090)
# monitoring/prometheus.yml keeps localhost targets for a bare-metal Prometheus.
# Inside the container localhost is the container itself, so generate a copy
# whose scrape targets point at the host. Only the scrape_configs section is
# rewritten: prometheus-self (localhost:9090) scrapes the container itself, and
# the alerting target (localhost:9093) is left as-is since no Alertmanager runs.
# Regenerated on every run so template edits reach the reused container too.
sed -e '/^scrape_configs:/,$ {' \
    -e '/localhost:9090/!s/localhost:/host.docker.internal:/' \
    -e '}' \
    monitoring/prometheus.yml > monitoring/prometheus.docker.yml

mkdir -p data/prometheus
# host-gateway makes host.docker.internal resolve on Linux Docker too
# (Docker Desktop on Mac/Windows already provides it).
docker start prom-p4 2>/dev/null || \
docker run -d --name prom-p4 -p 9090:9090 \
  --add-host=host.docker.internal:host-gateway \
  -v $(pwd)/monitoring:/etc/prometheus:ro \
  -v $(pwd)/data/prometheus:/prometheus \
  prom/prometheus:v2.48.0 \
  --config.file=/etc/prometheus/prometheus.docker.yml \
  --storage.tsdb.path=/prometheus \
  --web.enable-lifecycle

#    GRAFANA (3000)
mkdir -p data/grafana && chmod -R 777 data/grafana
docker start graf-p4 2>/dev/null || \
docker run -d --name graf-p4 -p 3000:3000 \
  -e GF_SECURITY_ADMIN_PASSWORD=admin \
  -v $(pwd)/data/grafana:/var/lib/grafana \
  grafana/grafana:10.2.0

#    NODE EXPORTER (9100)
docker start ne-p4 2>/dev/null || \
docker run -d --name ne-p4 -p 9100:9100 \
  quay.io/prometheus/node-exporter:v1.7.0

#    KAFKA EXPORTER (9308)
docker start kafka-exp-p4 2>/dev/null || \
docker run -d --name kafka-exp-p4 -p 9308:9308 \
  --add-host=host.docker.internal:host-gateway \
  danielqsj/kafka-exporter:v1.7.0 \
  --kafka.server=host.docker.internal:9092

#    CLICKHOUSE EXPORTER (9116)
docker start ch-exp-p4 2>/dev/null || \
docker run -d --name ch-exp-p4 -p 9116:9116 \
  --add-host=host.docker.internal:host-gateway \
  f1yegor/clickhouse-exporter \
  -scrape_uri=http://host.docker.internal:8123/

# Wait 20 seconds
sleep 20
docker ps --filter "name=-p4"
echo "Prometheus: http://localhost:9090/targets"
echo "Grafana:    http://localhost:3000 (admin/admin)"
