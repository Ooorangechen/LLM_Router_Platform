"""P3 M4 / task 3.4: ClickHouse batch write, retry, local DLQ and replay-dlq, with a fake client."""

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import pytest

from src.llm_router_part3_pipeline import ClickHouseWriter
from src.utils.constants import ClickHouseTables


pytestmark = pytest.mark.asyncio

TABLE = ClickHouseTables.SYSTEM_METRICS.value
ROWS = [{"metric_name": "a", "value": 1.0}, {"metric_name": "b", "value": 2.0}]


@pytest.fixture
def dlq_dir(tmp_path):
    return tmp_path / "dlq"


@pytest.fixture
def writer(dlq_dir):
    config = {
        "pipeline": {"enabled": True, "dlq_local_dir": str(dlq_dir)},
        # Zero backoff keeps the retry loop instant.
        "clickhouse": {"batch_size": 2, "retry_max": 3, "retry_backoff_base_ms": 0},
    }
    result = ClickHouseWriter(config)
    result.client = Mock()
    return result


def _dlq_entries(dlq_dir):
    return [json.loads(line)
            for path in sorted(dlq_dir.glob("[0-9]*.jsonl"))
            for line in path.read_text(encoding="utf-8").splitlines()]


async def test_buffer_flushes_at_batch_size_as_json_each_row(writer, dlq_dir):
    assert await writer.buffer_write(TABLE, ROWS[0]) == (0, 0)
    writer.client.raw_insert.assert_not_called()

    assert await writer.buffer_write(TABLE, ROWS[1]) == (2, 0)

    args, kwargs = writer.client.raw_insert.call_args
    assert args == (TABLE,)
    assert kwargs["fmt"] == "JSONEachRow"
    assert [json.loads(line) for line in kwargs["insert_block"].splitlines()] == ROWS
    assert not dlq_dir.exists() or _dlq_entries(dlq_dir) == []


async def test_transient_failure_is_retried_without_dlq(writer, dlq_dir):
    writer.client.raw_insert.side_effect = [ConnectionError("blip"), None]
    writer._buffers[TABLE] = list(ROWS)

    assert await writer.flush_table(TABLE) == (2, 0)
    assert writer.client.raw_insert.call_count == 2
    assert not dlq_dir.exists() or _dlq_entries(dlq_dir) == []


async def test_persistent_failure_retries_then_lands_in_local_dlq(writer, dlq_dir):
    writer.client.raw_insert.side_effect = ConnectionError("clickhouse down")
    writer._buffers[TABLE] = list(ROWS)

    # Rows safely in the DLQ are recoverable, so they are not reported as failed.
    assert await writer.flush_table(TABLE) == (0, 0)
    assert writer.client.raw_insert.call_count == writer.retry_max
    assert writer._buffers[TABLE] == []
    [entry] = _dlq_entries(dlq_dir)
    assert entry["table"] == TABLE
    assert entry["rows"] == ROWS
    assert "clickhouse down" in entry["reason"]


async def test_failure_with_unwritable_dlq_reports_rows_as_failed(writer, dlq_dir):
    writer.client.raw_insert.side_effect = ConnectionError("clickhouse down")
    dlq_dir.parent.mkdir(parents=True, exist_ok=True)
    dlq_dir.write_text("a file where the DLQ directory should be")
    writer._buffers[TABLE] = list(ROWS)

    assert await writer.flush_table(TABLE) == (0, 2)


async def test_replay_reinserts_and_archives_dlq_files(writer, dlq_dir):
    writer.client.raw_insert.side_effect = ConnectionError("clickhouse down")
    writer._buffers[TABLE] = list(ROWS)
    await writer.flush_table(TABLE)
    [dlq_file] = list(dlq_dir.glob("[0-9]*.jsonl"))

    writer.client.raw_insert.reset_mock(side_effect=True)
    assert await writer.replay_dlq() == (2, 0)

    writer.client.raw_insert.assert_called_once()
    assert not dlq_file.exists()
    assert (dlq_dir / "archived" / dlq_file.name).exists()


async def test_replay_keeps_partially_failed_file_and_skips_other_files(writer, dlq_dir):
    dlq_dir.mkdir(parents=True)
    hour = datetime.now(timezone.utc).strftime("%Y%m%d_%H")
    mixed = dlq_dir / f"{hour}.jsonl"
    mixed.write_text(
        json.dumps({"table": TABLE, "rows": ROWS, "reason": "x"}) + "\n"
        + json.dumps({"table": "no_such_table", "rows": ROWS, "reason": "x"}) + "\n",
        encoding="utf-8")
    old_hour = (datetime.now(timezone.utc) - timedelta(days=8)).strftime("%Y%m%d_%H")
    (dlq_dir / f"{old_hour}.jsonl").write_text(
        json.dumps({"table": TABLE, "rows": ROWS, "reason": "x"}) + "\n", encoding="utf-8")
    # Raw Kafka payloads have no target table and are never replayed.
    (dlq_dir / f"kafka_{hour}.jsonl").write_text('{"original_topic": "llm-metrics"}\n')

    assert await writer.replay_dlq() == (2, 1)

    writer.client.raw_insert.assert_called_once()  # only the valid line of the recent file
    assert mixed.exists()                          # partial failure stays for the next replay
    assert list((dlq_dir / "archived").iterdir()) == []
