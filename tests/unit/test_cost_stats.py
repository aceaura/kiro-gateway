# -*- coding: utf-8 -*-

"""
Unit tests for per-request credit cost accounting.

Covers effort classification, defensive value coercion, the aggregate store,
and the record_request_cost join point.
"""

import json

import pytest


# ==================================================================================================
# Effort Classification
# ==================================================================================================

class TestClassifyEffort:
    """Tests for classify_effort()."""

    def test_thinking_disabled_is_none_effort(self):
        """
        What it does: Returns "none" when the client disabled thinking.
        Goal: A disabled-thinking request must never be grouped with reasoning runs.
        """
        from kiro.cost_stats import classify_effort

        print("Action: Classifying with thinking_enabled=False...")
        result = classify_effort(thinking_budget=8000, max_tokens=10000, thinking_enabled=False)

        print(f"Result: {result}")
        assert result == "none"

    def test_no_budget_is_default_effort(self):
        """
        What it does: Returns "default" when thinking is on with no explicit budget.
        Goal: Distinguish "server default depth" from an explicit client choice.
        """
        from kiro.cost_stats import classify_effort

        print("Action: Classifying with thinking_budget=None...")
        result = classify_effort(thinking_budget=None, max_tokens=4096)

        print(f"Result: {result}")
        assert result == "default"

    def test_zero_budget_is_none_effort(self):
        """
        What it does: Treats a zero budget as no reasoning.
        Goal: A 0-token budget is functionally thinking-off.
        """
        from kiro.cost_stats import classify_effort

        print("Action: Classifying budget=0...")
        assert classify_effort(thinking_budget=0, max_tokens=4096) == "none"

    def test_negative_budget_is_none_effort(self):
        """
        What it does: Treats a negative budget as no reasoning.
        Goal: Malformed negative budgets must not fall through to a high label.
        """
        from kiro.cost_stats import classify_effort

        print("Action: Classifying budget=-100...")
        assert classify_effort(thinking_budget=-100, max_tokens=4096) == "none"

    @pytest.mark.parametrize("effort,percent", [
        ("minimal", 0.10),
        ("low", 0.20),
        ("medium", 0.50),
        ("high", 0.80),
        ("xhigh", 0.95),
    ])
    def test_round_trips_reasoning_effort_to_budget(self, effort, percent):
        """
        What it does: Recovers the original effort label from a generated budget.
        Goal: classify_effort must invert reasoning_effort_to_budget so OpenAI and
              Anthropic requests land in the same aggregate buckets.
        """
        from kiro.cost_stats import classify_effort
        from kiro.converters_openai import reasoning_effort_to_budget

        max_tokens = 10000
        budget = reasoning_effort_to_budget(max_tokens, effort)

        print(f"Setup: effort={effort} percent={percent} -> budget={budget}")
        result = classify_effort(thinking_budget=budget, max_tokens=max_tokens)

        print(f"Result: {result}")
        assert result == effort

    def test_round_trips_at_small_max_tokens(self):
        """
        What it does: Round-trips effort labels at a small output limit.
        Goal: Ratio classification must hold when integer truncation is relatively large.
        """
        from kiro.cost_stats import classify_effort
        from kiro.converters_openai import reasoning_effort_to_budget

        for effort in ("minimal", "low", "medium", "high", "xhigh"):
            budget = reasoning_effort_to_budget(4096, effort)
            result = classify_effort(thinking_budget=budget, max_tokens=4096)
            print(f"effort={effort} budget={budget} -> {result}")
            assert result == effort

    def test_ratio_above_all_thresholds_is_xhigh(self):
        """
        What it does: Classifies a budget exceeding max_tokens as "xhigh".
        Goal: Ratios above 1.0 must saturate instead of falling through to None.
        """
        from kiro.cost_stats import classify_effort

        print("Action: Classifying budget greater than max_tokens...")
        result = classify_effort(thinking_budget=9000, max_tokens=1000)

        print(f"Result: {result}")
        assert result == "xhigh"

    def test_uses_absolute_thresholds_without_max_tokens(self):
        """
        What it does: Falls back to absolute buckets when max_tokens is unknown.
        Goal: Anthropic clients may send a budget with no usable output limit.
        """
        from kiro.cost_stats import classify_effort

        cases = [
            (1024, "minimal"),
            (4096, "low"),
            (10000, "medium"),
            (30000, "high"),
            (100000, "xhigh"),
        ]
        for budget, expected in cases:
            result = classify_effort(thinking_budget=budget, max_tokens=None)
            print(f"budget={budget} -> {result} (expected {expected})")
            assert result == expected

    def test_zero_max_tokens_uses_absolute_thresholds(self):
        """
        What it does: Avoids division by zero when max_tokens is 0.
        Goal: A degenerate limit must not raise while recording statistics.
        """
        from kiro.cost_stats import classify_effort

        print("Action: Classifying with max_tokens=0...")
        result = classify_effort(thinking_budget=10000, max_tokens=0)

        print(f"Result: {result}")
        assert result == "medium"

    def test_openai_no_max_tokens_round_trips_all_efforts(self):
        """
        What it does: Recovers every effort label when the client omitted max_tokens.
        Goal: Regression guard for the OpenAI default path. When a client sends
              reasoning_effort but no max_tokens, the converter derives the budget
              from FAKE_REASONING_MAX_TOKENS, and routes pass that same base as
              request_max_tokens. classify_effort must then invert every level
              exactly, or cross-effort cost comparisons are silently skewed.
        """
        from kiro.cost_stats import classify_effort
        from kiro.converters_openai import reasoning_effort_to_budget
        from kiro.config import FAKE_REASONING_MAX_TOKENS

        print(f"Setup: Fallback base = {FAKE_REASONING_MAX_TOKENS}")
        for effort in ("minimal", "low", "medium", "high", "xhigh"):
            budget = reasoning_effort_to_budget(FAKE_REASONING_MAX_TOKENS, effort)
            result = classify_effort(thinking_budget=budget, max_tokens=FAKE_REASONING_MAX_TOKENS)
            print(f"  effort={effort} budget={budget} -> {result}")
            assert result == effort, f"{effort} mis-bucketed as {result} at fallback base"


# ==================================================================================================
# Defensive Coercion
# ==================================================================================================

class TestCoercion:
    """Tests for the value coercion helpers."""

    def test_coerce_credits_accepts_numbers(self):
        """
        What it does: Converts int and float credits to float.
        Goal: Downstream arithmetic needs a stable numeric type.
        """
        from kiro.cost_stats import coerce_credits

        assert coerce_credits(1.5) == 1.5
        assert coerce_credits(2) == 2.0
        assert isinstance(coerce_credits(2), float)

    def test_coerce_credits_rejects_bool(self):
        """
        What it does: Rejects booleans as credit amounts.
        Goal: bool subclasses int, so True would otherwise count as 1 credit.
        """
        from kiro.cost_stats import coerce_credits

        assert coerce_credits(True) is None
        assert coerce_credits(False) is None

    def test_coerce_credits_rejects_non_numeric(self):
        """
        What it does: Rejects strings, None, and objects.
        Goal: Upstream data is untrusted and must not enter cost math.
        """
        from kiro.cost_stats import coerce_credits

        assert coerce_credits("1.5") is None
        assert coerce_credits(None) is None
        assert coerce_credits(object()) is None

    def test_coerce_token_count_clamps_negatives(self):
        """
        What it does: Clamps negative token counts to zero.
        Goal: Estimation bugs must not produce negative totals.
        """
        from kiro.cost_stats import coerce_token_count

        assert coerce_token_count(-5) == 0
        assert coerce_token_count(42) == 42

    def test_coerce_token_count_degrades_to_zero(self):
        """
        What it does: Returns 0 for unusable token values.
        Goal: Recording statistics must never raise on a bad estimate.
        """
        from kiro.cost_stats import coerce_token_count

        assert coerce_token_count(None) == 0
        assert coerce_token_count("many") == 0
        assert coerce_token_count(True) == 0

    def test_coerce_token_count_truncates_float(self):
        """
        What it does: Truncates fractional token counts to int.
        Goal: Token counts are whole units in the aggregate.
        """
        from kiro.cost_stats import coerce_token_count

        assert coerce_token_count(12.9) == 12

    def test_coerce_optional_int_preserves_none(self):
        """
        What it does: Keeps None distinct from 0.
        Goal: "No budget provided" must not be confused with "zero budget".
        """
        from kiro.cost_stats import coerce_optional_int

        assert coerce_optional_int(None) is None
        assert coerce_optional_int(0) == 0
        assert coerce_optional_int(8000) == 8000
        assert coerce_optional_int("high") is None
        assert coerce_optional_int(True) is None


# ==================================================================================================
# Aggregate Store
# ==================================================================================================

class TestCostStatsStore:
    """Tests for CostStatsStore aggregation."""

    def test_records_group_by_model_and_effort(self):
        """
        What it does: Keeps separate buckets per (model, effort) pair.
        Goal: The whole point is comparing cost across models and depths.
        """
        from kiro.cost_stats import CostStatsStore

        store = CostStatsStore()
        store.record("claude-opus-5", "high", 100, 50, 2.0)
        store.record("claude-opus-5", "low", 100, 50, 1.0)
        store.record("claude-sonnet-5", "high", 100, 50, 1.3)

        snapshot = store.snapshot()
        print(f"Groups: {[(g['model'], g['effort']) for g in snapshot['groups']]}")

        assert len(snapshot["groups"]) == 3
        assert snapshot["totals"]["requests"] == 3

    def test_accumulates_same_group(self):
        """
        What it does: Sums tokens and credits within one bucket.
        Goal: Repeated calls must aggregate rather than overwrite.
        """
        from kiro.cost_stats import CostStatsStore

        store = CostStatsStore()
        store.record("claude-opus-5", "high", 100, 50, 2.0)
        store.record("claude-opus-5", "high", 200, 100, 3.0)

        snapshot = store.snapshot()
        group = snapshot["groups"][0]
        print(f"Group: {group}")

        assert group["requests"] == 2
        assert group["input_tokens"] == 300
        assert group["output_tokens"] == 150
        assert group["credits"] == pytest.approx(5.0)

    def test_computes_efficiency_metrics(self):
        """
        What it does: Derives tokens-per-credit and credits-per-request.
        Goal: These ratios are the actual cost-efficiency answer.
        """
        from kiro.cost_stats import CostStatsStore

        store = CostStatsStore()
        store.record("claude-opus-5", "high", 900, 100, 2.0)

        group = store.snapshot()["groups"][0]
        print(f"Group: {group}")

        assert group["total_tokens"] == 1000
        assert group["tokens_per_credit"] == pytest.approx(500.0)
        assert group["output_tokens_per_credit"] == pytest.approx(50.0)
        assert group["credits_per_request"] == pytest.approx(2.0)

    def test_requests_without_credits_excluded_from_ratios(self):
        """
        What it does: Uses only priced requests for credits_per_request.
        Goal: Unreported credits must not deflate the average cost.
        """
        from kiro.cost_stats import CostStatsStore

        store = CostStatsStore()
        store.record("claude-opus-5", "high", 100, 100, 2.0)
        store.record("claude-opus-5", "high", 100, 100, None)

        group = store.snapshot()["groups"][0]
        print(f"Group: {group}")

        assert group["requests"] == 2
        assert group["requests_with_credits"] == 1
        assert group["credits_per_request"] == pytest.approx(2.0)

    def test_tokens_counted_even_without_credits(self):
        """
        What it does: Still accumulates tokens when credits are unreported.
        Goal: Token volume remains observable even if metering is missing.
        """
        from kiro.cost_stats import CostStatsStore

        store = CostStatsStore()
        store.record("claude-opus-5", "high", 100, 50, None)

        group = store.snapshot()["groups"][0]
        print(f"Group: {group}")

        assert group["input_tokens"] == 100
        assert group["output_tokens"] == 50
        assert group["tokens_per_credit"] is None
        assert group["credits_per_request"] is None

    def test_zero_credits_yields_no_ratio(self):
        """
        What it does: Avoids division by zero when credits total 0.
        Goal: A free request must not crash the snapshot.
        """
        from kiro.cost_stats import CostStatsStore

        store = CostStatsStore()
        store.record("claude-opus-5", "high", 100, 50, 0.0)

        group = store.snapshot()["groups"][0]
        print(f"Group: {group}")

        assert group["tokens_per_credit"] is None
        assert group["credits_per_request"] == pytest.approx(0.0)

    def test_negative_tokens_clamped(self):
        """
        What it does: Clamps negative token counts inside the store.
        Goal: Totals must stay monotonic and non-negative.
        """
        from kiro.cost_stats import CostStatsStore

        store = CostStatsStore()
        store.record("claude-opus-5", "high", -100, -50, 1.0)

        group = store.snapshot()["groups"][0]
        print(f"Group: {group}")

        assert group["input_tokens"] == 0
        assert group["output_tokens"] == 0

    def test_groups_sorted_by_credits_descending(self):
        """
        What it does: Puts the most expensive bucket first.
        Goal: The costliest usage pattern should be immediately visible.
        """
        from kiro.cost_stats import CostStatsStore

        store = CostStatsStore()
        store.record("cheap", "low", 10, 10, 0.5)
        store.record("expensive", "xhigh", 10, 10, 9.0)
        store.record("mid", "medium", 10, 10, 3.0)

        models = [g["model"] for g in store.snapshot()["groups"]]
        print(f"Order: {models}")

        assert models == ["expensive", "mid", "cheap"]

    def test_snapshot_totals_sum_all_groups(self):
        """
        What it does: Aggregates totals across every bucket.
        Goal: Provide an overall efficiency figure, not just per-group.
        """
        from kiro.cost_stats import CostStatsStore

        store = CostStatsStore()
        store.record("a", "high", 100, 100, 1.0)
        store.record("b", "low", 300, 500, 3.0)

        totals = store.snapshot()["totals"]
        print(f"Totals: {totals}")

        assert totals["requests"] == 2
        assert totals["input_tokens"] == 400
        assert totals["output_tokens"] == 600
        assert totals["credits"] == pytest.approx(4.0)
        assert totals["total_tokens"] == 1000
        assert totals["tokens_per_credit"] == pytest.approx(250.0)

    def test_reset_clears_everything(self):
        """
        What it does: Empties all buckets and counters.
        Goal: Enable clean measurement runs.
        """
        from kiro.cost_stats import CostStatsStore

        store = CostStatsStore()
        store.record("a", "high", 100, 100, 1.0)
        store.reset()

        snapshot = store.snapshot()
        print(f"After reset: {snapshot}")

        assert snapshot["groups"] == []
        assert snapshot["totals"]["requests"] == 0
        assert snapshot["totals"]["credits"] == 0

    def test_empty_snapshot_has_no_ratios(self):
        """
        What it does: Returns None ratios before any request is recorded.
        Goal: An empty store must be safe to serialise.
        """
        from kiro.cost_stats import CostStatsStore

        totals = CostStatsStore().snapshot()["totals"]
        print(f"Empty totals: {totals}")

        assert totals["requests"] == 0
        assert totals["tokens_per_credit"] is None
        assert totals["credits_per_request"] is None

    def test_snapshot_is_json_serialisable(self):
        """
        What it does: Ensures the snapshot survives JSON encoding.
        Goal: It is returned directly from an HTTP endpoint.
        """
        from kiro.cost_stats import CostStatsStore

        store = CostStatsStore()
        store.record("claude-opus-5", "high", 100, 50, 1.25)

        encoded = json.dumps(store.snapshot())
        print(f"Encoded: {encoded[:120]}")

        assert "claude-opus-5" in encoded

    def test_snapshot_does_not_expose_internal_state(self):
        """
        What it does: Returns copies, not live internal dicts.
        Goal: Callers must not be able to corrupt the running totals.
        """
        from kiro.cost_stats import CostStatsStore

        store = CostStatsStore()
        store.record("claude-opus-5", "high", 100, 50, 1.0)

        snapshot = store.snapshot()
        snapshot["groups"][0]["credits"] = 999.0

        assert store.snapshot()["groups"][0]["credits"] == pytest.approx(1.0)

    def test_concurrent_records_are_not_lost(self):
        """
        What it does: Records from many threads without losing updates.
        Goal: The store is shared across concurrent requests.
        """
        import threading
        from kiro.cost_stats import CostStatsStore

        store = CostStatsStore()

        def worker():
            for _ in range(50):
                store.record("claude-opus-5", "high", 10, 10, 1.0)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        totals = store.snapshot()["totals"]
        print(f"Totals after concurrency: {totals}")

        assert totals["requests"] == 400
        assert totals["credits"] == pytest.approx(400.0)


# ==================================================================================================
# record_request_cost
# ==================================================================================================

class TestRecordRequestCost:
    """Tests for the record_request_cost join point."""

    @pytest.fixture(autouse=True)
    def clean_store(self):
        """Resets the module-level store around each test."""
        from kiro.cost_stats import cost_stats
        cost_stats.reset()
        yield
        cost_stats.reset()

    def test_returns_joined_record(self):
        """
        What it does: Returns model, effort, tokens, and credits together.
        Goal: This is the three-way join the feature exists to produce.
        """
        from kiro.cost_stats import record_request_cost

        record = record_request_cost(
            model="claude-opus-5",
            input_tokens=900,
            output_tokens=100,
            credits=2.0,
            thinking_budget=8000,
            max_tokens=10000,
        )
        print(f"Record: {record}")

        assert record["model"] == "claude-opus-5"
        assert record["effort"] == "high"
        assert record["total_tokens"] == 1000
        assert record["credits"] == pytest.approx(2.0)
        assert record["tokens_per_credit"] == pytest.approx(500.0)
        assert record["output_tokens_per_credit"] == pytest.approx(50.0)

    def test_updates_shared_store(self):
        """
        What it does: Feeds the module-level aggregate.
        Goal: The endpoint reads from this same store.
        """
        from kiro.cost_stats import cost_stats, record_request_cost

        record_request_cost(model="claude-sonnet-5", input_tokens=10, output_tokens=10, credits=1.0)

        snapshot = cost_stats.snapshot()
        print(f"Snapshot: {snapshot}")

        assert snapshot["totals"]["requests"] == 1
        assert snapshot["groups"][0]["model"] == "claude-sonnet-5"

    def test_missing_credits_recorded_as_none(self):
        """
        What it does: Handles a stream that reported no metering event.
        Goal: Absent billing data must be visible as unknown, not zero.
        """
        from kiro.cost_stats import record_request_cost

        record = record_request_cost(model="m", input_tokens=10, output_tokens=10, credits=None)
        print(f"Record: {record}")

        assert record["credits"] is None
        assert record["tokens_per_credit"] is None

    def test_survives_non_numeric_inputs(self):
        """
        What it does: Does not raise when handed unusable values.
        Goal: Cost accounting runs after a successful response and must never
              break it. Mock objects in tests exercise this same path.
        """
        from kiro.cost_stats import record_request_cost

        record = record_request_cost(
            model="m",
            input_tokens="lots",
            output_tokens=None,
            credits="free",
            thinking_budget="high",
            max_tokens="big",
        )
        print(f"Record: {record}")

        assert record["input_tokens"] == 0
        assert record["output_tokens"] == 0
        assert record["credits"] is None
        assert record["thinking_budget"] is None
        assert record["effort"] == "default"

    def test_zero_credits_avoids_division(self):
        """
        What it does: Leaves ratios unset when credits are zero.
        Goal: Prevent ZeroDivisionError on a free request.
        """
        from kiro.cost_stats import record_request_cost

        record = record_request_cost(model="m", input_tokens=10, output_tokens=10, credits=0.0)
        print(f"Record: {record}")

        assert record["tokens_per_credit"] is None

    def test_thinking_disabled_records_none_effort(self):
        """
        What it does: Labels a thinking-off request as "none".
        Goal: Compare cost with and without reasoning.
        """
        from kiro.cost_stats import record_request_cost

        record = record_request_cost(
            model="m",
            input_tokens=10,
            output_tokens=10,
            credits=1.0,
            thinking_budget=5000,
            thinking_enabled=False,
        )
        print(f"Record: {record}")

        assert record["effort"] == "none"

    def test_stream_flag_recorded(self):
        """
        What it does: Records whether the request was streamed.
        Goal: Streaming and non-streaming token estimates differ in accuracy.
        """
        from kiro.cost_stats import record_request_cost

        streamed = record_request_cost(model="m", input_tokens=1, output_tokens=1, stream=True)
        blocking = record_request_cost(model="m", input_tokens=1, output_tokens=1, stream=False)

        assert streamed["stream"] is True
        assert blocking["stream"] is False

    def test_logs_cost_line(self, caplog):
        """
        What it does: Emits a one-line cost summary.
        Goal: Cost must be observable in logs without enabling a file sink.
        """
        from loguru import logger
        from kiro.cost_stats import record_request_cost

        messages = []
        sink_id = logger.add(lambda message: messages.append(message), level="INFO")
        try:
            record_request_cost(
                model="claude-opus-5",
                input_tokens=900,
                output_tokens=100,
                credits=2.0,
                thinking_budget=8000,
                max_tokens=10000,
            )
        finally:
            logger.remove(sink_id)

        text = "".join(messages)
        print(f"Logged: {text.strip()}")

        assert "[Cost]" in text
        assert "claude-opus-5" in text
        assert "effort=high" in text
        assert "tok/credit" in text

    def test_logs_na_when_credits_missing(self):
        """
        What it does: Prints n/a rather than a fabricated credit number.
        Goal: Never imply a cost that upstream did not report.
        """
        from loguru import logger
        from kiro.cost_stats import record_request_cost

        messages = []
        sink_id = logger.add(lambda message: messages.append(message), level="INFO")
        try:
            record_request_cost(model="m", input_tokens=1, output_tokens=1, credits=None)
        finally:
            logger.remove(sink_id)

        text = "".join(messages)
        print(f"Logged: {text.strip()}")

        assert "credits=n/a" in text


# ==================================================================================================
# File Persistence
# ==================================================================================================

class TestCostStatsFile:
    """Tests for optional JSON Lines persistence."""

    # NOTE: These tests patch the module attribute rather than reloading the
    # module. Reloading would rebind the cost_stats singleton, while routes_usage
    # still holds a reference to the previous instance, silently splitting the
    # aggregate and breaking unrelated endpoint tests.

    def test_appends_json_lines_when_configured(self, tmp_path, monkeypatch):
        """
        What it does: Appends one JSON object per request to COST_STATS_FILE.
        Goal: Enable offline analysis of credit efficiency.
        """
        import kiro.cost_stats as module

        target = tmp_path / "cost.jsonl"
        monkeypatch.setattr(module, "COST_STATS_FILE", str(target))

        print("Action: Recording two requests...")
        module.record_request_cost(model="m1", input_tokens=10, output_tokens=5, credits=1.0)
        module.record_request_cost(model="m2", input_tokens=20, output_tokens=5, credits=2.0)

        lines = target.read_text(encoding="utf-8").strip().splitlines()
        print(f"Lines: {lines}")

        assert len(lines) == 2
        assert json.loads(lines[0])["model"] == "m1"
        assert json.loads(lines[1])["model"] == "m2"

    def test_persisted_record_contains_efficiency_fields(self, tmp_path, monkeypatch):
        """
        What it does: Verifies the persisted record carries the joined metrics.
        Goal: The file must be sufficient for offline analysis on its own.
        """
        import kiro.cost_stats as module

        target = tmp_path / "cost.jsonl"
        monkeypatch.setattr(module, "COST_STATS_FILE", str(target))

        module.record_request_cost(
            model="claude-opus-5",
            input_tokens=900,
            output_tokens=100,
            credits=2.0,
            thinking_budget=8000,
            max_tokens=10000,
        )

        record = json.loads(target.read_text(encoding="utf-8").strip())
        print(f"Persisted: {record}")

        assert record["model"] == "claude-opus-5"
        assert record["effort"] == "high"
        assert record["total_tokens"] == 1000
        assert record["credits"] == pytest.approx(2.0)
        assert record["tokens_per_credit"] == pytest.approx(500.0)

    def test_disabled_by_default(self, tmp_path, monkeypatch):
        """
        What it does: Writes no file when COST_STATS_FILE is empty.
        Goal: Persistence must be opt-in.
        """
        import kiro.cost_stats as module

        monkeypatch.setattr(module, "COST_STATS_FILE", "")

        print("Action: Recording with persistence disabled...")
        module.record_request_cost(model="m", input_tokens=1, output_tokens=1, credits=1.0)

        print(f"Directory contents: {list(tmp_path.iterdir())}")
        assert list(tmp_path.iterdir()) == []

    def test_reads_configuration_from_environment(self, monkeypatch):
        """
        What it does: Verifies the setting is sourced from the environment and trimmed.
        Goal: The path must be configurable without code changes.
        """
        from kiro.cost_stats import read_cost_stats_file_setting

        monkeypatch.setenv("COST_STATS_FILE", "  /tmp/kiro-cost.jsonl  ")

        result = read_cost_stats_file_setting()
        print(f"Parsed value: {result!r}")
        assert result == "/tmp/kiro-cost.jsonl"

    def test_configuration_defaults_to_empty(self, monkeypatch):
        """
        What it does: Returns an empty string when the variable is unset.
        Goal: Persistence must default to off.
        """
        from kiro.cost_stats import read_cost_stats_file_setting

        monkeypatch.delenv("COST_STATS_FILE", raising=False)

        result = read_cost_stats_file_setting()
        print(f"Parsed value: {result!r}")
        assert result == ""

    def test_whitespace_only_configuration_disables_persistence(self, monkeypatch):
        """
        What it does: Treats a whitespace-only value as disabled.
        Goal: Avoid attempting to write to an empty path.
        """
        from kiro.cost_stats import read_cost_stats_file_setting

        monkeypatch.setenv("COST_STATS_FILE", "   ")

        result = read_cost_stats_file_setting()
        print(f"Parsed value: {result!r}")
        assert result == ""

    def test_unwritable_path_does_not_raise(self, tmp_path, monkeypatch):
        """
        What it does: Swallows write failures.
        Goal: A logging problem must never fail a request that already succeeded.
        """
        import kiro.cost_stats as module

        unwritable = tmp_path / "missing-dir" / "cost.jsonl"
        monkeypatch.setattr(module, "COST_STATS_FILE", str(unwritable))

        print("Action: Recording with an unwritable target...")
        record = module.record_request_cost(
            model="m", input_tokens=1, output_tokens=1, credits=1.0
        )

        print(f"Record still returned: {record['model']}")
        assert record["model"] == "m"
        assert not unwritable.exists()

    def test_stats_still_aggregated_when_write_fails(self, tmp_path, monkeypatch):
        """
        What it does: Keeps the in-memory aggregate correct despite a write error.
        Goal: File persistence is auxiliary and must not affect accounting.
        """
        import kiro.cost_stats as module

        monkeypatch.setattr(module, "COST_STATS_FILE", str(tmp_path / "nope" / "x.jsonl"))
        module.cost_stats.reset()

        module.record_request_cost(model="m", input_tokens=10, output_tokens=10, credits=1.0)

        totals = module.cost_stats.snapshot()["totals"]
        print(f"Totals: {totals}")
        assert totals["requests"] == 1
