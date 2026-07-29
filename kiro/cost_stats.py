# -*- coding: utf-8 -*-

# Kiro Gateway
# https://github.com/jwadow/kiro-gateway
# Copyright (C) 2025 Jwadow
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
Per-request cost accounting for Kiro credits.

Kiro bills in credit "invocations" rather than tokens: each request costs a
model-specific credit multiplier regardless of how many tokens it moves. That
makes the interesting question "how many tokens did I get per credit?", which
requires joining three values that live in different layers:

- credits: reported by Kiro's meteringEvent near the end of the stream
- tokens: counted/estimated by the gateway (input + output)
- model + effort: taken from the client request

This module collects those three per request, logs a one-line summary, and keeps
an in-memory aggregate grouped by (model, effort) so cost efficiency can be
compared across models and reasoning depths.

Optional persistence: set COST_STATS_FILE to append one JSON object per request
for offline analysis. Records are appended as JSON Lines.
"""

from __future__ import annotations

import json
import math
import os
import threading
from typing import Any, Dict, List, Optional

from loguru import logger


# ==================================================================================================
# Configuration
# ==================================================================================================

def read_cost_stats_file_setting() -> str:
    """
    Reads the COST_STATS_FILE setting from the environment.

    Exposed as a function so the parsing can be tested without reloading this
    module: a reload would rebind the cost_stats singleton while other modules
    still hold the previous instance, silently splitting the aggregate.

    Returns:
        Trimmed path, or "" when persistence is disabled

    Examples:
        >>> import os
        >>> os.environ["COST_STATS_FILE"] = "  /tmp/cost.jsonl  "
        >>> read_cost_stats_file_setting()
        '/tmp/cost.jsonl'
        >>> del os.environ["COST_STATS_FILE"]
        >>> read_cost_stats_file_setting()
        ''
    """
    return os.getenv("COST_STATS_FILE", "").strip()


# Optional path for JSON Lines persistence. Empty/unset disables file output.
COST_STATS_FILE: str = read_cost_stats_file_setting()

# Effort labels ordered from cheapest to most expensive reasoning depth.
# Thresholds are the *upper* bound of the thinking-budget-to-max-tokens ratio,
# mirroring the percentages used by reasoning_effort_to_budget().
_EFFORT_RATIO_THRESHOLDS: List[tuple] = [
    (0.0, "none"),
    (0.15, "minimal"),
    (0.35, "low"),
    (0.65, "medium"),
    (0.875, "high"),
    (1.0, "xhigh"),
]

# Fallback buckets when max_tokens is unknown and only an absolute budget exists.
_EFFORT_ABSOLUTE_THRESHOLDS: List[tuple] = [
    (0, "none"),
    (2048, "minimal"),
    (6144, "low"),
    (16384, "medium"),
    (32768, "high"),
]


# ==================================================================================================
# Effort Classification
# ==================================================================================================

def classify_effort(
    thinking_budget: Optional[int],
    max_tokens: Optional[int] = None,
    thinking_enabled: bool = True,
) -> str:
    """
    Derive a comparable effort label from a thinking budget.

    Clients express reasoning depth differently (Anthropic sends an absolute
    thinking.budget_tokens, OpenAI sends reasoning_effort). Both end up as a
    token budget, so the budget is normalised back into a label here to make
    aggregates comparable across API shapes.

    Args:
        thinking_budget: Thinking budget in tokens (None = server default)
        max_tokens: Request output limit, used to compute the budget ratio
        thinking_enabled: False when the client disabled thinking entirely

    Returns:
        One of "none", "minimal", "low", "medium", "high", "xhigh", or
        "default" when thinking is on but no explicit budget was given.

    Examples:
        >>> classify_effort(None, thinking_enabled=False)
        'none'
        >>> classify_effort(0, 4096)
        'none'
        >>> classify_effort(3276, 4096)
        'high'
        >>> classify_effort(None, 4096)
        'default'
    """
    if not thinking_enabled:
        return "none"

    if thinking_budget is None:
        return "default"

    if thinking_budget <= 0:
        return "none"

    # Prefer the ratio-based mapping, which matches how efforts are converted
    # into budgets in the first place.
    if max_tokens and max_tokens > 0:
        ratio = thinking_budget / max_tokens
        for threshold, label in _EFFORT_RATIO_THRESHOLDS:
            if ratio <= threshold:
                return label
        return "xhigh"

    for threshold, label in _EFFORT_ABSOLUTE_THRESHOLDS:
        if thinking_budget <= threshold:
            return label
    return "xhigh"


# ==================================================================================================
# Aggregate Store
# ==================================================================================================

class CostStatsStore:
    """
    Thread-safe aggregate of per-request cost records.

    Records are grouped by "<model>|<effort>" so that the credit cost of a model
    can be compared across reasoning depths. Only totals are kept, which keeps
    memory flat regardless of traffic volume.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._groups: Dict[str, Dict[str, Any]] = {}
        self._requests_total: int = 0
        self._requests_with_credits: int = 0

    @staticmethod
    def group_key(model: str, effort: str) -> str:
        """Builds the aggregate key for a model/effort pair."""
        return f"{model}|{effort}"

    def record(
        self,
        model: str,
        effort: str,
        input_tokens: int,
        output_tokens: int,
        credits: Optional[float],
    ) -> Dict[str, Any]:
        """
        Adds one request to the aggregate.

        Args:
            model: Model id used for the request
            effort: Effort label from classify_effort()
            input_tokens: Input/prompt tokens
            output_tokens: Output/completion tokens
            credits: Credits reported by Kiro, or None when not reported

        Returns:
            The updated group aggregate (a copy, safe to serialise)
        """
        key = self.group_key(model, effort)

        with self._lock:
            group = self._groups.setdefault(key, {
                "model": model,
                "effort": effort,
                "requests": 0,
                "requests_with_credits": 0,
                "credits": 0.0,
                "input_tokens": 0,
                "output_tokens": 0,
            })

            group["requests"] += 1
            group["input_tokens"] += max(0, input_tokens)
            group["output_tokens"] += max(0, output_tokens)

            self._requests_total += 1

            if credits is not None:
                group["requests_with_credits"] += 1
                group["credits"] += credits
                self._requests_with_credits += 1

            return dict(group)

    def snapshot(self) -> Dict[str, Any]:
        """
        Returns the current aggregate with derived efficiency metrics.

        Derived fields are computed on read so that no precision is lost in the
        running totals:
            credits_per_request: average credit cost of one call
            tokens_per_credit: total tokens obtained per credit spent
            output_tokens_per_credit: output-only tokens per credit spent

        Returns:
            Dict with "totals" and "groups" (sorted by credits spent, descending)
        """
        with self._lock:
            groups = [dict(group) for group in self._groups.values()]
            requests_total = self._requests_total
            requests_with_credits = self._requests_with_credits

        total_credits = 0.0
        total_input = 0
        total_output = 0

        for group in groups:
            total_credits += group["credits"]
            total_input += group["input_tokens"]
            total_output += group["output_tokens"]
            _attach_derived_metrics(group)

        groups.sort(key=lambda item: (item["credits"], item["requests"]), reverse=True)

        totals: Dict[str, Any] = {
            "requests": requests_total,
            "requests_with_credits": requests_with_credits,
            "credits": round(total_credits, 6),
            "input_tokens": total_input,
            "output_tokens": total_output,
        }
        _attach_derived_metrics(totals)

        return {"totals": totals, "groups": groups}

    def reset(self) -> None:
        """Clears all collected statistics."""
        with self._lock:
            self._groups.clear()
            self._requests_total = 0
            self._requests_with_credits = 0


def _attach_derived_metrics(bucket: Dict[str, Any]) -> None:
    """
    Adds efficiency ratios to an aggregate bucket in place.

    Ratios are only meaningful for requests where Kiro actually reported credits,
    so requests_with_credits is used as the denominator rather than requests.
    """
    credits = bucket.get("credits", 0.0) or 0.0
    priced_requests = bucket.get("requests_with_credits", 0) or 0
    total_tokens = (bucket.get("input_tokens", 0) or 0) + (bucket.get("output_tokens", 0) or 0)

    bucket["credits"] = round(credits, 6)
    bucket["total_tokens"] = total_tokens
    bucket["credits_per_request"] = round(credits / priced_requests, 6) if priced_requests else None
    bucket["tokens_per_credit"] = round(total_tokens / credits, 2) if credits > 0 else None
    bucket["output_tokens_per_credit"] = (
        round((bucket.get("output_tokens", 0) or 0) / credits, 2) if credits > 0 else None
    )


# Module-level singleton used by the request paths.
cost_stats = CostStatsStore()


# ==================================================================================================
# Recording
# ==================================================================================================

def coerce_token_count(value: Any) -> int:
    """
    Normalises a token count to a non-negative int.

    Token counts reach this module from several estimation paths, so a missing
    or non-numeric value must degrade to 0 rather than raise while building a
    statistics record.

    Args:
        value: Candidate token count

    Returns:
        The count as a non-negative int, or 0 if it is not a usable number

    Examples:
        >>> coerce_token_count(42)
        42
        >>> coerce_token_count(-5)
        0
        >>> coerce_token_count(None)
        0
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return max(0, int(value))


def coerce_optional_int(value: Any) -> Optional[int]:
    """
    Normalises an optional integer field, preserving None for "not provided".

    Args:
        value: Candidate value (e.g. thinking budget or max_tokens)

    Returns:
        The value as an int, or None if absent or not a usable number

    Examples:
        >>> coerce_optional_int(8000)
        8000
        >>> coerce_optional_int(None) is None
        True
        >>> coerce_optional_int("high") is None
        True
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def coerce_credits(value: Any) -> Optional[float]:
    """
    Normalises a reported credit amount to a float, or None when unusable.

    Credits arrive from upstream stream events, so the value cannot be trusted
    to be numeric. Booleans are rejected explicitly because bool is a subclass
    of int and would otherwise be silently counted as 0 or 1 credits. NaN and
    infinities are rejected because they would poison every running total and
    are not valid JSON, which would break /v1/cost-stats serialisation.

    Args:
        value: Candidate credit amount from upstream

    Returns:
        The amount as a float, or None if it is not a usable finite number

    Examples:
        >>> coerce_credits(1.5)
        1.5
        >>> coerce_credits(True) is None
        True
        >>> coerce_credits("1.5") is None
        True
        >>> coerce_credits(float("nan")) is None
        True
        >>> coerce_credits(float("inf")) is None
        True
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    amount = float(value)
    if not math.isfinite(amount):
        return None
    return amount


def record_request_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    credits: Optional[float] = None,
    thinking_budget: Optional[int] = None,
    max_tokens: Optional[int] = None,
    thinking_enabled: bool = True,
    stream: bool = False,
) -> Dict[str, Any]:
    """
    Records and logs the cost of one completed request.

    This is the single join point for the three values needed to judge credit
    efficiency: the credits Kiro charged, the tokens moved, and the model/effort
    that produced them.

    Args:
        model: Model id used for the request
        input_tokens: Input/prompt tokens
        output_tokens: Output/completion tokens
        credits: Credits from Kiro's meteringEvent (None when not reported)
        thinking_budget: Thinking budget in tokens, if the client set one
        max_tokens: Request output limit, used to normalise the effort label
        thinking_enabled: False when the client disabled thinking
        stream: Whether the request was streamed

    Returns:
        The record that was logged and aggregated
    """
    # Values arrive from upstream events and several estimation paths, so they are
    # normalised before any arithmetic. Cost accounting must never break a
    # response that was already produced successfully.
    safe_credits = coerce_credits(credits)
    safe_input_tokens = coerce_token_count(input_tokens)
    safe_output_tokens = coerce_token_count(output_tokens)
    safe_thinking_budget = coerce_optional_int(thinking_budget)
    safe_max_tokens = coerce_optional_int(max_tokens)

    effort = classify_effort(
        thinking_budget=safe_thinking_budget,
        max_tokens=safe_max_tokens,
        thinking_enabled=bool(thinking_enabled),
    )

    total_tokens = safe_input_tokens + safe_output_tokens

    record: Dict[str, Any] = {
        "model": model,
        "effort": effort,
        "thinking_budget": safe_thinking_budget,
        "max_tokens": safe_max_tokens,
        "input_tokens": safe_input_tokens,
        "output_tokens": safe_output_tokens,
        "total_tokens": total_tokens,
        "credits": safe_credits,
        "stream": bool(stream),
    }

    if safe_credits is not None and safe_credits > 0:
        record["tokens_per_credit"] = round(total_tokens / safe_credits, 2)
        record["output_tokens_per_credit"] = round(safe_output_tokens / safe_credits, 2)
    else:
        record["tokens_per_credit"] = None
        record["output_tokens_per_credit"] = None

    cost_stats.record(
        model=model,
        effort=effort,
        input_tokens=safe_input_tokens,
        output_tokens=safe_output_tokens,
        credits=safe_credits,
    )

    credits_text = f"{safe_credits:.4f}" if safe_credits is not None else "n/a"
    efficiency_text = (
        f", {record['tokens_per_credit']} tok/credit"
        if record["tokens_per_credit"] is not None
        else ""
    )
    logger.info(
        f"[Cost] {model} effort={effort} "
        f"in={safe_input_tokens} out={safe_output_tokens} total={total_tokens} "
        f"credits={credits_text}{efficiency_text}"
    )

    _append_to_file(record)

    return record


def _append_to_file(record: Dict[str, Any]) -> None:
    """
    Appends one record to COST_STATS_FILE as JSON Lines.

    Persistence is best-effort: a failing write must never break the response
    that was already produced, so errors are logged and swallowed.
    """
    if not COST_STATS_FILE:
        return

    try:
        with open(COST_STATS_FILE, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as error:
        logger.warning(f"Failed to append cost stats to {COST_STATS_FILE}: {error}")
