"""差异复现包引擎（纯函数，无副作用、无 I/O）。

用途
----
给定两个策略 run 的规则快照与一条在两侧**决定不同**的目标告警，用 ddmin
(Zeller & Hildebrandt, 简化 Delta Debugging) 对"目标前缀"（原 seq 不大于
目标的全部告警）反复删块，直到再删任一条告警都会使目标在任一侧的
``(决定, 命中规则)`` 偏离原结果——得到能复现该差异的**最小告警序列**。

每次成员测试都：
1. 分别载入两份 run 各自的规则快照（互不影响）；
2. 使用全新的 ``EvalState``（冷却状态清空）；
3. 按原 seq/ts 顺序重放候选子集；
全程不落库、不转发（调用方只允许传入内存数据）。

本模块不依赖 FastAPI 与数据库，可单独导入做单元测试。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from .engine import EvalState, RuleView, evaluate, ordered_rules

# 导出 JSON 的格式版本（只读重新验证按该版本解析）
PACKAGE_VERSION = 1

_RULE_FIELDS = (
    "id", "name", "description", "action", "priority", "cooldown_seconds",
    "match_device", "match_batch", "match_level", "time_start", "time_end",
    "date_start", "date_end", "enabled",
)

_ALERT_FIELDS = (
    "id", "seq", "device_id", "batch_id", "level", "alert_type", "message", "ts",
)


# ---------------------------------------------------------------------------
# 数据规整
# ---------------------------------------------------------------------------

def _rule_plain(rule: Any) -> dict:
    """把规则快照行 / RuleView 规整为可 JSON 序列化的普通 dict（仅取快照字段）。"""
    if isinstance(rule, RuleView):
        return {k: getattr(rule, k) for k in _RULE_FIELDS}
    return {k: rule[k] for k in _RULE_FIELDS if k in rule.keys()}


def snapshot_views(snapshot: list[dict]) -> list[RuleView]:
    """一份 run 的规则快照（dict 列表）→ 按评估顺序排列的 RuleView。

    仅启用规则参与评估，与 live/replay 的 ``ordered_rules`` 语义完全一致。
    """
    return ordered_rules([_rule_plain(r) for r in snapshot])


def canonical_json(data: Any) -> str:
    """规范序列化：键排序、无多余空白，使哈希只取决于内容。"""
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def snapshot_summary(snapshot: list[dict]) -> dict:
    """规则快照摘要：条目数、启用数、内容哈希与逐条简述（展示用，不参与重放）。"""
    rules = [_rule_plain(r) for r in snapshot]
    rules.sort(key=lambda r: (r["priority"], r["id"]))
    digest = hashlib.sha256(canonical_json(rules).encode("utf-8")).hexdigest()
    enabled = [r for r in rules if r["enabled"]]
    return {
        "rule_count": len(rules),
        "enabled_count": len(enabled),
        "digest_sha256": digest,
        "rules": rules,
    }


def _alert_plain(alert: dict) -> dict:
    """规整一条告警为可序列化形式（ts 统一为 ISO 字符串）。"""
    out = {k: alert.get(k) for k in _ALERT_FIELDS}
    ts = alert.get("ts")
    if hasattr(ts, "isoformat"):
        out["ts"] = ts.isoformat(timespec="seconds")
    return out


# ---------------------------------------------------------------------------
# 内存重放
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TargetOutcome:
    """目标告警在某一侧的期望结果：决定 + 命中规则（None=默认放行）。"""
    decision: str
    rule_id: Optional[int]

    def as_dict(self) -> dict:
        return {"decision": self.decision, "rule_id": self.rule_id}


def outcome_of(result: dict) -> TargetOutcome:
    return TargetOutcome(decision=result["decision"], rule_id=result["rule_id"])


def replay_subset(alerts: list[dict], rules: list[RuleView]) -> list[dict]:
    """用给定规则快照对告警子集按原顺序、从全新冷却状态重放（纯内存）。

    - 告警按 (seq, id) 稳定排序后重放，与 live/replay 的注入顺序一致；
    - 每次调用都新建 ``EvalState``，冷却状态从零开始；
    - 不做任何持久化、不触发转发。
    """
    ordered = sorted(alerts, key=lambda a: (a["seq"], a.get("id", 0)))
    state = EvalState()
    results = []
    for alert in ordered:
        results.append(evaluate(alert, rules, state))
    return results


def find_target(alerts: list[dict], target_id: int) -> dict:
    for a in alerts:
        if a.get("id") == target_id:
            return a
    raise ValueError(f"候选告警中不存在目标告警 id={target_id}")


def evaluate_sides(
    alerts: list[dict],
    rules_a: list[RuleView],
    rules_b: list[RuleView],
    target_id: int,
) -> dict:
    """分别用两侧规则快照重放同一告警序列，提取目标在两侧的 (决定, 命中规则)。

    返回 ``{"a": outcome|None, "b": outcome|None, "missing": bool}``。
    目标不在某侧重放序列中（理论上不会发生，目标始终保留）时对应侧为 None。
    注意：两侧结果**本来就允许不同**（目标是差异告警），这里只做提取不做比较。
    """
    def one_side(rules: list[RuleView]) -> Optional[TargetOutcome]:
        ordered = sorted(alerts, key=lambda a: (a["seq"], a.get("id", 0)))
        state = EvalState()
        for alert in ordered:
            result = evaluate(alert, rules, state)
            if alert.get("id") == target_id:
                return outcome_of(result)
        return None

    out_a = one_side(rules_a)
    out_b = one_side(rules_b)
    return {"a": out_a, "b": out_b, "missing": out_a is None or out_b is None}


def matches_expected(
    alerts: list[dict],
    rules_a: list[RuleView],
    rules_b: list[RuleView],
    target_id: int,
    expected_a: TargetOutcome,
    expected_b: TargetOutcome,
) -> tuple[bool, Optional[str]]:
    """成员测试：删掉该块后，目标在两侧的 (决定, 命中规则) 是否仍与原结果一致。

    一致 → True（该块可以删，继续缩小）；任一偏离 → False（必须保留）。
    """
    got = evaluate_sides(alerts, rules_a, rules_b, target_id)
    out_a, out_b = got["a"], got["b"]
    if got["missing"] or out_a != expected_a or out_b != expected_b:
        # 给出更明确的逐条偏离原因，便于展示"保留原因"
        if got["missing"]:
            return False, "目标告警缺失（被意外删出候选序列）"
        reasons = []
        if out_a != expected_a:
            reasons.append(
                f"A 侧 {expected_a.decision}/"
                f"{expected_a.rule_id if expected_a.rule_id is not None else '默认'} → "
                f"{out_a.decision}/{out_a.rule_id if out_a.rule_id is not None else '默认'}"
            )
        if out_b != expected_b:
            reasons.append(
                f"B 侧 {expected_b.decision}/"
                f"{expected_b.rule_id if expected_b.rule_id is not None else '默认'} → "
                f"{out_b.decision}/{out_b.rule_id if out_b.rule_id is not None else '默认'}"
            )
        return False, "目标结果偏离：" + "；".join(reasons)
    return True, None


# ---------------------------------------------------------------------------
# ddmin：简化 Delta Debugging
# ---------------------------------------------------------------------------

def _split(items: list, n: int) -> list[list]:
    """把 items 确定性地恰好切成 n 个非空块（余数依次分给前面的块）。"""
    n = max(1, min(n, len(items)))
    base, extra = divmod(len(items), n)
    chunks, start = [], 0
    for i in range(n):
        size = base + (1 if i < extra else 0)
        chunks.append(items[start:start + size])
        start += size
    return chunks


def minimize(
    prefix: list[dict],
    snapshot_a: list[dict],
    snapshot_b: list[dict],
    target_id: int,
    expected_a: TargetOutcome,
    expected_b: TargetOutcome,
) -> dict:
    """对目标前缀执行 ddmin，返回最小复现序列与逐轮删减/保留记录。

    ``expected_a/b`` 为两个 run 中目标告警**存储的原结果**（决定 + 命中规则），
    由调用方从 decisions 表读取（纯函数本身不访问数据库）。两边允许且应当
    不同——目标本来就是差异告警。

    开始前先用完整前缀、各自规则快照、全新冷却状态重放一次做基线校验：
    若清空冷却后无法重现存储的原结果（目标的冷却依赖前缀之外的更早告警），
    抛 ValueError，由调用方提示用户。

    返回::

        {
          "minimal": [...告警 dict...],
          "removed":  [...被删掉的告警 dict...],
          "expected_a": {"decision", "rule_id"},
          "expected_b": {"decision", "rule_id"},
          "rounds":   [{round, n, kept_size, removed_any, chunks: [
              {index, alert_ids, alert_seqs, removed: bool, reason}
          ]}],
          "stats": {"test_count", "round_count", "initial_size", "minimal_size"},
        }
    """
    rules_a = snapshot_views(snapshot_a)
    rules_b = snapshot_views(snapshot_b)

    prefix = sorted(prefix, key=lambda a: (a["seq"], a.get("id", 0)))
    find_target(prefix, target_id)

    baseline = evaluate_sides(prefix, rules_a, rules_b, target_id)
    if baseline["missing"]:
        raise ValueError("完整前缀重放后未找到目标告警的决定")
    if baseline["a"] != expected_a or baseline["b"] != expected_b:
        def _fmt(got, want, side):
            g = f"{got.decision}/{'规则#' + str(got.rule_id) if got.rule_id is not None else '默认策略'}"
            w = f"{want.decision}/{'规则#' + str(want.rule_id) if want.rule_id is not None else '默认策略'}"
            return f"{side} 侧存储原结果 {w}，用该 run 规则快照清空冷却重放得到 {g}"
        raise ValueError(
            "完整前缀在清空冷却状态后无法重现目标的原结果，不能生成自包含的最小复现序列"
            "（常见原因：① 目标的冷却基准由前缀之外的更早告警建立；"
            "② live 链规则快照与之后新增/修改的规则不一致）："
            + _fmt(baseline["a"], expected_a, "A") + "；" + _fmt(baseline["b"], expected_b, "B")
        )

    current = list(prefix)  # 当前保留的候选
    n = 2
    test_count = 0
    rounds: list[dict] = []
    round_no = 0

    while len(current) >= 2:
        round_no += 1
        n = min(n, len(current))
        chunks = _split(current, n)

        chunk_logs = []
        removed_any = False
        for idx, chunk in enumerate(chunks):
            chunk_ids = [a.get("id") for a in chunk]
            chunk_id_set = set(chunk_ids)
            kept = [a for a in current if a.get("id") not in chunk_id_set]
            test_count += 1
            ok, reason = matches_expected(
                kept, rules_a, rules_b, target_id, expected_a, expected_b
            )
            if ok:
                # 删除后目标两侧结果不变 → 整块删除（缩小到补集）
                current = kept
                removed_any = True
                chunk_logs.append({
                    "index": idx + 1,
                    "alert_ids": chunk_ids,
                    "alert_seqs": [a["seq"] for a in chunk],
                    "removed": True,
                    "reason": "删除该块后目标在 A/B 两侧的决定与命中规则均与原结果一致，整块删减",
                })
            else:
                chunk_logs.append({
                    "index": idx + 1,
                    "alert_ids": chunk_ids,
                    "alert_seqs": [a["seq"] for a in chunk],
                    "removed": False,
                    "reason": reason or "删除后目标结果偏离，保留该块",
                })

        rounds.append({
            "round": round_no,
            "n": n,
            "chunk_count": len(chunks),
            "kept_size": len(current),
            "removed_any": removed_any,
            "chunks": chunk_logs,
        })

        if removed_any:
            n = max(n - 1, 2)  # 删除成功：在更小的候选上回到粗粒度重试
        else:
            if n == len(current):
                break  # 每个块都只剩 1 条且均不可删 → 1-最小
            n = min(n * 2, len(current))  # 全部保留：加细粒度再试

    kept_ids = {a.get("id") for a in current}
    removed = [a for a in prefix if a.get("id") not in kept_ids]
    return {
        "minimal": [_alert_plain(a) for a in current],
        "removed": [_alert_plain(a) for a in removed],
        "expected_a": expected_a.as_dict(),
        "expected_b": expected_b.as_dict(),
        "rounds": rounds,
        "stats": {
            "test_count": test_count,
            "round_count": round_no,
            "initial_size": len(prefix),
            "minimal_size": len(current),
        },
    }


# ---------------------------------------------------------------------------
# 差异复现包：组装与只读重新验证
# ---------------------------------------------------------------------------

def _outcome_from_dict(data: dict, label: str) -> TargetOutcome:
    if not isinstance(data, dict) or data.get("decision") not in ("pass", "suppress"):
        raise ValueError(f"{label} 期望结果格式无效：需要 decision(pass|suppress) 与 rule_id")
    rule_id = data.get("rule_id")
    if rule_id is not None and not isinstance(rule_id, int):
        raise ValueError(f"{label} 期望结果的 rule_id 必须是整数或 null")
    return TargetOutcome(decision=data["decision"], rule_id=rule_id)


def _alert_from_package(item: dict) -> dict:
    if not isinstance(item, dict) or "seq" not in item or "ts" not in item:
        raise ValueError("最小告警序列中的条目格式无效（需要 seq 与 ts）")
    return {
        "id": item.get("id"),
        "seq": item["seq"],
        "device_id": item.get("device_id", "") or "",
        "batch_id": item.get("batch_id", "") or "",
        "level": item.get("level", "info") or "info",
        "alert_type": item.get("alert_type", "") or "",
        "message": item.get("message", "") or "",
        "ts": item["ts"],
    }


def build_package(
    run_a: dict,
    run_b: dict,
    target: dict,
    minimal_result: dict,
    created_at: str,
) -> dict:
    """组装差异复现包 JSON（自包含：内嵌两侧规则快照与最小告警序列）。

    ``run_a/run_b`` 需含 id/kind/note 与 rule_snapshot；``target`` 为目标告警
    dict；``minimal_result`` 为 :func:`minimize` 的返回。
    """
    snap_a = list(run_a.get("rule_snapshot") or [])
    snap_b = list(run_b.get("rule_snapshot") or [])
    target_plain = _alert_plain(target)
    expected_a = minimal_result["expected_a"]
    expected_b = minimal_result["expected_b"]
    package = {
        "package_version": PACKAGE_VERSION,
        "kind": "difference_reproduction_package",
        "created_at": created_at,
        "source": {
            "run_a": {"id": run_a.get("id"), "kind": run_a.get("kind"),
                      "note": run_a.get("note", "")},
            "run_b": {"id": run_b.get("id"), "kind": run_b.get("kind"),
                      "note": run_b.get("note", "")},
        },
        "target": target_plain,
        "expected": {"a": expected_a, "b": expected_b},
        "rule_snapshots": {
            "a": {"run_id": run_a.get("id"), **snapshot_summary(snap_a)},
            "b": {"run_id": run_b.get("id"), **snapshot_summary(snap_b)},
        },
        "minimal_alerts": minimal_result["minimal"],
        "ddmin": {
            "stats": minimal_result["stats"],
            "rounds": minimal_result["rounds"],
        },
    }
    package["package_digest_sha256"] = package_digest(package)
    return package


def package_digest(package: dict) -> str:
    """对包内决定复现结果的关键内容求哈希（不含摘要展示字段本身）。"""
    payload = {
        "v": PACKAGE_VERSION,
        "target": {k: package["target"].get(k) for k in _ALERT_FIELDS},
        "expected": package["expected"],
        "rules_a": package["rule_snapshots"]["a"]["rules"],
        "rules_b": package["rule_snapshots"]["b"]["rules"],
        "alerts": package["minimal_alerts"],
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def verify_package(package: dict) -> dict:
    """只读重新验证：用包内规则快照从零重放最小序列，核对目标的期望结果。

    完全自包含、无数据库访问、不落库、不转发。返回::

        {
          "ok": bool,                # 目标在两侧是否都重现期望
          "difference_reproduced": bool,  # 两侧决定是否确实不同
          "target": {...},
          "expected": {...}, "actual": {"a": outcome, "b": outcome},
          "checks": [{"side", "passed", "reason"}],
          "digest_sha256": "...", "digest_matches": bool,
          "minimal_alert_count": int,
        }
    """
    if not isinstance(package, dict):
        raise ValueError("复现包必须是 JSON 对象")
    if package.get("package_version") != PACKAGE_VERSION:
        raise ValueError(f"不支持的复现包版本: {package.get('package_version')!r}")
    try:
        target = package["target"]
        target_id = target["id"]
        expected_a = _outcome_from_dict(package["expected"]["a"], "A 侧")
        expected_b = _outcome_from_dict(package["expected"]["b"], "B 侧")
        alerts = [_alert_from_package(x) for x in package["minimal_alerts"]]
        rules_a = snapshot_views(package["rule_snapshots"]["a"]["rules"])
        rules_b = snapshot_views(package["rule_snapshots"]["b"]["rules"])
    except (KeyError, TypeError) as exc:
        raise ValueError(f"复现包结构不完整：{exc}")

    if not any(a.get("id") == target_id for a in alerts):
        raise ValueError("最小告警序列中找不到目标告警，无法重放")

    # ts 为 ISO 字符串，evaluate 需要 datetime；直接 fromisoformat 与引擎解析同语义
    for a in alerts:
        a["ts"] = datetime.fromisoformat(a["ts"])

    got = evaluate_sides(alerts, rules_a, rules_b, target_id)
    actual_a, actual_b = got["a"], got["b"]

    checks = []
    for side, actual, expected in (("a", actual_a, expected_a), ("b", actual_b, expected_b)):
        if actual is None:
            checks.append({"side": side, "passed": False,
                           "reason": "重放后未找到目标告警的决定"})
        elif actual == expected:
            checks.append({"side": side, "passed": True,
                           "reason": f"{side.upper()} 侧重现期望：{expected.decision} / "
                                     f"{'规则#' + str(expected.rule_id) if expected.rule_id is not None else '默认策略'}"})
        else:
            checks.append({"side": side, "passed": False,
                           "reason": f"{side.upper()} 侧结果偏离：期望 {expected.decision}/"
                                     f"{expected.rule_id}，实际 {actual.decision}/{actual.rule_id}"})

    digest_now = package_digest(package)
    digest_matches = package.get("package_digest_sha256") == digest_now
    ok = all(c["passed"] for c in checks)
    difference_reproduced = (
        ok and actual_a is not None and actual_b is not None
        and actual_a.decision != actual_b.decision
    )
    return {
        "ok": ok,
        "difference_reproduced": difference_reproduced,
        "target": target,
        "expected": {"a": expected_a.as_dict(), "b": expected_b.as_dict()},
        "actual": {
            "a": actual_a.as_dict() if actual_a else None,
            "b": actual_b.as_dict() if actual_b else None,
        },
        "checks": checks,
        "digest_sha256": digest_now,
        "digest_matches": digest_matches,
        "minimal_alert_count": len(alerts),
    }
