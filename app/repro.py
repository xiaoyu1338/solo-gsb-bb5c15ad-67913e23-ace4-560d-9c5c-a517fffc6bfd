"""差异复现包（differential reproduction package）—— ddmin 最小化引擎（纯函数）。

给定两个 run 各自的**规则快照**、一条"决定不同"的目标告警以及**目标前缀**告警，
用 ddmin（delta-debugging minimize）反复删除告警块，每一轮都：

1. 分别载入两份 run 的规则快照（不读当前 rules 表）；
2. 两侧各自使用**全新的冷却状态**（``EvalState`` 为空）；
3. 按候选序列的原 seq 顺序、以告警自身 ts 重放；

直到再删任一告警都会使目标在任一侧的**决定或命中规则**偏离原结果。

本模块不依赖 FastAPI / 数据库、不做任何 I/O：既不会写入
alerts / decisions / runs，也不可能触发任何转发。复现包 JSON 自带两侧完整规则
快照与最小告警序列，可脱离数据库做只读重新验证。
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from .engine import EvalState, RuleView, evaluate, normalize_alert, ordered_rules

PACKAGE_TYPE = "cold-chain-diff-repro"
SCHEMA_VERSION = "diff-repro-v1"

_RULE_KEYS = (
    "id", "name", "description", "action", "priority", "cooldown_seconds",
    "match_device", "match_batch", "match_level", "time_start", "time_end",
    "date_start", "date_end", "enabled",
)

_SIDE_LABEL = {"a": "A 侧（版本 A）", "b": "B 侧（版本 B）"}


class BaselineMismatch(Exception):
    """目标前缀在全新冷却状态下无法复现两个 run 存储的原结果（如 live 链曾用
    与快照不同的规则评估），此时最小化没有可信基准。"""

    def __init__(self, deviations: list[dict], observed: dict):
        self.deviations = deviations
        self.observed = observed
        super().__init__("；".join(d["detail"] for d in deviations))


# ---------------------------------------------------------------------------
# 规则快照与试放
# ---------------------------------------------------------------------------

def rules_from_snapshot(snapshot: list[dict]) -> list[RuleView]:
    """把 run.rule_snapshot（启用规则的快照行）转为按 (priority, id) 排序的视图。"""
    return ordered_rules([{k: r[k] for k in _RULE_KEYS} for r in snapshot])


def replay_sequence(
    alerts: list[dict], rules_a: list[RuleView], rules_b: list[RuleView]
) -> tuple[list[dict], list[dict]]:
    """两侧分别以**全新冷却状态**按给定顺序重放同一序列，返回逐条评估结果。

    两侧状态相互独立，且都从空开始；evaluate 只更新各自内存中的状态，无任何 I/O。
    """
    state_a, state_b = EvalState(), EvalState()
    out_a, out_b = [], []
    for alert in alerts:
        out_a.append(evaluate(alert, rules_a, state_a))
        out_b.append(evaluate(alert, rules_b, state_b))
    return out_a, out_b


def _brief(result: Optional[dict]) -> Optional[dict]:
    if result is None:
        return None
    return {
        "decision": result["decision"],
        "rule_id": result["rule_id"],
        "rule_name": result["rule_name"],
        "reason": result["reason"],
    }


def _deviations(side: str, observed: Optional[dict], expected: dict) -> list[dict]:
    """比较一侧目标结果与原结果：决定或命中规则任一不同即视为偏离。"""
    label = _SIDE_LABEL[side]
    if observed is None:
        return [{
            "side": side, "kind": "target_missing",
            "detail": f"{label}侧重放序列中不包含目标告警，无法复现原结果",
        }]
    out: list[dict] = []
    if observed["decision"] != expected["decision"]:
        out.append({
            "side": side, "kind": "decision_changed",
            "detail": (
                f"{label}侧决定偏离：原结果为 {expected['decision']}，"
                f"重放为 {observed['decision']}"
            ),
        })
    if observed["rule_id"] != expected["rule_id"]:
        orig_rule = (
            f"#{expected['rule_id']}「{expected['rule_name']}」"
            if expected["rule_id"] is not None else "默认策略（无命中规则）"
        )
        new_rule = (
            f"#{observed['rule_id']}「{observed['rule_name']}」"
            if observed["rule_id"] is not None else "默认策略（无命中规则）"
        )
        out.append({
            "side": side, "kind": "rule_changed",
            "detail": f"{label}侧命中规则偏离：原命中 {orig_rule}，重放命中 {new_rule}",
        })
    return out


def _trial(
    seq_alerts: list[dict],
    rules_a: list[RuleView],
    rules_b: list[RuleView],
    target_id: int,
    expected: dict,
) -> dict:
    """一次删除尝试：清空两侧冷却状态重放候选序列，检查目标 (决定, 命中规则)。"""
    res_a, res_b = replay_sequence(seq_alerts, rules_a, rules_b)
    obs_a = obs_b = None
    for alert, result in zip(seq_alerts, res_a):
        if alert["id"] == target_id:
            obs_a = result
            break
    for alert, result in zip(seq_alerts, res_b):
        if alert["id"] == target_id:
            obs_b = result
            break
    deviations = (
        _deviations("a", obs_a, expected["a"])
        + _deviations("b", obs_b, expected["b"])
    )
    return {
        "passes": not deviations,
        "deviations": deviations,
        "observed": {"a": _brief(obs_a), "b": _brief(obs_b)},
    }


# ---------------------------------------------------------------------------
# ddmin
# ---------------------------------------------------------------------------

def _split_chunks(seq: list[Any], n: int) -> list[list[Any]]:
    """把序列尽量均分为 n 个连续块（前余数块多 1 个元素），与经典 ddmin 一致。"""
    k, m = divmod(len(seq), n)
    chunks, start = [], 0
    for i in range(n):
        size = k + (1 if i < m else 0)
        if size:
            chunks.append(seq[start:start + size])
        start += size
    return chunks


def _seqs(alerts: list[dict]) -> list[int]:
    return [a["seq"] for a in alerts]


def _dev_text(trial_result: dict) -> str:
    return "；".join(d["detail"] for d in trial_result["deviations"]) or "目标结果与原结果一致"


def _ddmin(
    prefix: list[dict],
    rules_a: list[RuleView],
    rules_b: list[RuleView],
    target_id: int,
    target_seq: int,
    expected: dict,
) -> tuple[list[dict], list[dict], int]:
    """对目标前缀执行 ddmin，返回 (最小序列, 逐轮日志, 试放次数)。

    判据（test predicate）：删除后分别在两侧清空冷却状态重放，目标的
    (decision, rule_id) 与原结果**两侧都一致**才算通过（= 可删）。目标所在的块
    永不参与删除；所有非目标块都删不掉时放大粒度；粒度到序列长度仍无可删块，
    即"再删任一告警都会偏离"。
    """
    logs: list[dict] = []
    current = list(prefix)
    tests = 0
    round_no = 0

    while True:
        n = 2
        while len(current) >= 2:
            round_no += 1
            chunks = _split_chunks(current, n)
            progressed = False
            for idx, chunk in enumerate(chunks, start=1):
                chunk_ids = {a["id"] for a in chunk}
                if target_id in chunk_ids:
                    logs.append({
                        "round": round_no, "phase": "remove", "granularity": n,
                        "sequence_size": len(current),
                        "chunk_index": idx, "chunk_count": len(chunks),
                        "chunk_seqs": _seqs(chunk),
                        "action": "skipped_target",
                        "reason": (
                            f"第 {idx}/{len(chunks)} 块（#{'、#'.join(map(str, _seqs(chunk)))}）"
                            f"包含目标告警 #{target_seq}，目标本身不可删除，跳过该块"
                        ),
                        "deviations": [], "observed": None,
                    })
                    continue
                trial_seq = [a for a in current if a["id"] not in chunk_ids]
                tests += 1
                trial_result = _trial(trial_seq, rules_a, rules_b, target_id, expected)
                if trial_result["passes"]:
                    logs.append({
                        "round": round_no, "phase": "remove", "granularity": n,
                        "sequence_size": len(current),
                        "chunk_index": idx, "chunk_count": len(chunks),
                        "chunk_seqs": _seqs(chunk),
                        "trial_seqs": _seqs(trial_seq),
                        "action": "removed",
                        "passes": True,
                        "reason": (
                            f"删除第 {idx}/{len(chunks)} 块（#{'、#'.join(map(str, _seqs(chunk)))}）后，"
                            f"目标 #{target_seq} 在两侧的决定与命中规则仍与原结果一致，"
                            f"该块可删：序列 {len(current)} → {len(trial_seq)} 条，"
                            f"粒度回落至 {max(n - 1, 2)} 并重新开始"
                        ),
                        "deviations": [], "observed": trial_result["observed"],
                    })
                    current = trial_seq
                    n = max(n - 1, 2)
                    progressed = True
                    break
                logs.append({
                    "round": round_no, "phase": "remove", "granularity": n,
                    "sequence_size": len(current),
                    "chunk_index": idx, "chunk_count": len(chunks),
                    "chunk_seqs": _seqs(chunk),
                    "trial_seqs": _seqs(trial_seq),
                    "action": "retained",
                    "passes": False,
                    "reason": (
                        f"尝试删除第 {idx}/{len(chunks)} 块"
                        f"（#{'、#'.join(map(str, _seqs(chunk)))}）：{_dev_text(trial_result)}"
                        f"——该块必须保留，继续尝试下一块"
                    ),
                    "deviations": trial_result["deviations"],
                    "observed": trial_result["observed"],
                })
            if progressed:
                continue
            if n < len(current):
                n2 = min(n * 2, len(current))
                logs.append({
                    "round": round_no, "phase": "granularity", "granularity": n,
                    "next_granularity": n2, "sequence_size": len(current),
                    "action": "increase",
                    "reason": (
                        f"第 {round_no} 轮中所有非目标块单独删除都会使目标偏离原结果，"
                        f"放粗切分粒度 n={n} → {n2}，用更小的块继续尝试"
                    ),
                    "deviations": [], "observed": None,
                })
                n = n2
                continue
            break

        # 最小性复核：主循环终止时 n==len(current)，每个非目标成员都已作为单元素块
        # 试过删除并被保留（len==1 时只剩目标）。这里再逐条复核一遍并记录"保留原因"；
        # 若谓词因非单调性意外放行了某个成员，则删除后重新进入块级 ddmin。
        round_no += 1
        removed_any = False
        for alert in current:
            if alert["id"] == target_id:
                logs.append({
                    "round": round_no, "phase": "minimality",
                    "sequence_size": len(current), "chunk_seqs": [alert["seq"]],
                    "action": "target",
                    "reason": "选定的目标告警是差异复现的观察对象，必须保留",
                    "deviations": [], "observed": None,
                })
                continue
            trial_seq = [a for a in current if a["id"] != alert["id"]]
            tests += 1
            trial_result = _trial(trial_seq, rules_a, rules_b, target_id, expected)
            if trial_result["passes"]:
                logs.append({
                    "round": round_no, "phase": "minimality",
                    "sequence_size": len(current), "chunk_seqs": [alert["seq"]],
                    "trial_seqs": _seqs(trial_seq),
                    "action": "removed", "passes": True,
                    "reason": (
                        f"最小性复核：删除 #{alert['seq']} 后目标仍可复现，"
                        f"序列 {len(current)} → {len(trial_seq)} 条，重新进入 ddmin"
                    ),
                    "deviations": [], "observed": trial_result["observed"],
                })
                current = trial_seq
                removed_any = True
                break
            logs.append({
                "round": round_no, "phase": "minimality",
                "sequence_size": len(current), "chunk_seqs": [alert["seq"]],
                "trial_seqs": _seqs(trial_seq),
                "action": "retained_final", "passes": False,
                "reason": (
                    f"最小性复核：再删 #{alert['seq']} 则 {_dev_text(trial_result)}"
                    f"——#{alert['seq']} 是复现差异所必需的告警，保留"
                ),
                "deviations": trial_result["deviations"],
                "observed": trial_result["observed"],
            })
        if removed_any:
            continue

        logs.append({
            "round": round_no, "phase": "done", "granularity": n,
            "sequence_size": len(current), "action": "final",
            "reason": (
                f"最小化完成：保留 {len(current)} 条告警（含目标 #{target_seq}），"
                "再删任一告警都会使目标在任一侧的决定或命中规则偏离原结果"
            ),
            "deviations": [], "observed": None,
        })
        return current, logs, tests


# ---------------------------------------------------------------------------
# 序列化与复现包
# ---------------------------------------------------------------------------

def _package_alert(alert: dict) -> dict:
    ts = alert["ts"]
    return {
        "seq": alert["seq"],
        "device_id": alert["device_id"],
        "batch_id": alert["batch_id"],
        "level": alert["level"],
        "alert_type": alert["alert_type"],
        "message": alert["message"],
        "ts": ts.isoformat(timespec="seconds") if isinstance(ts, datetime) else str(ts),
    }


def _compact_rule(rule: dict) -> dict:
    return {
        "id": rule["id"],
        "name": rule["name"],
        "action": rule["action"],
        "priority": rule["priority"],
        "cooldown_seconds": rule["cooldown_seconds"],
        "match_device": rule["match_device"] or "*",
        "match_batch": rule["match_batch"] or "*",
        "match_level": rule["match_level"] or "*",
        "time_window": (
            f"{rule['time_start']}-{rule['time_end']}" if rule["time_start"] else "全天"
        ),
        "date_range": (
            f"{rule['date_start'] or '…'} ~ {rule['date_end'] or '…'}"
            if (rule["date_start"] or rule["date_end"]) else "不限"
        ),
    }


def _snapshot_section(run_id: int, snapshot: list[dict]) -> dict:
    return {
        "run_id": run_id,
        "rule_count": len(snapshot),
        "rules": [{k: r[k] for k in _RULE_KEYS} for r in snapshot],
    }


def build_package(
    prefix: list[dict],
    run_a_id: int,
    run_b_id: int,
    snapshot_a: list[dict],
    snapshot_b: list[dict],
    target: dict,
    expected: dict,
    source_meta: dict,
    exported_at: Optional[datetime] = None,
) -> dict:
    """基准校验 + ddmin 最小化 + 组装可导出/可重新验证的复现包。

    ``prefix`` 为按 seq 升序、截至目标（含）的告警（ts 为 datetime）；
    ``expected`` 为目标在两个 run 中存储的原结果
    ``{"a": {decision, rule_id, rule_name, reason, run_id}, "b": {...}}``。
    基准重放（全前缀、两侧全新冷却状态）若不能复现原结果，抛 BaselineMismatch。
    """
    rules_a = rules_from_snapshot(snapshot_a)
    rules_b = rules_from_snapshot(snapshot_b)
    target_id = target["id"]
    target_seq = target["seq"]

    logs: list[dict] = [{
        "round": 0, "phase": "baseline", "granularity": None,
        "sequence_size": len(prefix), "action": "baseline",
        "chunk_seqs": _seqs(prefix),
        "reason": (
            f"基准校验：以目标前缀（{len(prefix)} 条，#1 … #{target_seq}）分别载入两个 run "
            "的规则快照，两侧均清空冷却状态，按原 seq/ts 重放"
        ),
        "deviations": [], "observed": None,
    }]
    baseline = _trial(prefix, rules_a, rules_b, target_id, expected)
    if not baseline["passes"]:
        raise BaselineMismatch(baseline["deviations"], baseline["observed"])
    logs[0]["passes"] = True
    logs[0]["observed"] = baseline["observed"]
    logs[0]["reason"] += "——目标在两侧的决定与命中规则均与原结果一致，最小化前提成立"

    minimal, ddmin_logs, tests = _ddmin(
        prefix, rules_a, rules_b, target_id, target_seq, expected
    )
    logs.extend(ddmin_logs)
    rounds = max((e.get("round", 0) for e in logs), default=0)

    exported_at = exported_at or datetime.now()
    return {
        "package_type": PACKAGE_TYPE,
        "schema_version": SCHEMA_VERSION,
        "exported_at": exported_at.isoformat(timespec="seconds"),
        "source": source_meta,
        "target": {
            "alert_id": target_id,
            "seq": target_seq,
            "device_id": target["device_id"],
            "batch_id": target["batch_id"],
            "level": target["level"],
            "alert_type": target["alert_type"],
            "message": target["message"],
            "ts": target["ts"].isoformat(timespec="seconds")
            if isinstance(target["ts"], datetime) else str(target["ts"]),
        },
        "expected": {
            "a": dict(expected["a"]),
            "b": dict(expected["b"]),
        },
        "rule_snapshots": {
            "a": _snapshot_section(run_a_id, snapshot_a),
            "b": _snapshot_section(run_b_id, snapshot_b),
        },
        "rule_snapshot_summary": {
            "a": {"run_id": run_a_id, "rule_count": len(snapshot_a),
                  "rules": [_compact_rule(r) for r in snapshot_a]},
            "b": {"run_id": run_b_id, "rule_count": len(snapshot_b),
                  "rules": [_compact_rule(r) for r in snapshot_b]},
        },
        "minimal_sequence": [_package_alert(a) for a in minimal],
        "minimal_count": len(minimal),
        "candidate_count": len(prefix),
        "ddmin": {
            "algorithm": "ddmin (delta-debugging minimize, 连续块切分)",
            "rounds": rounds,
            "replay_tests": tests + 1,  # 含基准校验
            "removed_count": len(prefix) - len(minimal),
            "logs": logs,
        },
        "replay_semantics": (
            "每一轮试放都分别载入 run A / run B 的规则快照，两侧均从空的冷却状态开始，"
            "按 minimal_sequence 的数组顺序（即原 seq 顺序）以告警自身 ts 重放；"
            "全程内存计算，不写入 alerts/decisions/runs，不触发任何转发"
        ),
    }


# ---------------------------------------------------------------------------
# 只读重新验证（可对导入的复现包离线执行）
# ---------------------------------------------------------------------------

def _load_package(pkg: Any) -> dict:
    """校验复现包结构并解析出规则视图、告警序列与目标位置，失败抛 ValueError。"""
    if not isinstance(pkg, dict):
        raise ValueError("请求体必须是差异复现包 JSON 对象")
    if pkg.get("package_type") != PACKAGE_TYPE:
        raise ValueError(f"不是 {PACKAGE_TYPE} 类型的复现包（package_type 不匹配）")
    if pkg.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"不支持的复现包版本：{pkg.get('schema_version')!r}，当前支持 {SCHEMA_VERSION}"
        )

    target = pkg.get("target")
    if not isinstance(target, dict) or target.get("seq") is None:
        raise ValueError("复现包缺少 target.seq")
    try:
        target_seq = int(target["seq"])
    except (TypeError, ValueError):
        raise ValueError("target.seq 必须是整数")

    snaps = pkg.get("rule_snapshots")
    expected = pkg.get("expected")
    items = pkg.get("minimal_sequence")
    if not isinstance(snaps, dict) or not isinstance(expected, dict):
        raise ValueError("复现包缺少 rule_snapshots 或 expected 段")
    for side in ("a", "b"):
        sec = snaps.get(side)
        exp = expected.get(side)
        if not isinstance(sec, dict) or not isinstance(sec.get("rules"), list):
            raise ValueError(f"rule_snapshots.{side} 缺少规则快照列表")
        if not isinstance(exp, dict) or exp.get("decision") not in ("pass", "suppress"):
            raise ValueError(f"expected.{side} 缺少合法的期望决定")

    if not isinstance(items, list) or not items:
        raise ValueError("minimal_sequence 必须是非空告警数组")

    alerts: list[dict] = []
    target_index: Optional[int] = None
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"最小序列第 {idx + 1} 项不是 JSON 对象")
        if not item.get("ts"):
            raise ValueError(f"最小序列第 {idx + 1} 项（#{item.get('seq')}）缺少 ts，无法按原时间重放")
        try:
            alert = normalize_alert(item, seq=int(item.get("seq", idx + 1)),
                                    fallback_ts=datetime.now())
        except ValueError as exc:
            raise ValueError(f"最小序列第 {idx + 1} 项告警无效：{exc}")
        alert["id"] = idx + 1  # 包内位置 id，仅用于试放时定位目标
        alerts.append(alert)
        if alert["seq"] == target_seq:
            target_index = idx
    if target_index is None:
        raise ValueError(f"最小序列中找不到目标告警 #{target_seq}")

    return {
        "alerts": alerts,
        "target_index": target_index,
        "target_seq": target_seq,
        "rules_a": rules_from_snapshot(snaps["a"]["rules"]),
        "rules_b": rules_from_snapshot(snaps["b"]["rules"]),
        "snapshots": snaps,
        "expected": expected,
    }


def verify_package(pkg: Any) -> dict:
    """对复现包做**只读重新验证**：载入包内两份规则快照、清空冷却状态重放最小序列。

    不访问数据库、不写任何表、不转发。返回目标观察结果、偏离清单与逐条重放行。
    """
    ctx = _load_package(pkg)
    alerts = ctx["alerts"]
    res_a, res_b = replay_sequence(alerts, ctx["rules_a"], ctx["rules_b"])
    target_idx = ctx["target_index"]
    obs_a, obs_b = res_a[target_idx], res_b[target_idx]
    deviations = (
        _deviations("a", obs_a, ctx["expected"]["a"])
        + _deviations("b", obs_b, ctx["expected"]["b"])
    )

    rows = []
    for idx, (alert, ra, rb) in enumerate(zip(alerts, res_a, res_b)):
        rows.append({
            "seq": alert["seq"],
            "device_id": alert["device_id"],
            "batch_id": alert["batch_id"],
            "level": alert["level"],
            "alert_type": alert["alert_type"],
            "message": alert["message"],
            "ts": alert["ts"].isoformat(timespec="seconds"),
            "is_target": idx == target_idx,
            "a": _brief(ra),
            "b": _brief(rb),
        })

    return {
        "valid": not deviations,
        "target_seq": ctx["target_seq"],
        "sequence_length": len(alerts),
        "rules_loaded": {
            "a": len(ctx["snapshots"]["a"]["rules"]),
            "b": len(ctx["snapshots"]["b"]["rules"]),
        },
        "expected": {"a": _exp_brief(ctx["expected"]["a"]),
                     "b": _exp_brief(ctx["expected"]["b"])},
        "observed": {"a": _brief(obs_a), "b": _brief(obs_b)},
        "deviations": deviations,
        "rows": rows,
        "read_only": True,
    }


def _exp_brief(exp: dict) -> dict:
    return {
        "decision": exp["decision"],
        "rule_id": exp["rule_id"],
        "rule_name": exp["rule_name"],
        "reason": exp.get("reason"),
    }
