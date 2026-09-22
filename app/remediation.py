"""规则审计修复建议引擎（纯函数，无副作用，不依赖 FastAPI / SQLite）。

基于 :func:`app.audit.audit_rules` 的审计关系，为**动作冲突**、**完全遮蔽**
与**冗余重叠**三类问题生成确定性的合并 / 收窄 / 删除 / 停用建议，并提供
应用后校验与应用前后审计关系对比的纯函数。

确定性保证
----------
建议只取决于启用规则集合本身：同一规则集在任何时刻生成的建议完全一致
（建议 id、操作序列、说明文本均确定），与告警数据、live/replay 历史无关。
本模块不读写任何数据库表。

建议类型（每条审计关系至多一条）
------------------------------
- ``delete_rule``   完全遮蔽：低优先级规则的可达域被高优先级规则逐维完全
                    包含，永远不会命中任何告警（死规则），删除不改变评估结果；
- ``merge_rules``   冗余重叠且两规则可达域的并集能**精确**表达为一条规则：
                    高优先级规则条件扩为并集，删除低优先级规则；
- ``narrow_rule``   动作冲突 / 冗余重叠：把低优先级规则的时段或日期收窄为
                    "原可达域 − 交集"，收窄后两规则不再相交，且交集内原本就
                    由高优先级规则决定结果，因此不改变评估结果；
- ``disable_rule``  以上都不可精确表达时的确定性兜底：停用低优先级规则
                    （会改变交集外的评估结果，文案中明确提示，由用户勾选确认）。

冷却差异（cooldown_conflict）需要人工选择保留哪个冷却窗，不生成自动建议，
在结果中以 ``unsupported`` 单列（除非它同时构成完全遮蔽，此时按完全遮蔽
给出删除建议）。

影响预演（草稿规则）
--------------------
:func:`plan_draft` 在不落库的前提下，按与正式应用**完全相同的确定性顺序**
把勾选建议应用到内存中的规则副本上，生成草稿规则与重审计结果，供
"影响预演"分别用当前规则与草稿规则从零回放全部已注入告警。
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Optional, Sequence

from . import audit
from .audit import DAY_MINUTES, TimeDomain
from .engine import normalize_rule

KIND_LABELS = {
    "delete_rule": "删除死规则",
    "merge_rules": "合并规则",
    "narrow_rule": "收窄规则",
    "disable_rule": "停用规则",
}

_FIELD_DIMS = ("device", "batch", "level")
_FIELD_COLS = {"device": "match_device", "batch": "match_batch", "level": "match_level"}
_DIM_ORDER = ("device", "batch", "level", "time", "date")


# ---------------------------------------------------------------------------
# 关系指纹与时段窗口工具
# ---------------------------------------------------------------------------

def relation_fingerprint(rel: dict) -> str:
    """审计关系的稳定指纹：高优先级 id | 低优先级 id | 冲突类型。"""
    return f"{rel['higher_rule_id']}|{rel['lower_rule_id']}|{rel['conflict_type']}"


def _hm(minute: int) -> str:
    return f"{minute // 60:02d}:{minute % 60:02d}"


def _window_from_segments(segments) -> Optional[tuple[str, str]]:
    """把一天内的分钟区间集合转换为规则可表达的每日时段窗口。

    返回 ``(time_start, time_end)``；无法精确表达（空集、单点区间、两个以上
    不相邻区间等）时返回 None。
    """
    segs = sorted(segments)
    if not segs:
        return None
    if any(s == e for s, e in segs):
        return None  # 单点区间无法表达（规则里起止相等表示全天）
    if len(segs) == 1:
        s, e = segs[0]
        if (s, e) == (0, DAY_MINUTES):
            return "", ""  # 全天
        if s == 0:
            return "00:00", _hm(e)
        if e == DAY_MINUTES:
            return _hm(s), "00:00"  # 到午夜（00:00 与 24:00 是同一时刻）
        return _hm(s), _hm(e)
    if len(segs) == 2:
        (s1, e1), (s2, e2) = segs
        if s1 == 0 and e2 == DAY_MINUTES:
            return _hm(s2), _hm(e1)  # 跨午夜窗口
    return None


def _canon_segments(segments) -> set:
    """规范化分钟区间集合：24:00 与 00:00 是同一时刻。

    跨午夜窗口（如 22:00-00:00）重新解析后会多出单点 (0,0)，比较前剔除，
    避免对"精确表达"的误判。
    """
    s = set(segments)
    if (0, 0) in s and any(e == DAY_MINUTES for _, e in s):
        s.discard((0, 0))
    return s


def _window_exact(window: tuple[str, str], segments) -> bool:
    """校验窗口重新解析后的可达分钟与目标区间集合完全一致（精确表达）。"""
    dom = audit.rule_time_domain(window[0], window[1])
    return _canon_segments(dom.segments) == _canon_segments(segments)


def _subtract_segments(lo_segs, hi_segs) -> list:
    """分钟闭区间差集 lo − hi，结果合并为互不重叠的区间列表。"""
    result: list[tuple[int, int]] = []
    for s, e in sorted(lo_segs):
        cursor = s
        for hs, he in sorted(hi_segs):
            if he < cursor or hs > e:
                continue
            if hs > cursor:
                result.append((cursor, hs - 1))
            cursor = max(cursor, he + 1)
            if cursor > e:
                break
        if cursor <= e:
            result.append((cursor, e))
    return audit._merge_segments(result)


# ---------------------------------------------------------------------------
# 日期范围工具
# ---------------------------------------------------------------------------

def _shift_day(value: str, days: int) -> str:
    return (date.fromisoformat(value) + timedelta(days=days)).isoformat()


def _date_remainder(lo_start: str, lo_end: str,
                    hi_start: str, hi_end: str) -> Optional[tuple[str, str]]:
    """lo 日期范围减去 hi 日期范围；仅当差集可表达为单个范围时返回它。

    返回 None 表示：lo 被完全覆盖（0 段）或差集是左右两段（单条规则无法表达）。
    """
    pieces: list[tuple[str, str]] = []
    if hi_start:  # hi 有下界 → 左段 [lo_start, hi_start − 1 天]
        left_end = _shift_day(hi_start, -1)
        if not lo_start or lo_start <= left_end:
            pieces.append((lo_start, left_end))
    if hi_end:  # hi 有上界 → 右段 [hi_end + 1 天, lo_end]
        right_start = _shift_day(hi_end, 1)
        if not lo_end or right_start <= lo_end:
            pieces.append((right_start, lo_end))
    valid = [p for p in pieces if not (p[0] and p[1] and p[0] > p[1])]
    if len(valid) != 1:
        return None
    return valid[0]


# ---------------------------------------------------------------------------
# 收窄与合并的精确性计算
# ---------------------------------------------------------------------------

def _narrow_option(hi: dict, lo: dict) -> Optional[dict]:
    """低优先级规则的精确收窄方案（先时段后日期，确定性顺序）。

    把任一维度收窄到与 hi 不相交即可消除整条关系；收窄只缩小可达域，
    不会与其他规则产生新的交集。成功返回 {"dim", "set", "text"}。
    """
    remainder = _subtract_segments(lo["time"].segments, hi["time"].segments)
    if remainder and _canon_segments(remainder) != _canon_segments(lo["time"].segments):
        window = _window_from_segments(remainder)
        if window is not None and _window_exact(window, remainder):
            text = audit.describe_time(TimeDomain(tuple(remainder)))
            return {
                "dim": "time",
                "set": {"time_start": window[0], "time_end": window[1]},
                "text": f"每日时段收窄为 {text}",
            }
    piece = _date_remainder(lo["date_start"], lo["date_end"],
                            hi["date_start"], hi["date_end"])
    if piece is not None and piece != (lo["date_start"], lo["date_end"]):
        return {
            "dim": "date",
            "set": {"date_start": piece[0], "date_end": piece[1]},
            "text": f"日期范围收窄为 {audit.describe_date(*piece)}",
        }
    return None


def _conditions_payload(norm: dict) -> dict:
    """规范化规则 → 规则表条件列（match_* / time_* / date_*）。"""
    window = _window_from_segments(norm["time"].segments) or ("", "")
    return {
        "match_device": norm["device"] or "",
        "match_batch": norm["batch"] or "",
        "match_level": norm["level"] or "",
        "time_start": window[0],
        "time_end": window[1],
        "date_start": norm["date_start"],
        "date_end": norm["date_end"],
    }


def _conditions_text(c: dict) -> str:
    time_text = audit.describe_time(audit.rule_time_domain(c["time_start"], c["time_end"]))
    date_text = audit.describe_date(c["date_start"], c["date_end"])
    return (
        f"设备 {c['match_device'] or '*'} · 批次 {c['match_batch'] or '*'} · "
        f"级别 {c['match_level'] or '*'} · 时段 {time_text} · 日期 {date_text}"
    )


def _dim_rel(hi_covers: bool, lo_covers: bool) -> str:
    if hi_covers and lo_covers:
        return "equal"
    if hi_covers:
        return "hi_wider"
    if lo_covers:
        return "lo_wider"
    return "incomparable"


def _dim_relations(hi: dict, lo: dict) -> dict:
    rels = {
        dim: _dim_rel(audit.field_covers(hi[dim], lo[dim]),
                      audit.field_covers(lo[dim], hi[dim]))
        for dim in _FIELD_DIMS
    }
    rels["time"] = _dim_rel(audit.time_covers(hi["time"], lo["time"]),
                            audit.time_covers(lo["time"], hi["time"]))
    rels["date"] = _dim_rel(
        audit.date_covers(hi["date_start"], hi["date_end"],
                          lo["date_start"], lo["date_end"]),
        audit.date_covers(lo["date_start"], lo["date_end"],
                          hi["date_start"], hi["date_end"]),
    )
    return rels


def _merge_conditions(hi: dict, lo: dict) -> Optional[dict]:
    """两规则可达域的精确并集（可表达为一条规则时返回条件列，否则 None）。

    精确性判定：
    - 一侧在所有维度都覆盖另一侧 → 并集就是较宽一侧的条件；
    - 只有一个维度不同且该维度的并集可精确表达（字段维一侧通配、时段并集
      落回单个每日窗口、日期范围相交必连续）→ 取该维度的并集；
    - 两个以上维度朝不同方向各自更宽（"L 形"并集）或字段维是两个不同的
      精确值 → 无法用单条规则精确表达，返回 None。
    """
    rels = _dim_relations(hi, lo)
    non_equal = [d for d in _DIM_ORDER if rels[d] != "equal"]
    if not non_equal:
        return None  # 条件完全相同（属于完全遮蔽，不走合并）
    if all(rels[d] == "lo_wider" for d in non_equal):
        return _conditions_payload(lo)  # 并集 = lo 的可达域
    if all(rels[d] == "hi_wider" for d in non_equal):
        return _conditions_payload(hi)  # 并集 = hi 的可达域
    if len(non_equal) == 1:
        dim = non_equal[0]  # 此时该维度必为 incomparable
        if dim in _FIELD_DIMS:
            return None  # 两个不同的精确值，并集无法用单条规则表达
        if dim == "time":
            union = audit._merge_segments(
                list(hi["time"].segments) + list(lo["time"].segments)
            )
            window = _window_from_segments(union)
            if window is None or not _window_exact(window, union):
                return None
            merged = _conditions_payload(hi)
            merged["time_start"], merged["time_end"] = window
            return merged
        # date：关系存在 ⇒ 日期范围相交 ⇒ 并集必为连续范围
        merged = _conditions_payload(hi)
        merged["date_start"] = (
            "" if not hi["date_start"] or not lo["date_start"]
            else min(hi["date_start"], lo["date_start"])
        )
        merged["date_end"] = (
            "" if not hi["date_end"] or not lo["date_end"]
            else max(hi["date_end"], lo["date_end"])
        )
        return merged
    return None  # 多维不同向：并集是 L 形，无法用单条规则精确表达


# ---------------------------------------------------------------------------
# 建议生成
# ---------------------------------------------------------------------------

def _base(rel: dict) -> dict:
    return {
        "relation_fingerprint": relation_fingerprint(rel),
        "conflict_type": rel["conflict_type"],
        "conflict_label": rel["conflict_label"],
        "rule_ids": rel["rule_ids"],
        "higher_rule": rel["higher_rule"],
        "lower_rule": rel["lower_rule"],
        "intersection": rel["intersection"],
    }


def _narrow_suggestion(rel: dict, hi: dict, lo: dict, narrow: dict) -> dict:
    hi_id, lo_id = hi["id"], lo["id"]
    return {
        **_base(rel),
        "id": f"narrow_rule:{hi_id}:{lo_id}",
        "kind": "narrow_rule",
        "kind_label": KIND_LABELS["narrow_rule"],
        "target_rule_ids": [lo_id],
        "summary": f"收窄 #{lo_id}「{lo['name']}」：{narrow['text']}，消除与 #{hi_id} 的交集",
        "detail": (
            f"交集内原本就由高优先级的 #{hi_id} 决定结果；把 #{lo_id} 的可达域减去交集后，"
            "两条规则不再相交，交集外的行为保持不变，因此不改变任何评估结果。"
        ),
        "ops": [{"op": "update_rule", "rule_id": lo_id, "set": narrow["set"]}],
        "ops_text": [f"更新 #{lo_id}「{lo['name']}」：{narrow['text']}"],
    }


def _disable_suggestion(rel: dict, hi: dict, lo: dict) -> dict:
    hi_id, lo_id = hi["id"], lo["id"]
    return {
        **_base(rel),
        "id": f"disable_rule:{hi_id}:{lo_id}",
        "kind": "disable_rule",
        "kind_label": KIND_LABELS["disable_rule"],
        "target_rule_ids": [lo_id],
        "summary": f"停用 #{lo_id}「{lo['name']}」（交集无法精确合并/收窄）",
        "detail": (
            f"#{lo_id} 与 #{hi_id} 的交集无法通过单条规则的合并或收窄精确消除；"
            f"停用 #{lo_id} 可彻底消除该关系，但交集外原本由 #{lo_id} 覆盖的告警"
            "将改由其他规则或默认策略处理，请确认后勾选。"
        ),
        "ops": [{"op": "disable_rule", "rule_id": lo_id}],
        "ops_text": [f"停用规则 #{lo_id}「{lo['name']}」（enabled → 停用）"],
    }


def _suggestion_for(rel: dict, hi: dict, lo: dict) -> Optional[dict]:
    """为一条审计关系生成至多一条确定性建议；不支持的类型返回 None。"""
    hi_id, lo_id = hi["id"], lo["id"]

    if rel["fully_shadowed"]:
        # 完全遮蔽（无论冲突类型）：低优先级规则永远不会命中，删除无影响
        return {
            **_base(rel),
            "id": f"delete_rule:{hi_id}:{lo_id}",
            "kind": "delete_rule",
            "kind_label": KIND_LABELS["delete_rule"],
            "target_rule_ids": [lo_id],
            "summary": f"删除被 #{hi_id} 完全遮蔽的死规则 #{lo_id}「{lo['name']}」",
            "detail": (
                f"#{lo_id} 的可达域被 #{hi_id} 逐维完全包含：任何能命中 #{lo_id} 的告警"
                f"都会先命中 #{hi_id}，#{lo_id} 永远不会成为命中规则；删除它不改变任何评估结果。"
            ),
            "ops": [{"op": "delete_rule", "rule_id": lo_id}],
            "ops_text": [f"删除规则 #{lo_id}「{lo['name']}」（死规则，评估结果不变）"],
        }

    if rel["conflict_type"] == "redundant_overlap":
        merged = _merge_conditions(hi, lo)
        if merged is not None:
            desc = (hi["description"] + f"［已合并 #{lo_id}「{lo['name']}」］").strip()
            cond_text = _conditions_text(merged)
            return {
                **_base(rel),
                "id": f"merge_rules:{hi_id}:{lo_id}",
                "kind": "merge_rules",
                "kind_label": KIND_LABELS["merge_rules"],
                "target_rule_ids": [hi_id, lo_id],
                "summary": f"把 #{lo_id}「{lo['name']}」并入 #{hi_id}「{hi['name']}」（条件取精确并集）",
                "detail": (
                    "两条规则动作与冷却一致，可达域的并集可精确表达为一条规则："
                    f"#{hi_id} 的条件扩为并集（{cond_text}），#{lo_id} 删除。"
                    "对这两条规则自身而言评估结果等价；与第三条规则的潜在交互由应用后校验把关。"
                ),
                "ops": [
                    {"op": "update_rule", "rule_id": hi_id,
                     "set": {**merged, "description": desc}},
                    {"op": "delete_rule", "rule_id": lo_id},
                ],
                "ops_text": [
                    f"更新 #{hi_id}「{hi['name']}」条件为并集：{cond_text}",
                    f"删除 #{lo_id}「{lo['name']}」（可达域已并入 #{hi_id}）",
                ],
            }
        narrow = _narrow_option(hi, lo)
        if narrow is not None:
            return _narrow_suggestion(rel, hi, lo, narrow)
        return _disable_suggestion(rel, hi, lo)

    if rel["conflict_type"] == "action_conflict":
        narrow = _narrow_option(hi, lo)
        if narrow is not None:
            return _narrow_suggestion(rel, hi, lo, narrow)
        return _disable_suggestion(rel, hi, lo)

    return None  # cooldown_conflict：需人工决策，不生成自动建议


def generate_suggestions(audit_result: dict, rule_objects: Sequence[Any]) -> dict:
    """根据审计结果生成确定性修复建议（纯函数，不读写数据库）。

    入参为 :func:`app.audit.audit_rules` 的返回与原始规则对象序列；
    返回 {"suggestions": [...], "unsupported": [...]}，顺序与审计关系一致
    （按双方规则 id 稳定排序）。
    """
    norm: dict[int, dict] = {}
    for r in rule_objects:
        if int(audit._get(r, "enabled", 1)):
            n = audit._norm_rule(r)
            norm[n["id"]] = n

    suggestions: list[dict] = []
    unsupported: list[dict] = []
    for rel in audit_result["overlaps"]:
        hi = norm.get(rel["higher_rule_id"])
        lo = norm.get(rel["lower_rule_id"])
        sugg = _suggestion_for(rel, hi, lo) if hi and lo else None
        if sugg is not None:
            suggestions.append(sugg)
        else:
            unsupported.append({
                "relation_fingerprint": relation_fingerprint(rel),
                "conflict_type": rel["conflict_type"],
                "conflict_label": rel["conflict_label"],
                "rule_ids": rel["rule_ids"],
                "intersection": rel["intersection"],
                "reason": "冷却差异需要人工决策保留哪个冷却窗，暂不提供自动修复建议",
            })
    return {"suggestions": suggestions, "unsupported": unsupported}


# ---------------------------------------------------------------------------
# 应用后校验与应用前后对比（纯函数）
# ---------------------------------------------------------------------------

def validate_remediation(before: dict, after: dict, selected: list[dict]) -> list[str]:
    """应用后校验，返回错误列表（空列表 = 通过；任一失败应整体回滚）。

    校验项：
    1. 勾选的每条建议针对的审计关系在应用后必须全部消除；
    2. 不得新增可能改变结果的关系（动作冲突 / 冷却差异）；
    3. 不得新增被完全遮蔽的规则。
    """
    errors: list[str] = []
    after_fps = {relation_fingerprint(o) for o in after["overlaps"]}
    for s in selected:
        fp = s["relation_fingerprint"]
        if fp in after_fps:
            errors.append(f"建议 {s['id']} 针对的关系（{fp}）在应用后仍然存在")

    def mco_fps(a: dict) -> set:
        return {relation_fingerprint(o) for o in a["overlaps"] if o["may_change_outcome"]}

    new_mco = sorted(mco_fps(after) - mco_fps(before))
    if new_mco:
        errors.append("应用后新增可能改变结果的审计关系：" + "、".join(new_mco))

    before_dead = set(before["totals"]["fully_shadowed_rules"])
    after_dead = set(after["totals"]["fully_shadowed_rules"])
    new_dead = sorted(after_dead - before_dead)
    if new_dead:
        errors.append("应用后新增被完全遮蔽的规则：#" + "、#".join(str(i) for i in new_dead))
    return errors


def diff_audits(before: dict, after: dict) -> dict:
    """应用前后审计关系的确定性变化（按规则 id 稳定排序）。"""
    def rel_map(a: dict) -> dict:
        return {relation_fingerprint(o): o for o in a["overlaps"]}

    b, a = rel_map(before), rel_map(after)

    def brief(o: dict) -> dict:
        return {
            "fingerprint": relation_fingerprint(o),
            "rule_ids": o["rule_ids"],
            "conflict_type": o["conflict_type"],
            "conflict_label": o["conflict_label"],
            "may_change_outcome": o["may_change_outcome"],
            "intersection_summary": o["intersection"]["summary"],
        }

    def sort_key(item: dict):
        return (min(item["rule_ids"]), max(item["rule_ids"]), item["conflict_type"])

    removed = sorted((brief(o) for fp, o in b.items() if fp not in a), key=sort_key)
    added = sorted((brief(o) for fp, o in a.items() if fp not in b), key=sort_key)
    return {
        "before_enabled": before["enabled_rule_count"],
        "after_enabled": after["enabled_rule_count"],
        "before_totals": before["totals"],
        "after_totals": after["totals"],
        "removed": removed,
        "added": added,
        "removed_count": len(removed),
        "added_count": len(added),
    }


# ---------------------------------------------------------------------------
# 影响预演：草稿规则的确定性生成（纯函数，不落库）
# ---------------------------------------------------------------------------

def apply_ops_to_rules(rules: Sequence[Any], ops: Sequence[dict]) -> list[dict]:
    """把修复操作序列应用到规则 dict 列表的副本上，返回新的规则列表。

    与数据库写入路径语义一致，但只操作内存中的 dict：
    ``update_rule`` 使用与规则 CRUD 相同的 ``normalize_rule`` 校验；
    任何校验失败抛 ValueError，由调用方决定回绝方式。
    """
    working = [dict(r) for r in rules]
    for op in ops:
        kind = op.get("op")
        rule_id = int(op.get("rule_id"))
        idx = next((i for i, r in enumerate(working) if int(r["id"]) == rule_id), None)
        if idx is None:
            raise ValueError(f"规则 #{rule_id} 已不存在，修复无法应用")
        if kind == "delete_rule":
            working.pop(idx)
        elif kind == "disable_rule":
            working[idx]["enabled"] = 0
        elif kind == "update_rule":
            payload = {**working[idx], **(op.get("set") or {})}
            data = normalize_rule(payload)  # 与规则 CRUD 相同的校验
            working[idx].update(data)
        else:
            raise ValueError(f"未知修复操作: {kind}")
    return working


def plan_draft(rule_objects: Sequence[Any], audit_result: dict,
               suggestions: Sequence[dict], selected_ids: Sequence[str]) -> dict:
    """按实际确定性应用顺序在内存中应用勾选建议，生成草稿规则（不落库）。

    顺序与正式应用完全一致：建议按审计关系的稳定顺序（``suggestions`` 的
    给定顺序）排列；某条建议针对的关系若已被本次先前的修复消除，则标记为
    ``skipped`` 且不执行其操作。返回::

        {"ordered": 按确定性顺序排列的勾选建议,
         "results": 逐条 applied/skipped 结果,
         "rules":   草稿规则列表（dict）,
         "audit":   草稿规则的重审计结果}
    """
    wanted = set(selected_ids)
    ordered = [s for s in suggestions if s["id"] in wanted]
    working_rules = [dict(r) for r in rule_objects]
    working_audit = audit_result
    results: list[dict] = []
    for s in ordered:
        fp = s["relation_fingerprint"]
        current_fps = {relation_fingerprint(o) for o in working_audit["overlaps"]}
        if fp not in current_fps:
            results.append({
                "suggestion_id": s["id"], "kind": s["kind"],
                "status": "skipped",
                "reason": "针对的审计关系已被本次草稿中先前的修复消除",
            })
            continue
        working_rules = apply_ops_to_rules(working_rules, s["ops"])
        working_audit = audit.audit_rules(working_rules)
        results.append({
            "suggestion_id": s["id"], "kind": s["kind"],
            "status": "applied",
            "reason": "已应用：" + "；".join(s["ops_text"]),
        })
    return {"ordered": ordered, "results": results,
            "rules": working_rules, "audit": working_audit}
