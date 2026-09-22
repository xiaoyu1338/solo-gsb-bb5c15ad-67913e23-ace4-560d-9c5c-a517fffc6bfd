"""FastAPI 主应用：规则 CRUD、告警注入/撤销、策略回放对比、内置模拟接收端。

默认数据文件位于 ./data/sandbox.db，可用环境变量 SANDBOX_DATA 覆盖。
"""
from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import db
from .audit import audit_rules
from .correlation import (
    DEFAULT_GAP_SECONDS,
    compare_snapshot_items,
    correlate_events,
    event_signature,
    normalize_gap,
    parse_filter_ts,
)
from .engine import (
    LEVELS,
    EvalState,
    RuleView,
    alert_fingerprint,
    evaluate,
    normalize_alert,
    normalize_rule,
    ordered_rules,
)
from .remediation import (
    diff_audits,
    generate_suggestions,
    plan_draft,
    validate_remediation,
)

app = FastAPI(title="冷链告警抑制策略沙盒", version="1.0.0")

STATIC_DIR = Path(__file__).resolve().parent / "static"


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _rule_views(conn) -> list[RuleView]:
    rows = conn.execute("SELECT * FROM rules").fetchall()
    return ordered_rules([dict(r) for r in rows])


def _alert_dict(row) -> dict:
    return {
        "id": row["id"],
        "seq": row["seq"],
        "device_id": row["device_id"],
        "batch_id": row["batch_id"],
        "level": row["level"],
        "alert_type": row["alert_type"],
        "message": row["message"],
        "ts": row["ts"],
        "raw_json": json.loads(row["raw_json"]),
    }


def _decision_dict(row) -> dict:
    return {
        "id": row["id"],
        "run_id": row["run_id"],
        "alert_id": row["alert_id"],
        "position": row["position"],
        "decision": row["decision"],
        "rule_id": row["rule_id"],
        "rule_name": row["rule_name"],
        "reason": row["reason"],
        "fingerprint": row["fingerprint"],
        "fired_at_epoch": row["fired_at_epoch"],
        "candidates": json.loads(row["detail_json"]),
    }


def _live_run_id(conn) -> int:
    row = conn.execute("SELECT id FROM runs WHERE kind='live'").fetchone()
    return row["id"]


def _load_live_state(conn, run_id: int) -> EvalState:
    """从已有放行决定中重建冷却状态（同一规则+指纹取最近一次放行时间）。"""
    state = EvalState()
    rows = conn.execute(
        "SELECT rule_id, fingerprint, fired_at_epoch FROM decisions "
        "WHERE run_id=? AND fired_at_epoch IS NOT NULL",
        (run_id,),
    ).fetchall()
    for r in rows:
        state.seed(r["rule_id"], r["fingerprint"], r["fired_at_epoch"])
    return state


def _persist_decision(conn, run_id: int, alert_id: int, result: dict) -> int:
    position = conn.execute(
        "SELECT COALESCE(MAX(position), 0) + 1 FROM decisions WHERE run_id=?",
        (run_id,),
    ).fetchone()[0]
    cur = conn.execute(
        """INSERT INTO decisions(run_id, alert_id, position, decision, rule_id, rule_name,
           reason, fingerprint, fired_at_epoch, detail_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            run_id, alert_id, position, result["decision"], result["rule_id"],
            result["rule_name"], result["reason"], result["fingerprint"],
            result["fired_at_epoch"],
            json.dumps(result["candidates"], ensure_ascii=False),
        ),
    )
    return cur.lastrowid


def _get_settings(conn) -> dict:
    rows = conn.execute("SELECT key, value FROM settings").fetchall()
    return {r["key"]: r["value"] for r in rows}


def _deliver(conn, alert: dict, result: dict, settings: dict) -> dict:
    """把放行告警转发到接收端。replay 回放不会调用本函数。"""
    ts = alert["ts"]
    payload = {
        "alert": {
            **{k: alert[k] for k in
               ("seq", "device_id", "batch_id", "level", "alert_type", "message")},
            "ts": ts.isoformat(timespec="seconds") if isinstance(ts, datetime) else ts,
        },
        "decision": {
            "decision": result["decision"],
            "rule_id": result["rule_id"],
            "rule_name": result["rule_name"],
            "reason": result["reason"],
        },
        "forwarded_at": datetime.now().isoformat(timespec="seconds"),
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    if settings.get("receiver_mode", "mock") == "mock":
        target = "内置模拟接收端 mock-inbox"
        cur = conn.execute(
            """INSERT INTO receiver_logs(alert_id, status, target, payload_json, detail)
               VALUES (?, 'received', ?, ?, '放行告警已投递到内置模拟接收端')""",
            (alert["id"], target, json.dumps(payload, ensure_ascii=False)),
        )
        return {"delivered": True, "target": target, "log_id": cur.lastrowid}

    url = settings.get("external_url", "")
    if not url:
        raise HTTPException(400, "已选择外部 webhook 但未配置 external_url")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=3) as resp:
            detail = f"HTTP {resp.status}"
            status = "delivered"
            ok = 200 <= resp.status < 300
    except (urllib.error.URLError, OSError) as exc:
        detail = f"请求失败: {exc}"
        status = "failed"
        ok = False
    cur = conn.execute(
        """INSERT INTO receiver_logs(alert_id, status, target, payload_json, detail)
           VALUES (?, ?, ?, ?, ?)""",
        (alert["id"], status, url, json.dumps(payload, ensure_ascii=False), detail),
    )
    return {"delivered": ok, "target": url, "log_id": cur.lastrowid, "detail": detail}


def _inject_one(conn, payload: dict) -> dict:
    """规范化 -> 评估 -> 持久化 ->（可选）转发，返回完整结果。"""
    try:
        seq = conn.execute("SELECT COALESCE(MAX(seq), 0) + 1 FROM alerts").fetchone()[0]
        alert = normalize_alert(payload, seq=seq, fallback_ts=datetime.now())
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    rules = _rule_views(conn)
    run_id = _live_run_id(conn)
    state = _load_live_state(conn, run_id)
    result = evaluate(alert, rules, state)

    cur = conn.execute(
        """INSERT INTO alerts(seq, device_id, batch_id, level, alert_type, message, ts, raw_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            alert["seq"], alert["device_id"], alert["batch_id"], alert["level"],
            alert["alert_type"], alert["message"],
            alert["ts"].isoformat(timespec="seconds"),
            json.dumps(alert["raw_json"], ensure_ascii=False),
        ),
    )
    alert["id"] = cur.lastrowid
    decision_id = _persist_decision(conn, run_id, alert["id"], result)

    delivery = None
    settings = _get_settings(conn)
    if result["decision"] == "pass" and settings.get("forward_enabled") == "true":
        delivery = _deliver(conn, alert, result, settings)

    return {
        "alert": alert,
        "decision": {**result, "id": decision_id, "run_id": run_id},
        "delivery": delivery,
    }


# ---------------------------------------------------------------------------
# 启动
# ---------------------------------------------------------------------------

@app.on_event("startup")
def _startup() -> None:
    db.init_db()


@app.exception_handler(HTTPException)
def _http_exc_handler(request: Request, exc: HTTPException):  # noqa: ARG001
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})


# ---------------------------------------------------------------------------
# 元信息 / 健康检查
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    return {"status": "ok", "service": "cold-chain-alert-suppression-sandbox",
            "time": datetime.now().isoformat(timespec="seconds")}


@app.get("/api/meta")
def meta():
    return {"levels": LEVELS}


# ---------------------------------------------------------------------------
# 规则 CRUD
# ---------------------------------------------------------------------------

@app.get("/api/rules")
def list_rules():
    conn = db.get_conn()
    try:
        rows = conn.execute("SELECT * FROM rules ORDER BY priority, id").fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


@app.post("/api/rules")
def create_rule(payload: dict = Body(...)):
    try:
        data = normalize_rule(payload)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    conn = db.get_conn()
    try:
        cols = ", ".join(data)
        cur = conn.execute(
            f"INSERT INTO rules({cols}) VALUES ({', '.join(':' + k for k in data)})", data
        )
        conn.commit()
        row = conn.execute("SELECT * FROM rules WHERE id=?", (cur.lastrowid,)).fetchone()
        return dict(row)
    finally:
        conn.close()


@app.put("/api/rules/{rule_id}")
def update_rule(rule_id: int, payload: dict = Body(...)):
    try:
        data = normalize_rule(payload)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    conn = db.get_conn()
    try:
        if not conn.execute("SELECT 1 FROM rules WHERE id=?", (rule_id,)).fetchone():
            raise HTTPException(404, "规则不存在")
        sets = ", ".join(f"{k}=:{k}" for k in data)
        data["id"] = rule_id
        conn.execute(f"UPDATE rules SET {sets} WHERE id=:id", data)
        conn.commit()
        return dict(conn.execute("SELECT * FROM rules WHERE id=?", (rule_id,)).fetchone())
    finally:
        conn.close()


@app.delete("/api/rules/{rule_id}")
def delete_rule(rule_id: int):
    conn = db.get_conn()
    try:
        cur = conn.execute("DELETE FROM rules WHERE id=?", (rule_id,))
        conn.commit()
        if cur.rowcount == 0:
            raise HTTPException(404, "规则不存在")
        return {"ok": True}
    finally:
        conn.close()


@app.post("/api/rules/{rule_id}/toggle")
def toggle_rule(rule_id: int):
    conn = db.get_conn()
    try:
        row = conn.execute("SELECT enabled FROM rules WHERE id=?", (rule_id,)).fetchone()
        if not row:
            raise HTTPException(404, "规则不存在")
        new_val = 0 if row["enabled"] else 1
        conn.execute("UPDATE rules SET enabled=? WHERE id=?", (new_val, rule_id))
        conn.commit()
        return {"id": rule_id, "enabled": new_val}
    finally:
        conn.close()


@app.get("/api/rule-audit")
def rule_audit(rule_id: Optional[int] = None):
    """对当前**启用**规则做静态覆盖审计（只读，绝不触碰 live/replay 数据）。

    按设备/批次/级别通配、每日时段（含跨午夜）与日期范围计算每对规则的
    可达条件交集，标出动作冲突、冷却差异、完全遮蔽与冗余重叠；结果按规则
    id 稳定排序。可选 ``rule_id`` 只保留涉及该规则的关系。
    """
    conn = db.get_conn()
    try:
        rows = conn.execute("SELECT * FROM rules ORDER BY priority, id").fetchall()
        all_rules = [dict(r) for r in rows]
        if rule_id is not None:
            if not any(r["id"] == rule_id for r in all_rules):
                raise HTTPException(404, "规则不存在")
            if not any(r["id"] == rule_id and r["enabled"] for r in all_rules):
                return {
                    "filter_rule_id": rule_id,
                    "filter_note": "该规则当前为停用状态，不参与评估，因此没有审计关系",
                    **audit_rules(all_rules),
                }
        result = audit_rules(all_rules)
        result["filter_rule_id"] = rule_id
        if rule_id is None:
            result["overlaps_filtered"] = len(result["overlaps"])
            return result
        filtered = [
            o for o in result["overlaps"] if rule_id in o["rule_ids"]
        ]
        result["overlaps_filtered"] = len(filtered)
        result["overlaps"] = filtered
        return result
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 规则审计修复工作台：确定性建议、影响预演（不落库）、单事务应用（失败自动回滚）
# ---------------------------------------------------------------------------

def _audit_and_suggestions(conn):
    """当前规则表 → 审计结果 + 确定性修复建议（只读计算）。"""
    rows = conn.execute("SELECT * FROM rules ORDER BY priority, id").fetchall()
    all_rules = [dict(r) for r in rows]
    audit_result = audit_rules(all_rules)
    sugg = generate_suggestions(audit_result, all_rules)
    return all_rules, audit_result, sugg


@app.get("/api/rule-audit/suggestions")
def rule_audit_suggestions():
    """基于当前启用规则的审计关系，生成确定性的合并/收窄/删除/停用建议（只读）。

    同一规则集在任何时刻生成的建议完全一致；冷却差异类关系不生成自动建议，
    在 ``unsupported`` 中单列。
    """
    conn = db.get_conn()
    try:
        _, audit_result, sugg = _audit_and_suggestions(conn)
        return {
            "enabled_rule_count": audit_result["enabled_rule_count"],
            "overlap_count": len(audit_result["overlaps"]),
            "pending_count": len(sugg["suggestions"]),
            "suggestions": sugg["suggestions"],
            "unsupported": sugg["unsupported"],
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 影响预演：草稿规则 + 从零回放 + 绑定令牌（全部只读计算，不落库、不转发）
# ---------------------------------------------------------------------------

_TOKEN_RULE_COLS = (
    "id", "name", "description", "action", "priority", "cooldown_seconds",
    "match_device", "match_batch", "match_level", "time_start", "time_end",
    "date_start", "date_end", "enabled",
)
_TOKEN_ALERT_COLS = (
    "id", "seq", "device_id", "batch_id", "level", "alert_type", "ts",
)


def _preview_token(rules_rows, ordered_ids: list[str], alert_rows) -> str:
    """预演令牌：绑定规则快照 + 建议集合（确定性应用顺序）+ 告警集合的内容哈希。

    纯内容寻址、无状态不落库：任一输入变化都会得到不同令牌；正式应用在
    同一事务内用当前数据重算并比对，不一致即以 409 拒绝。
    """
    payload = {
        "rules": [{k: r[k] for k in _TOKEN_RULE_COLS} for r in rules_rows],
        "suggestion_ids": list(ordered_ids),
        "alerts": [{k: a[k] for k in _TOKEN_ALERT_COLS} for a in alert_rows],
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"))
    return "pv1." + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _replay_in_memory(rules: list[RuleView], alert_rows) -> list[dict]:
    """用给定规则对全部已注入告警按注入顺序从零回放（全新冷却状态）。

    与 ``POST /api/replays`` 语义一致，但只在内存中求值：不落库、不转发。
    """
    state = EvalState()
    results = []
    for row in alert_rows:
        alert = _alert_dict(row)
        alert["ts"] = datetime.fromisoformat(alert["ts"])
        results.append(evaluate(alert, rules, state))
    return results


def _decision_brief(result: dict) -> dict:
    return {k: result[k] for k in ("decision", "rule_id", "rule_name", "reason")}


@app.post("/api/rule-audit/remediations/preview")
def preview_rule_remediation(payload: dict = Body(...)):
    """影响预演：不落库生成草稿规则，并分别用当前/草稿规则从零回放全部已注入告警。

    流程（全部只读计算）：按与正式应用完全相同的确定性顺序在内存中应用勾选
    建议得到草稿规则 → 用当前规则与草稿规则各自从**全新冷却状态**按注入顺序
    重放全部已注入告警 → 汇总放行→抑制、抑制→放行与逐条规则归因变化。
    返回绑定规则快照、建议集合与告警集合的 ``preview_token``，正式应用必须
    携带该令牌并在同一事务内复核。无已注入告警时返回零变化预演（仍可应用）。
    """
    selected_ids = (payload or {}).get("suggestion_ids")
    if not isinstance(selected_ids, list) or not selected_ids:
        raise HTTPException(400, "suggestion_ids 必须是非空数组")
    if not all(isinstance(s, str) for s in selected_ids):
        raise HTTPException(400, "suggestion_ids 必须全部是字符串")

    conn = db.get_conn()
    try:
        all_rules, before, sugg = _audit_and_suggestions(conn)
        by_id = {s["id"]: s for s in sugg["suggestions"]}
        unknown = [sid for sid in selected_ids if sid not in by_id]
        if unknown:
            raise HTTPException(
                409, "建议已失效（规则可能已变更），请刷新后重试: "
                + ", ".join(str(s) for s in unknown)
            )
        try:
            plan = plan_draft(all_rules, before, sugg["suggestions"], selected_ids)
        except ValueError as exc:
            raise HTTPException(409, f"草稿规则生成失败：{exc}")
        errors = validate_remediation(before, plan["audit"], plan["ordered"])
        diff = diff_audits(before, plan["audit"])

        rules_rows = conn.execute("SELECT * FROM rules ORDER BY id").fetchall()
        alert_rows = conn.execute("SELECT * FROM alerts ORDER BY seq").fetchall()
        ordered_ids = [s["id"] for s in plan["ordered"]]
        token = _preview_token(rules_rows, ordered_ids, alert_rows)

        current_views = ordered_rules([dict(r) for r in rules_rows])
        draft_views = ordered_rules(plan["rules"])
        before_results = _replay_in_memory(current_views, alert_rows)
        after_results = _replay_in_memory(draft_views, alert_rows)

        rows_out = []
        n_p2s = n_s2p = n_attr = 0
        for row, b, a in zip(alert_rows, before_results, after_results):
            transition = None
            if b["decision"] == "pass" and a["decision"] == "suppress":
                transition = "pass_to_suppress"
                n_p2s += 1
            elif b["decision"] == "suppress" and a["decision"] == "pass":
                transition = "suppress_to_pass"
                n_s2p += 1
            elif b["rule_id"] != a["rule_id"]:
                transition = "attribution"
                n_attr += 1
            rows_out.append({
                "alert": {
                    "id": row["id"], "seq": row["seq"],
                    "device_id": row["device_id"], "batch_id": row["batch_id"],
                    "level": row["level"], "alert_type": row["alert_type"],
                    "message": row["message"], "ts": row["ts"],
                },
                "before": _decision_brief(b),
                "after": _decision_brief(a),
                "changed": transition is not None,
                "transition": transition,
            })
        changed = n_p2s + n_s2p + n_attr
        return {
            "preview_token": token,
            "suggestion_ids": ordered_ids,
            "applicable": not errors,
            "validation_errors": errors,
            "draft": {
                "results": plan["results"],
                "applied_count": sum(1 for r in plan["results"] if r["status"] == "applied"),
                "skipped_count": sum(1 for r in plan["results"] if r["status"] == "skipped"),
                "rules": plan["rules"],
                "audit_diff": diff,
            },
            "impact": {
                "alert_count": len(alert_rows),
                "changed_count": changed,
                "pass_to_suppress": n_p2s,
                "suppress_to_pass": n_s2p,
                "attribution_changed": n_attr,
                "unchanged": len(alert_rows) - changed,
                "rows": rows_out,
            },
        }
    finally:
        conn.close()


def _apply_remediation_op(conn, op: dict) -> None:
    """在事务内执行单个修复操作；任何校验失败抛异常，由调用方回滚。"""
    kind = op.get("op")
    rule_id = int(op.get("rule_id"))
    if kind == "delete_rule":
        cur = conn.execute("DELETE FROM rules WHERE id=?", (rule_id,))
    elif kind == "disable_rule":
        cur = conn.execute("UPDATE rules SET enabled=0 WHERE id=?", (rule_id,))
    elif kind == "update_rule":
        row = conn.execute("SELECT * FROM rules WHERE id=?", (rule_id,)).fetchone()
        if not row:
            raise HTTPException(409, f"规则 #{rule_id} 已不存在，修复无法应用")
        payload = dict(row)
        payload.update(op.get("set") or {})
        try:
            data = normalize_rule(payload)  # 与规则 CRUD 相同的校验
        except ValueError as exc:
            raise HTTPException(409, f"修复操作校验失败（规则 #{rule_id}）：{exc}")
        sets = ", ".join(f"{k}=:{k}" for k in data)
        data["id"] = rule_id
        cur = conn.execute(f"UPDATE rules SET {sets} WHERE id=:id", data)
    else:
        raise HTTPException(400, f"未知修复操作: {kind}")
    if cur.rowcount == 0:
        raise HTTPException(409, f"规则 #{rule_id} 已不存在，修复无法应用")


@app.post("/api/rule-audit/remediations")
def apply_rule_remediation(payload: dict = Body(...)):
    """在同一事务中复核预演令牌并应用勾选的修复建议草稿。

    必须携带最近一次「影响预演」返回的 ``preview_token``。事务内流程：
    重新生成建议并按确定性顺序排列 → 用当前规则表与告警集合重算令牌并比对
    （规则快照 / 建议集合 / 告警集合任一已变化 → **409 拒绝且不修改任何
    规则**，需重新预演）→ 令牌一致才快照规则表、逐条应用（某条建议针对的
    关系若已被本次先前的修复消除，则标记为跳过）→ 重新审计并校验（勾选的
    关系必须全部消除、不得新增可能改变结果的关系、不得新增被完全遮蔽的
    规则）→ **任一校验失败自动回滚**；全部通过则同事务写入修复记录并提交，
    响应中给出应用前后审计关系变化。只改规则，不回溯改写任何 live/replay
    历史。
    """
    selected_ids = (payload or {}).get("suggestion_ids")
    if not isinstance(selected_ids, list) or not selected_ids:
        raise HTTPException(400, "suggestion_ids 必须是非空数组")
    if not all(isinstance(s, str) for s in selected_ids):
        raise HTTPException(400, "suggestion_ids 必须全部是字符串")
    preview_token = (payload or {}).get("preview_token")
    if not isinstance(preview_token, str) or not preview_token:
        raise HTTPException(400, "缺少 preview_token：请先执行「影响预演」再正式应用")
    note = str((payload or {}).get("note", "")).strip()[:200]

    conn = db.get_conn()
    try:
        conn.execute("BEGIN IMMEDIATE")
        try:
            all_rules, before, sugg = _audit_and_suggestions(conn)
            by_id = {s["id"]: s for s in sugg["suggestions"]}
            unknown = [sid for sid in selected_ids if sid not in by_id]
            if unknown:
                raise HTTPException(
                    409, "建议已失效（规则可能已变更），请重新预演: "
                    + ", ".join(str(s) for s in unknown)
                )
            try:
                plan = plan_draft(all_rules, before, sugg["suggestions"], selected_ids)
            except ValueError as exc:
                raise HTTPException(409, f"修复操作校验失败：{exc}")
            ordered = plan["ordered"]
            results = plan["results"]

            # 同一事务内复核预演令牌：任一输入在预演后变化即拒绝，规则不被修改
            rules_rows = conn.execute("SELECT * FROM rules ORDER BY id").fetchall()
            alert_rows = conn.execute("SELECT * FROM alerts ORDER BY seq").fetchall()
            expected = _preview_token(
                rules_rows, [s["id"] for s in ordered], alert_rows
            )
            if preview_token != expected:
                raise HTTPException(
                    409, "预演令牌已失效：规则、建议或告警集合在预演后已变化，"
                         "规则未被修改，请重新执行「影响预演」"
                )

            snapshot = json.dumps(
                [dict(r) for r in rules_rows], ensure_ascii=False,
            )
            for s, res in zip(ordered, results):
                if res["status"] != "applied":
                    continue
                for op in s["ops"]:
                    _apply_remediation_op(conn, op)
            rows = conn.execute(
                "SELECT * FROM rules ORDER BY priority, id"
            ).fetchall()
            after = audit_rules([dict(r) for r in rows])
            errors = validate_remediation(before, after, ordered)
            if errors:
                raise HTTPException(409, "修复校验失败，已自动回滚：" + "；".join(errors))
            diff = diff_audits(before, after)
            cur = conn.execute(
                """INSERT INTO remediation_runs(note, selected_json, results_json,
                   diff_json, before_audit, after_audit, rules_snapshot)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    note,
                    json.dumps([s["id"] for s in ordered], ensure_ascii=False),
                    json.dumps(results, ensure_ascii=False),
                    json.dumps(diff, ensure_ascii=False),
                    json.dumps(before, ensure_ascii=False),
                    json.dumps(after, ensure_ascii=False),
                    snapshot,
                ),
            )
            run_id = cur.lastrowid
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        return {
            "run_id": run_id,
            "note": note,
            "results": results,
            "applied_count": sum(1 for r in results if r["status"] == "applied"),
            "skipped_count": sum(1 for r in results if r["status"] == "skipped"),
            "diff": diff,
        }
    finally:
        conn.close()


def _remediation_run_dict(row, can_undo_id: Optional[int]) -> dict:
    return {
        "id": row["id"],
        "note": row["note"] or "",
        "created_at": row["created_at"],
        "selected": json.loads(row["selected_json"]),
        "results": json.loads(row["results_json"]),
        "diff": json.loads(row["diff_json"]),
        "undone": bool(row["undone"]),
        "undone_at": row["undone_at"],
        "can_undo": (not row["undone"]) and row["id"] == can_undo_id,
    }


@app.get("/api/rule-audit/remediations")
def list_rule_remediations():
    """修复历史（最新在前）；只有最近一次未撤销的修复可以撤销。"""
    conn = db.get_conn()
    try:
        rows = conn.execute(
            "SELECT * FROM remediation_runs ORDER BY id DESC"
        ).fetchall()
        latest = conn.execute(
            "SELECT id FROM remediation_runs WHERE undone=0 ORDER BY id DESC LIMIT 1"
        ).fetchone()
        can_undo_id = latest["id"] if latest else None
        return {
            "runs": [_remediation_run_dict(r, can_undo_id) for r in rows],
            "can_undo_id": can_undo_id,
        }
    finally:
        conn.close()


@app.post("/api/rule-audit/remediations/{run_id}/undo")
def undo_rule_remediation(run_id: int):
    """撤销最近一次修复：在同一事务中把规则表整体恢复到该次修复前的快照。"""
    conn = db.get_conn()
    try:
        row = conn.execute(
            "SELECT * FROM remediation_runs WHERE id=?", (run_id,)
        ).fetchone()
        if not row:
            raise HTTPException(404, "修复记录不存在")
        latest = conn.execute(
            "SELECT id FROM remediation_runs WHERE undone=0 ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if not latest or latest["id"] != run_id:
            raise HTTPException(409, "只能撤销最近一次未撤销的修复")
        snapshot = json.loads(row["rules_snapshot"])
        conn.execute("BEGIN")
        try:
            conn.execute("DELETE FROM rules")
            for r in snapshot:
                cols = list(r.keys())
                conn.execute(
                    f"INSERT INTO rules({', '.join(cols)}) "
                    f"VALUES ({', '.join(':' + c for c in cols)})",
                    r,
                )
            conn.execute(
                "UPDATE remediation_runs SET undone=1, undone_at=datetime('now') "
                "WHERE id=?",
                (run_id,),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        rows = conn.execute("SELECT * FROM rules ORDER BY priority, id").fetchall()
        restored = audit_rules([dict(r) for r in rows])
        return {
            "ok": True,
            "run_id": run_id,
            "restored_rules": len(snapshot),
            "audit_totals": restored["totals"],
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 待注入队列（staged）与导入
# ---------------------------------------------------------------------------

@app.get("/api/staged")
def list_staged():
    conn = db.get_conn()
    try:
        rows = conn.execute("SELECT * FROM staged ORDER BY ord, id").fetchall()
        return [{"id": r["id"], "ord": r["ord"], "payload": json.loads(r["payload"])}
                for r in rows]
    finally:
        conn.close()


@app.post("/api/alerts/import")
async def import_alerts(request: Request):
    """导入告警 JSON 到待注入队列。接受 [..] 数组或 {"alerts": [..], "append": true}。"""
    raw = await request.json()
    append = False
    if isinstance(raw, dict):
        items = raw.get("alerts")
        append = bool(raw.get("append", False))
    else:
        items = raw
    if not isinstance(items, list) or not items:
        raise HTTPException(400, "需要非空的告警数组（或 {\"alerts\": [...]} 对象）")

    conn = db.get_conn()
    inserted = 0
    try:
        if not append:
            conn.execute("DELETE FROM staged")
        next_ord = conn.execute("SELECT COALESCE(MAX(ord), -1) + 1 FROM staged").fetchone()[0]
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                raise HTTPException(400, f"第 {idx + 1} 条不是 JSON 对象")
            # 提前校验可解析性（不入库无效数据）
            try:
                normalize_alert(item, seq=0, fallback_ts=datetime.now())
            except ValueError as exc:
                raise HTTPException(400, f"第 {idx + 1} 条告警无效: {exc}")
            conn.execute(
                "INSERT INTO staged(ord, payload) VALUES (?, ?)",
                (next_ord + idx, json.dumps(item, ensure_ascii=False)),
            )
            inserted += 1
        conn.commit()
        return {"imported": inserted, "appended": append}
    finally:
        conn.close()


@app.delete("/api/staged/{staged_id}")
def delete_staged(staged_id: int):
    conn = db.get_conn()
    try:
        cur = conn.execute("DELETE FROM staged WHERE id=?", (staged_id,))
        conn.commit()
        if cur.rowcount == 0:
            raise HTTPException(404, "队列项不存在")
        return {"ok": True}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 注入 / 撤销 / 已注入列表
# ---------------------------------------------------------------------------

@app.post("/api/alerts/inject-next")
def inject_next():
    """从待注入队列头部取一条，按当前策略评估（逐条注入）。"""
    conn = db.get_conn()
    try:
        row = conn.execute("SELECT * FROM staged ORDER BY ord, id LIMIT 1").fetchone()
        if not row:
            raise HTTPException(404, "待注入队列为空")
        payload = json.loads(row["payload"])
        conn.execute("DELETE FROM staged WHERE id=?", (row["id"],))
        out = _inject_one(conn, payload)
        conn.commit()
        _reconcile_events(conn, _get_gap(conn))
        out["staged_remaining"] = conn.execute("SELECT COUNT(*) FROM staged").fetchone()[0]
        return out
    finally:
        conn.close()


@app.post("/api/alerts/inject-all")
def inject_all():
    """依次注入队列中的全部告警，返回每条决定（用于快速观察整条序列）。"""
    conn = db.get_conn()
    try:
        rows = conn.execute("SELECT * FROM staged ORDER BY ord, id").fetchall()
        if not rows:
            raise HTTPException(404, "待注入队列为空")
        results = []
        for row in rows:
            results.append(_inject_one(conn, json.loads(row["payload"])))
        conn.execute("DELETE FROM staged")
        conn.commit()
        _reconcile_events(conn, _get_gap(conn))
        return {"injected": len(results), "results": results}
    finally:
        conn.close()


@app.post("/api/alerts/inject")
async def inject_single(request: Request):
    """直接注入一条告警 JSON（不经过队列）。"""
    payload = await request.json()
    if not isinstance(payload, dict):
        raise HTTPException(400, "请求体必须是单个告警 JSON 对象")
    conn = db.get_conn()
    try:
        out = _inject_one(conn, payload)
        conn.commit()
        _reconcile_events(conn, _get_gap(conn))
        return out
    finally:
        conn.close()


@app.get("/api/alerts")
def list_alerts():
    """已注入告警 + live 链上的决定（按注入顺序）。"""
    conn = db.get_conn()
    try:
        run_id = _live_run_id(conn)
        alerts = conn.execute("SELECT * FROM alerts ORDER BY seq").fetchall()
        out = []
        for a in alerts:
            d = conn.execute(
                "SELECT * FROM decisions WHERE run_id=? AND alert_id=?",
                (run_id, a["id"]),
            ).fetchone()
            out.append({"alert": _alert_dict(a),
                        "decision": _decision_dict(d) if d else None})
        return out
    finally:
        conn.close()


@app.post("/api/alerts/undo")
def undo_last():
    """撤销最近一次注入：删除该告警及其全部决定/投递记录，并退回队列头部。"""
    conn = db.get_conn()
    try:
        last = conn.execute("SELECT * FROM alerts ORDER BY seq DESC LIMIT 1").fetchone()
        if not last:
            raise HTTPException(404, "没有可撤销的注入")
        alert = _alert_dict(last)
        conn.execute("BEGIN")
        try:
            head_ord = conn.execute("SELECT COALESCE(MIN(ord), 0) FROM staged").fetchone()[0]
            conn.execute(
                "INSERT INTO staged(ord, payload) VALUES (?, ?)",
                (head_ord - 1, last["raw_json"]),
            )
            # 级联删除各 run 中该告警的 decisions 与投递日志
            conn.execute("DELETE FROM receiver_logs WHERE alert_id=?", (last["id"],))
            conn.execute("DELETE FROM alerts WHERE id=?", (last["id"],))
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        _reconcile_events(conn, _get_gap(conn))
        return {"ok": True, "undone": alert,
                "remaining": conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 事件关联（与 live/replay 评估完全独立：只读写 events 表）
# ---------------------------------------------------------------------------

def _load_all_alerts(conn) -> list[dict]:
    rows = conn.execute("SELECT * FROM alerts ORDER BY seq").fetchall()
    alerts = []
    for row in rows:
        alert = _alert_dict(row)
        alert["ts"] = datetime.fromisoformat(alert["ts"])
        alerts.append(alert)
    return alerts


def _get_gap(conn) -> int:
    raw = conn.execute(
        "SELECT value FROM settings WHERE key='correlation_gap_seconds'"
    ).fetchone()
    try:
        return normalize_gap(raw["value"] if raw else DEFAULT_GAP_SECONDS)
    except ValueError:
        return DEFAULT_GAP_SECONDS


def _pick_legacy(
    candidates: list[Any], member_ids: set[int]
) -> Optional[Any]:
    """间隔修改导致事件拆分/合并时，确定性挑选确认状态的继承来源。

    规则（无随机性、与调用次数无关）：成员重叠数最多 → 旧事件独有成员最少
    （即更接近包含关系）→ 旧事件 id 最小。
    """
    def score(row) -> tuple[int, int, int]:
        old_ids = set(json.loads(row["member_ids"]))
        overlap = len(old_ids & member_ids)
        return (overlap, -(len(old_ids - member_ids)), -row["id"])

    if not candidates:
        return None
    best = max(candidates, key=score)
    return best if set(json.loads(best["member_ids"])) & member_ids else None


def _reconcile_events(conn, gap_seconds: int) -> dict:
    """按当前告警集合与间隔确定性重算事件，并与已持久化行对齐。

    - 内容签名不变的事件原样保留（含确认状态）；
    - 新出现的签名插入；间隔修改导致拆分/合并时，从旧事件确定性继承确认状态；
    - 消失的旧事件删除；
    - 仅操作 events 表，不触碰 alerts / decisions / runs / receiver_logs。
    """
    alerts = _load_all_alerts(conn)
    grouped = correlate_events(alerts, gap_seconds)
    new_sigs = {
        event_signature(gap_seconds, e["device_id"], e["alert_type"], e["member_ids"]): e
        for e in grouped
    }

    old_rows = conn.execute("SELECT * FROM events").fetchall()
    old_by_sig = {r["signature"]: r for r in old_rows}
    stale = [r for r in old_rows if r["signature"] not in new_sigs]
    used_legacy: set[int] = set()

    created = updated = 0
    conn.execute("BEGIN")
    try:
        for sig, event in new_sigs.items():
            member_ids = json.dumps(event["member_ids"])
            member_seqs = json.dumps(event["member_seqs"])
            first_ts = event["first_ts"].isoformat(timespec="seconds")
            last_ts = event["last_ts"].isoformat(timespec="seconds")
            existing = old_by_sig.get(sig)
            if existing is not None:
                conn.execute(
                    """UPDATE events SET device_id=?, alert_type=?, gap_seconds=?,
                       first_ts=?, last_ts=?, alert_count=?, max_level=?,
                       member_ids=?, member_seqs=?, updated_at=datetime('now')
                       WHERE id=?""",
                    (event["device_id"], event["alert_type"], gap_seconds,
                     first_ts, last_ts, event["alert_count"], event["max_level"],
                     member_ids, member_seqs, existing["id"]),
                )
                updated += 1
                continue

            ack, acked_at = 0, None
            candidates = [
                r for r in stale
                if r["id"] not in used_legacy
                and r["device_id"] == event["device_id"]
                and r["alert_type"] == event["alert_type"]
                and r["acknowledged"]
            ]
            legacy = _pick_legacy(candidates, set(event["member_ids"]))
            if legacy is not None:
                used_legacy.add(legacy["id"])
                ack, acked_at = 1, legacy["acked_at"]

            conn.execute(
                """INSERT INTO events(signature, device_id, alert_type, gap_seconds,
                   first_ts, last_ts, alert_count, max_level, member_ids, member_seqs,
                   acknowledged, acked_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (sig, event["device_id"], event["alert_type"], gap_seconds,
                 first_ts, last_ts, event["alert_count"], event["max_level"],
                 member_ids, member_seqs, ack, acked_at),
            )
            created += 1

        deleted = 0
        for row in stale:
            conn.execute("DELETE FROM events WHERE id=?", (row["id"],))
            deleted += 1
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    return {"gap_seconds": gap_seconds, "events": len(new_sigs),
            "created": created, "updated": updated, "deleted": deleted}


def _event_dict(conn, row, live_run_id: int) -> dict:
    member_ids = json.loads(row["member_ids"])
    members = []
    for mid in member_ids:
        a = conn.execute("SELECT * FROM alerts WHERE id=?", (mid,)).fetchone()
        if not a:
            continue  # 成员告警被撤销后，下次 reconcile 会重算；这里容错跳过
        d = conn.execute(
            "SELECT * FROM decisions WHERE run_id=? AND alert_id=?",
            (live_run_id, mid),
        ).fetchone()
        members.append({
            "alert": _alert_dict(a),
            "live_decision": _decision_dict(d)["decision"] if d else None,
        })
    return {
        "id": row["id"],
        "device_id": row["device_id"],
        "alert_type": row["alert_type"],
        "gap_seconds": row["gap_seconds"],
        "first_ts": row["first_ts"],
        "last_ts": row["last_ts"],
        "alert_count": row["alert_count"],
        "max_level": row["max_level"],
        "member_ids": member_ids,
        "member_seqs": json.loads(row["member_seqs"]),
        "acknowledged": bool(row["acknowledged"]),
        "acked_at": row["acked_at"],
        "note": row["note"] or "",
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "members": members,
    }


@app.get("/api/events")
def list_events(
    device: str = "",
    level: str = "",
    status: str = "",
    start: Optional[str] = None,
    end: Optional[str] = None,
):
    """事件列表（自动按当前告警集合与已保存间隔确定性重算），支持筛选。

    - device: 精确匹配 device_id；level: 事件最高级别精确匹配；
    - status: acknowledged / unacknowledged；
    - start/end: ISO8601，事件时间区间与筛选窗口有重叠即返回。
    """
    if level and level not in LEVELS:
        raise HTTPException(400, f"level 只能是 {', '.join(LEVELS)} 之一")
    if status and status not in ("ack", "acknowledged", "unack", "unacknowledged", "all"):
        raise HTTPException(400, "status 只能是 acknowledged 或 unacknowledged")
    try:
        start_dt = parse_filter_ts(start, "start")
        end_dt = parse_filter_ts(end, "end")
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    if start_dt and end_dt and start_dt > end_dt:
        raise HTTPException(400, "start 不能晚于 end")

    conn = db.get_conn()
    try:
        gap = _get_gap(conn)
        stats = _reconcile_events(conn, gap)
        live_id = _live_run_id(conn)
        rows = conn.execute("SELECT * FROM events ORDER BY last_ts DESC, id DESC").fetchall()
        events = []
        for row in rows:
            if device and row["device_id"] != device.strip():
                continue
            if level and row["max_level"] != level:
                continue
            want_ack = status in ("ack", "acknowledged")
            if status in ("ack", "acknowledged", "unack", "unacknowledged") \
                    and bool(row["acknowledged"]) != want_ack:
                continue
            first_dt = datetime.fromisoformat(row["first_ts"])
            last_dt = datetime.fromisoformat(row["last_ts"])
            if start_dt and last_dt < start_dt:
                continue
            if end_dt and first_dt > end_dt:
                continue
            events.append(_event_dict(conn, row, live_id))

        devices = [r["device_id"] for r in conn.execute(
            "SELECT DISTINCT device_id FROM events ORDER BY device_id"
        ).fetchall()]
        ack_count = conn.execute(
            "SELECT COUNT(*) FROM events WHERE acknowledged=1"
        ).fetchone()[0]
        return {
            "gap_seconds": gap,
            "events": events,
            "devices": devices,
            "total": len(events),
            "acknowledged_count": ack_count,
            "unacknowledged_count": len(rows) - ack_count,
            "recompute": stats,
        }
    finally:
        conn.close()


@app.post("/api/events/{event_id}/ack")
def acknowledge_event(event_id: int, payload: dict = Body(default={})):
    """确认（acknowledged=true）或取消确认（false）一个事件。"""
    ack = bool((payload or {}).get("acknowledged", True))
    conn = db.get_conn()
    try:
        row = conn.execute("SELECT * FROM events WHERE id=?", (event_id,)).fetchone()
        if not row:
            raise HTTPException(404, "事件不存在（可能因重算而合并/删除，刷新后重试）")
        if ack:
            conn.execute(
                "UPDATE events SET acknowledged=1, acked_at=datetime('now'), "
                "updated_at=datetime('now') WHERE id=?",
                (event_id,),
            )
        else:
            conn.execute(
                "UPDATE events SET acknowledged=0, acked_at=NULL, "
                "updated_at=datetime('now') WHERE id=?",
                (event_id,),
            )
        conn.commit()
        out = conn.execute("SELECT * FROM events WHERE id=?", (event_id,)).fetchone()
        return _event_dict(conn, out, _live_run_id(conn))
    finally:
        conn.close()


@app.get("/api/correlation/config")
def get_correlation_config():
    conn = db.get_conn()
    try:
        return {"gap_seconds": _get_gap(conn)}
    finally:
        conn.close()


@app.put("/api/correlation/config")
def update_correlation_config(payload: dict = Body(...)):
    """修改聚合间隔并立即确定性重算。仅影响事件关联结果，不触碰 live/replay 决定。"""
    try:
        gap = normalize_gap((payload or {}).get("gap_seconds"))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    conn = db.get_conn()
    try:
        conn.execute(
            "INSERT INTO settings(key, value) VALUES ('correlation_gap_seconds', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (str(gap),),
        )
        conn.commit()
        stats = _reconcile_events(conn, gap)
        return {"gap_seconds": gap, "recompute": stats}
    finally:
        conn.close()


@app.post("/api/events/recompute")
def recompute_events():
    """用当前间隔对全部已注入告警手动重算事件（幂等，结果确定）。"""
    conn = db.get_conn()
    try:
        gap = _get_gap(conn)
        return _reconcile_events(conn, gap)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 事件演进：不可变快照与血缘对比（只读写 event_snapshots* 表，不回写 events）
# ---------------------------------------------------------------------------

def _snapshot_item_dict(row) -> dict:
    return {
        "id": row["id"],
        "event_id": row["event_id"],
        "device_id": row["device_id"],
        "alert_type": row["alert_type"],
        "first_ts": row["first_ts"],
        "last_ts": row["last_ts"],
        "alert_count": row["alert_count"],
        "max_level": row["max_level"],
        "member_ids": json.loads(row["member_ids"]),
        "member_seqs": json.loads(row["member_seqs"]),
        "acknowledged": bool(row["acknowledged"]),
        "acked_at": row["acked_at"],
    }


def _snapshot_dict(conn, row, with_items: bool = False) -> dict:
    out = {
        "id": row["id"],
        "note": row["note"] or "",
        "gap_seconds": row["gap_seconds"],
        "event_count": row["event_count"],
        "created_at": row["created_at"],
        "acknowledged_count": conn.execute(
            "SELECT COUNT(*) FROM event_snapshot_items "
            "WHERE snapshot_id=? AND acknowledged=1",
            (row["id"],),
        ).fetchone()[0],
    }
    if with_items:
        rows = conn.execute(
            "SELECT * FROM event_snapshot_items WHERE snapshot_id=? ORDER BY ord, id",
            (row["id"],),
        ).fetchall()
        out["items"] = [_snapshot_item_dict(r) for r in rows]
    return out


@app.post("/api/event-snapshots")
def create_event_snapshot(payload: dict = Body(default={})):
    """把当前事件关联结果保存为不可变快照。

    快照整体复制 events 表当前内容（聚合间隔、成员告警序号、确认状态、生成
    时间）；之后的告警撤销、间隔调整与事件重算都不会改写已保存的快照。
    """
    note = str((payload or {}).get("note", "")).strip()[:200]
    conn = db.get_conn()
    try:
        gap = _get_gap(conn)
        _reconcile_events(conn, gap)  # 先把 events 表对齐到当前告警集合，再整体复制
        rows = conn.execute(
            "SELECT * FROM events ORDER BY first_ts, device_id, alert_type, id"
        ).fetchall()
        conn.execute("BEGIN")
        try:
            cur = conn.execute(
                "INSERT INTO event_snapshots(note, gap_seconds, event_count) "
                "VALUES (?, ?, ?)",
                (note, gap, len(rows)),
            )
            snapshot_id = cur.lastrowid
            for ord_, r in enumerate(rows):
                conn.execute(
                    """INSERT INTO event_snapshot_items(
                       snapshot_id, ord, event_id, device_id, alert_type,
                       first_ts, last_ts, alert_count, max_level,
                       member_ids, member_seqs, acknowledged, acked_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (snapshot_id, ord_, r["id"], r["device_id"], r["alert_type"],
                     r["first_ts"], r["last_ts"], r["alert_count"], r["max_level"],
                     r["member_ids"], r["member_seqs"], r["acknowledged"],
                     r["acked_at"]),
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        row = conn.execute(
            "SELECT * FROM event_snapshots WHERE id=?", (snapshot_id,)
        ).fetchone()
        return _snapshot_dict(conn, row, with_items=True)
    finally:
        conn.close()


@app.get("/api/event-snapshots")
def list_event_snapshots():
    """全部快照（按创建顺序），不含事件明细。"""
    conn = db.get_conn()
    try:
        rows = conn.execute("SELECT * FROM event_snapshots ORDER BY id").fetchall()
        return [_snapshot_dict(conn, r) for r in rows]
    finally:
        conn.close()


# 注意：/compare 必须注册在 /{snapshot_id} 之前，否则会被路径参数路由截获
@app.get("/api/event-snapshots/compare")
def compare_event_snapshots(from_id: int, to_id: int):
    """对比两个快照：按成员交集建立确定性血缘（新增/消失/拆分/合并/延续/确认变化）。"""
    conn = db.get_conn()
    try:
        old = conn.execute(
            "SELECT * FROM event_snapshots WHERE id=?", (from_id,)
        ).fetchone()
        new = conn.execute(
            "SELECT * FROM event_snapshots WHERE id=?", (to_id,)
        ).fetchone()
        if not old:
            raise HTTPException(404, f"快照 {from_id} 不存在")
        if not new:
            raise HTTPException(404, f"快照 {to_id} 不存在")

        def load_items(snapshot_id: int) -> list[dict]:
            rows = conn.execute(
                "SELECT * FROM event_snapshot_items WHERE snapshot_id=? "
                "ORDER BY ord, id",
                (snapshot_id,),
            ).fetchall()
            return [_snapshot_item_dict(r) for r in rows]

        relations = compare_snapshot_items(load_items(from_id), load_items(to_id))
        summary = {
            "relations": len(relations),
            "added": sum(1 for r in relations if r["kind"] == "added"),
            "removed": sum(1 for r in relations if r["kind"] == "removed"),
            "continued": sum(1 for r in relations if r["kind"] == "continued"),
            "ack_changed": sum(1 for r in relations if r["kind"] == "ack_changed"),
            "split": sum(1 for r in relations if r["split"]),
            "merge": sum(1 for r in relations if r["merge"]),
            "changed": sum(1 for r in relations if r["changed"]),
            "unchanged": sum(1 for r in relations if not r["changed"]),
        }
        return {
            "from": _snapshot_dict(conn, old),
            "to": _snapshot_dict(conn, new),
            "relations": relations,
            "summary": summary,
        }
    finally:
        conn.close()


@app.get("/api/event-snapshots/{snapshot_id}")
def get_event_snapshot(snapshot_id: int):
    """单个快照的完整内容（含每个事件的成员序号与确认状态）。"""
    conn = db.get_conn()
    try:
        row = conn.execute(
            "SELECT * FROM event_snapshots WHERE id=?", (snapshot_id,)
        ).fetchone()
        if not row:
            raise HTTPException(404, "快照不存在")
        return _snapshot_dict(conn, row, with_items=True)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 策略回放与前后对比
# ---------------------------------------------------------------------------

@app.post("/api/replays")
def create_replay(payload: dict = Body(default={})):
    """用当前启用的规则快照，按顺序重放全部已注入告警（全新冷却状态）。"""
    note = str((payload or {}).get("note", "")).strip() or "策略变更后手动回放"
    conn = db.get_conn()
    try:
        rules = _rule_views(conn)
        snapshot = [
            {k: getattr(r, k) for k in (
                "id", "name", "description", "action", "priority", "cooldown_seconds",
                "match_device", "match_batch", "match_level", "time_start", "time_end",
                "date_start", "date_end", "enabled")}
            for r in rules
        ]
        cur = conn.execute(
            "INSERT INTO runs(kind, note, rule_snapshot) VALUES ('replay', ?, ?)",
            (note[:200], json.dumps(snapshot, ensure_ascii=False)),
        )
        run_id = cur.lastrowid

        state = EvalState()
        alerts = conn.execute("SELECT * FROM alerts ORDER BY seq").fetchall()
        for position, row in enumerate(alerts, start=1):
            alert = _alert_dict(row)
            alert["ts"] = datetime.fromisoformat(alert["ts"])
            result = evaluate(alert, rules, state)
            _persist_decision(conn, run_id, row["id"], result)
        conn.commit()
        return {"id": run_id, "note": note, "replayed_alerts": len(alerts),
                "rules_in_snapshot": len(snapshot)}
    finally:
        conn.close()


@app.get("/api/runs")
def list_runs():
    conn = db.get_conn()
    try:
        rows = conn.execute(
            """SELECT r.*,
                      (SELECT COUNT(*) FROM decisions d WHERE d.run_id=r.id) AS decision_count,
                      (SELECT COUNT(*) FILTER (WHERE decision='pass') FROM decisions d
                       WHERE d.run_id=r.id) AS pass_count,
                      (SELECT COUNT(*) FILTER (WHERE decision='suppress') FROM decisions d
                       WHERE d.run_id=r.id) AS suppress_count
               FROM runs r ORDER BY r.id"""
        ).fetchall()
        out = []
        for r in rows:
            out.append({
                "id": r["id"], "kind": r["kind"], "note": r["note"],
                "created_at": r["created_at"], "decision_count": r["decision_count"],
                "pass_count": r["pass_count"], "suppress_count": r["suppress_count"],
                "rule_snapshot": json.loads(r["rule_snapshot"]),
            })
        return out
    finally:
        conn.close()


@app.get("/api/runs/{run_id}/decisions")
def run_decisions(run_id: int):
    conn = db.get_conn()
    try:
        run = conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if not run:
            raise HTTPException(404, "评估记录不存在")
        rows = conn.execute(
            """SELECT d.*, a.seq, a.device_id, a.batch_id, a.level, a.alert_type,
                      a.message, a.ts AS alert_ts
               FROM decisions d JOIN alerts a ON a.id=d.alert_id
               WHERE d.run_id=? ORDER BY d.position""",
            (run_id,),
        ).fetchall()
        decisions = []
        for r in rows:
            d = _decision_dict(r)
            d.update({
                "seq": r["seq"], "device_id": r["device_id"], "batch_id": r["batch_id"],
                "level": r["level"], "alert_type": r["alert_type"],
                "message": r["message"], "alert_ts": r["alert_ts"],
            })
            decisions.append(d)
        return {
            "run": {"id": run["id"], "kind": run["kind"], "note": run["note"],
                    "created_at": run["created_at"],
                    "rule_snapshot": json.loads(run["rule_snapshot"])},
            "decisions": decisions,
        }
    finally:
        conn.close()


@app.get("/api/compare")
def compare_runs(run_a: int, run_b: Optional[int] = None):
    """对比两次评估（默认 run_b=live 链）中每条告警的决定差异。"""
    conn = db.get_conn()
    try:
        live_id = _live_run_id(conn)
        run_b = run_b or live_id
        if not conn.execute("SELECT 1 FROM runs WHERE id=?", (run_a,)).fetchone():
            raise HTTPException(404, f"run {run_a} 不存在")
        if not conn.execute("SELECT 1 FROM runs WHERE id=?", (run_b,)).fetchone():
            raise HTTPException(404, f"run {run_b} 不存在")

        def load(rid: int) -> dict[int, dict]:
            rows = conn.execute(
                """SELECT d.*, a.seq FROM decisions d JOIN alerts a ON a.id=d.alert_id
                   WHERE d.run_id=?""", (rid,),
            ).fetchall()
            return {r["alert_id"]: _decision_dict(r) | {"seq": r["seq"]} for r in rows}

        da, db_ = load(run_a), load(run_b)
        alerts = conn.execute("SELECT * FROM alerts ORDER BY seq").fetchall()
        rows_out, changed = [], 0
        for a in alerts:
            x, y = da.get(a["id"]), db_.get(a["id"])
            is_changed = bool(x and y and x["decision"] != y["decision"])
            changed += is_changed
            rows_out.append({
                "alert": _alert_dict(a),
                "a": x, "b": y, "changed": is_changed,
            })

        def run_meta(rid: int) -> dict:
            r = conn.execute("SELECT * FROM runs WHERE id=?", (rid,)).fetchone()
            return {"id": r["id"], "kind": r["kind"], "note": r["note"],
                    "created_at": r["created_at"]}

        return {"run_a": run_meta(run_a), "run_b": run_meta(run_b),
                "rows": rows_out, "changed_count": changed,
                "total": len(rows_out)}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 模拟接收端 + 转发设置
# ---------------------------------------------------------------------------

@app.post("/mock-webhook")
async def mock_webhook(request: Request):
    """内置模拟接收端：任何 POST 都会落库展示，返回 200。可用 curl 验证连通性。"""
    try:
        body = await request.json()
    except Exception:
        body = {"raw": (await request.body()).decode("utf-8", "replace")}
    conn = db.get_conn()
    try:
        alert_id = None
        if isinstance(body, dict):
            seq = (body.get("alert") or {}).get("seq")
            if seq is not None:
                row = conn.execute("SELECT id FROM alerts WHERE seq=?", (seq,)).fetchone()
                alert_id = row["id"] if row else None
        cur = conn.execute(
            """INSERT INTO receiver_logs(alert_id, status, target, payload_json, detail)
               VALUES (?, 'received', ?, ?, '外部调用内置模拟接收端 /mock-webhook')""",
            (alert_id, f"{request.url.path}", json.dumps(body, ensure_ascii=False)),
        )
        conn.commit()
        return {"ok": True, "received": True, "log_id": cur.lastrowid}
    finally:
        conn.close()


@app.get("/api/receiver/logs")
def receiver_logs(limit: int = 100):
    conn = db.get_conn()
    try:
        rows = conn.execute(
            """SELECT l.*, a.seq AS alert_seq FROM receiver_logs l
               LEFT JOIN alerts a ON a.id=l.alert_id
               ORDER BY l.id DESC LIMIT ?""",
            (max(1, min(limit, 500)),),
        ).fetchall()
        return [{
            "id": r["id"], "created_at": r["created_at"], "alert_seq": r["alert_seq"],
            "status": r["status"], "target": r["target"],
            "payload": json.loads(r["payload_json"]), "detail": r["detail"],
        } for r in rows]
    finally:
        conn.close()


@app.post("/api/receiver/clear")
def clear_receiver_logs():
    conn = db.get_conn()
    try:
        cur = conn.execute("DELETE FROM receiver_logs")
        conn.commit()
        return {"ok": True, "cleared": cur.rowcount}
    finally:
        conn.close()


@app.get("/api/settings")
def get_settings():
    conn = db.get_conn()
    try:
        return _get_settings(conn)
    finally:
        conn.close()


@app.put("/api/settings")
async def update_settings(request: Request):
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(400, "请求体必须是对象")
    allowed = set(db.DEFAULT_SETTINGS)
    conn = db.get_conn()
    try:
        for key, value in body.items():
            if key not in allowed:
                raise HTTPException(400, f"未知设置项: {key}")
            if key == "external_url" and value:
                if not str(value).startswith(("http://", "https://")):
                    raise HTTPException(400, "external_url 必须以 http:// 或 https:// 开头")
            conn.execute(
                "INSERT INTO settings(key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, "true" if value is True else
                      "false" if value is False else str(value)),
            )
        conn.commit()
        return _get_settings(conn)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 重置沙盒
# ---------------------------------------------------------------------------

@app.post("/api/reset")
def reset_sandbox():
    """清空全部业务数据并恢复内置示例规则与告警队列。"""
    conn = db.get_conn()
    try:
        db.reset_all(conn)
        _reconcile_events(conn, _get_gap(conn))
        return {"ok": True}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 静态前端（放在最后，避免覆盖 API 路由）
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
