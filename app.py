from __future__ import annotations

import base64
import hashlib
import html
import hmac
import json
import math
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd
import requests
import streamlit as st


APP_TITLE = "Stellar"
DATA_FILE = Path("pulse_data.json")
SUPABASE_TABLE = "stellar_data"
SUPABASE_IDEAS_TABLE = "stellar_ideas"
SUPABASE_VOTES_TABLE = "stellar_votes"
SUPABASE_TASKS_TABLE = "stellar_tasks"
SUPABASE_HISTORY_TABLE = "stellar_status_history"
HERO_IMAGE = Path("assets/fuji-hero.webp")
NIGHT_HERO_IMAGE = Path("assets/fuji-night-stars.webp")
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_FALLBACK_MODELS = ["gemini-2.0-flash"]
LEGACY_GEMINI_MODELS = {"gemini-3.5-flash"}
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"
DEEPSEEK_MODELS = ["deepseek-v4-flash", "deepseek-v4-pro", "deepseek-chat", "deepseek-reasoner"]
GEMINI_MODELS = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-3-flash-preview"]
LANGUAGE_OPTIONS = {"中文": "zh", "日本語": "ja"}


UI_VALUE_JA = {
    "沟通协同": "コミュニケーション・連携",
    "权益激励": "待遇・インセンティブ",
    "流程规范": "プロセス整備",
    "文化活动": "文化・イベント",
    "成长发展": "成長・キャリア",
    "综合建议": "総合提案",
    "待确认": "確認待ち",
    "已受理": "受付済み",
    "推进中": "進行中",
    "已完成": "完了",
    "暂缓": "保留",
    "已合并": "統合済み",
    "高": "高",
    "中": "中",
    "低": "低",
    "匿名": "匿名",
    "系统": "システム",
    "系统迁移": "システム移行",
    "管理层": "管理層",
    "待定": "未定",
}

UI_SYSTEM_TEXT_JA = {
    "反馈已从旧数据迁移。": "旧データから声を移行しました。",
    "反馈已提交，等待确认。": "声を受け付けました。確認をお待ちください。",
    "反馈已提交，等待协同小组确认。": "声を受け付けました。連携チームの確認をお待ちください。",
    "AI 已协助整理表达，等待协同小组确认。": "AIが表現を整えました。連携チームの確認をお待ちください。",
}


def current_language() -> str:
    return str(st.session_state.get("language", "zh"))


def tx(zh: str, ja: str) -> str:
    return ja if current_language() == "ja" else zh


def ui_value(value: object) -> str:
    text = str(value)
    return UI_VALUE_JA.get(text, text) if current_language() == "ja" else text


def ui_system_text(value: object) -> str:
    text = str(value)
    return UI_SYSTEM_TEXT_JA.get(text, text) if current_language() == "ja" else text


def language_query_field() -> str:
    return f'<input type="hidden" name="lang" value="{current_language()}">'


def sync_language_from_widget(widget_key: str) -> None:
    selected = str(st.session_state.get(widget_key, "中文"))
    code = LANGUAGE_OPTIONS.get(selected, "zh")
    st.session_state["language"] = code
    st.query_params["lang"] = code


DEFAULT_DATA = {
    "ideas": [],
    "tasks": [],
    "events": [],
    "briefs": [],
    "council": {
        "cycle": "",
        "cadence": "",
        "members": [],
        "principles": [],
    },
}


HEAT_WEIGHTS = {
    "impact": 0.30,
    "urgency": 0.25,
    "resonance": 0.25,
    "duplication": 0.10,
    "actionability": 0.10,
}


def keyword_score(text: str, groups: list[tuple[set[str], int]], default: int) -> int:
    for keywords, score in groups:
        if any(keyword in text for keyword in keywords):
            return score
    return default


def extract_topic_tokens(text: str) -> set[str]:
    keywords = {
        "沟通", "信息", "同步", "流程", "负责人", "活动", "团建", "补贴", "奖金", "加班",
        "晋升", "培训", "反馈", "协作", "会议", "预算", "分工", "透明", "权益", "效率",
        "福利", "咖啡", "休息", "环境", "空间", "设备", "餐饮", "交通", "报销",
    }
    return {keyword for keyword in keywords if keyword in text}


def text_bigrams(text: str) -> set[str]:
    compact = re.sub(r"\s+", "", text)
    return {compact[index : index + 2] for index in range(max(0, len(compact) - 1))}


def idea_similarity(candidate: dict, existing: dict) -> float:
    candidate_text = f"{candidate.get('title', '')} {candidate.get('content', '')} {candidate.get('impact', '')}"
    existing_text = f"{existing.get('title', '')} {existing.get('content', '')} {existing.get('impact', '')}"
    candidate_tokens = extract_topic_tokens(candidate_text)
    existing_tokens = extract_topic_tokens(existing_text)
    token_union = candidate_tokens | existing_tokens
    token_score = len(candidate_tokens & existing_tokens) / len(token_union) if token_union else 0.0
    candidate_bigrams = text_bigrams(str(candidate.get("title", "")))
    existing_bigrams = text_bigrams(str(existing.get("title", "")))
    bigram_union = candidate_bigrams | existing_bigrams
    title_score = len(candidate_bigrams & existing_bigrams) / len(bigram_union) if bigram_union else 0.0
    category_score = 1.0 if candidate.get("category") == existing.get("category") else 0.0
    return round(category_score * 0.35 + token_score * 0.45 + title_score * 0.20, 3)


def find_similar_ideas(candidate: dict, ideas: list[dict], threshold: float = 0.42) -> list[tuple[dict, float]]:
    matches = []
    for idea in ideas:
        if idea.get("merged_into_id"):
            continue
        score = idea_similarity(candidate, idea)
        if score >= threshold:
            matches.append((idea, score))
    return sorted(matches, key=lambda item: (item[1], item[0].get("heat", 0)), reverse=True)[:3]


def calculate_heat_factors(idea: dict, all_ideas: list[dict] | None = None) -> dict:
    title = str(idea.get("title", ""))
    content = str(idea.get("content", ""))
    impact_text = str(idea.get("impact", ""))
    category = str(idea.get("category", ""))
    text = f"{title} {content} {impact_text} {category}"
    votes = max(0, int(idea.get("votes", 0) or 0))

    impact = keyword_score(
        text,
        [
            ({"全公司", "所有人", "大家", "整体", "公司"}, 92),
            ({"跨部门", "部门间", "多部门", "部门"}, 78),
            ({"小组", "团队", "项目"}, 62),
            ({"个人", "我自己"}, 38),
        ],
        55,
    )
    urgency = keyword_score(
        text,
        [
            ({"紧急", "马上", "立刻", "无法", "阻塞", "严重", "风险", "投诉", "安全"}, 90),
            ({"经常", "反复", "一直", "长期", "影响工作", "效率低"}, 76),
            ({"希望", "建议", "可以优化"}, 56),
        ],
        64 if category in {"沟通协同", "权益激励"} else 52,
    )
    resonance = round(min(100, math.log(votes + 1, 20) * 100))

    tokens = extract_topic_tokens(text)
    similar_count = 0
    for other in all_ideas or []:
        if other.get("id") == idea.get("id"):
            continue
        other_tokens = other.get("_tokens") or extract_topic_tokens(
            f"{other.get('title', '')} {other.get('content', '')} {other.get('impact', '')} {other.get('category', '')}"
        )
        if category and category == other.get("category") and (tokens & other_tokens):
            similar_count += 1
    duplication = min(100, similar_count * 25)

    actionability = 38
    if len(content) >= 30:
        actionability += 16
    if len(impact_text) >= 12:
        actionability += 20
    if any(keyword in impact_text for keyword in {"建议", "希望", "明确", "建立", "流程", "负责人", "周期", "机制", "预算", "反馈"}):
        actionability += 18
    if any(keyword in text for keyword in {"谁", "什么时候", "如何", "怎么", "下一步"}):
        actionability += 8
    actionability = min(100, actionability)

    return {
        "impact": impact,
        "urgency": urgency,
        "resonance": resonance,
        "duplication": duplication,
        "actionability": actionability,
    }


def recalculate_idea_heat(idea: dict, all_ideas: list[dict] | None = None) -> int:
    factors = calculate_heat_factors(idea, all_ideas)
    idea["heat_factors"] = factors
    heat = sum(factors[key] * weight for key, weight in HEAT_WEIGHTS.items())
    return max(0, min(100, round(heat)))


def normalize_idea(idea: dict) -> dict:
    normalized = dict(idea)
    normalized["id"] = str(normalized.get("id") or f"idea-{uuid4().hex[:8]}")
    normalized["title"] = str(normalized.get("title") or "未命名反馈")
    normalized["category"] = str(normalized.get("category") or "综合建议")
    normalized["author"] = str(normalized.get("author") or "匿名")
    normalized["anonymous"] = bool(normalized.get("anonymous", normalized["author"] == "匿名"))
    normalized["content"] = str(normalized.get("content") or "")
    normalized["impact"] = str(normalized.get("impact") or polish_text(normalized["content"]))
    normalized["status"] = str(normalized.get("status") or "待确认")
    normalized["base_heat"] = int(normalized.get("base_heat", normalized.get("heat", 64)) or 64)
    normalized["votes"] = max(0, int(normalized.get("votes", 0) or 0))
    normalized["voters"] = list(normalized.get("voters") or [])
    normalized["owner"] = str(normalized.get("owner") or "待确认")
    normalized["management_response"] = str(normalized.get("management_response") or "")
    normalized["next_update_at"] = str(normalized.get("next_update_at") or "")
    normalized["merged_into_id"] = str(normalized.get("merged_into_id") or "")
    normalized["delete_code_hash"] = str(normalized.get("delete_code_hash") or "")
    normalized["history"] = list(normalized.get("history") or [])
    normalized["heat"] = recalculate_idea_heat(normalized)
    normalized["created_at"] = str(normalized.get("created_at") or now_str())
    return normalized


def normalize_task(task: dict) -> dict:
    normalized = dict(task)
    normalized["id"] = str(normalized.get("id") or f"task-{uuid4().hex[:8]}")
    normalized["name"] = str(normalized.get("name") or "未命名事项")
    normalized["owner"] = str(normalized.get("owner") or "待确认")
    normalized["status"] = str(normalized.get("status") or "待确认")
    normalized["priority"] = str(normalized.get("priority") or "中")
    normalized["progress"] = max(0, min(100, int(normalized.get("progress") or 8)))
    normalized["due"] = str(normalized.get("due") or "待定")
    normalized["reward"] = str(normalized.get("reward") or "待定")
    normalized["members"] = list(normalized.get("members") or ["员工协同小组"])
    normalized["next_step"] = str(normalized.get("next_step") or "等待负责人确认")
    return normalized


def refresh_idea_heat_scores(ideas: list[dict]) -> list[dict]:
    for idea in ideas:
        text = f"{idea.get('title', '')} {idea.get('content', '')} {idea.get('impact', '')} {idea.get('category', '')}"
        idea["_tokens"] = extract_topic_tokens(text)
    for idea in ideas:
        idea["heat"] = recalculate_idea_heat(idea, ideas)
    for idea in ideas:
        idea.pop("_tokens", None)
    return ideas


def load_data() -> dict:
    remote_data = load_remote_data()
    if remote_data is not None:
        return remote_data
    if not DATA_FILE.exists():
        save_data(DEFAULT_DATA)
        return json.loads(json.dumps(DEFAULT_DATA))
    with DATA_FILE.open("r", encoding="utf-8") as f:
        data = normalize_data(json.load(f))
    if "council" not in data:
        data["council"] = json.loads(json.dumps(DEFAULT_DATA["council"]))
        save_data(data)
    return data


def save_data(data: dict) -> None:
    if save_remote_data(data):
        return
    with DATA_FILE.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_supabase_config() -> tuple[str, str]:
    url = get_secret_value("SUPABASE_URL").rstrip("/")
    key = get_secret_value("SUPABASE_SERVICE_ROLE_KEY") or get_secret_value("SUPABASE_ANON_KEY")
    return url, key


def supabase_headers(key: str) -> dict:
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }


def hash_delete_code(code: str, idea_id: str) -> str:
    salt = get_secret_value("DELETE_CODE_SALT") or get_secret_value("ADMIN_PASSWORD") or "stellar-local-salt"
    return hashlib.sha256(f"{salt}:{idea_id}:{code}".encode("utf-8")).hexdigest()


def verify_delete_code(idea: dict, entered: str) -> bool:
    stored_hash = str(idea.get("delete_code_hash") or "")
    if stored_hash:
        return hmac.compare_digest(stored_hash, hash_delete_code(entered, str(idea["id"])))
    return hmac.compare_digest(str(idea.get("delete_code") or ""), entered)


def supabase_get(url: str, key: str, table: str, params: dict | None = None) -> list[dict]:
    response = requests.get(
        f"{url}/rest/v1/{table}",
        headers=supabase_headers(key),
        params=params or {"select": "*"},
        timeout=12,
    )
    response.raise_for_status()
    return response.json()


def supabase_upsert(url: str, key: str, table: str, rows: list[dict] | dict) -> None:
    if not rows:
        return
    response = requests.post(
        f"{url}/rest/v1/{table}",
        headers={**supabase_headers(key), "Prefer": "resolution=merge-duplicates,return=minimal"},
        json=rows,
        timeout=12,
    )
    response.raise_for_status()


def normalized_supabase_available(url: str, key: str) -> bool:
    try:
        supabase_get(url, key, SUPABASE_IDEAS_TABLE, {"select": "id", "limit": "1"})
        return True
    except Exception:
        return False


def idea_to_supabase_row(idea: dict) -> dict:
    normalized = normalize_idea(idea)
    delete_hash = normalized.get("delete_code_hash", "")
    if not delete_hash and normalized.get("delete_code"):
        delete_hash = hash_delete_code(str(normalized["delete_code"]), normalized["id"])
    return {
        "id": normalized["id"],
        "title": normalized["title"],
        "category": normalized["category"],
        "author": normalized["author"],
        "anonymous": normalized["anonymous"],
        "content": normalized["content"],
        "impact": normalized["impact"],
        "status": normalized["status"],
        "owner": normalized["owner"],
        "management_response": normalized["management_response"],
        "next_update_at": normalized["next_update_at"],
        "merged_into_id": normalized["merged_into_id"] or None,
        "base_heat": normalized["base_heat"],
        "heat": normalized["heat"],
        "votes": normalized["votes"],
        "heat_factors": normalized.get("heat_factors", {}),
        "delete_code_hash": delete_hash,
        "created_at": normalized["created_at"],
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def task_to_supabase_row(task: dict) -> dict:
    normalized = normalize_task(task)
    return {
        "id": normalized["id"],
        "name": normalized["name"],
        "owner": normalized["owner"],
        "status": normalized["status"],
        "priority": normalized["priority"],
        "progress": normalized["progress"],
        "due": normalized["due"],
        "reward": normalized["reward"],
        "members": normalized["members"],
        "next_step": normalized["next_step"],
        "plan": normalized.get("plan", ""),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def load_legacy_remote_data(url: str, key: str) -> dict | None:
    try:
        rows = supabase_get(url, key, SUPABASE_TABLE, {"select": "data", "id": "eq.main"})
        return normalize_data(rows[0].get("data")) if rows else None
    except Exception:
        return None


def migrate_legacy_to_normalized(url: str, key: str, legacy: dict) -> None:
    normalized = normalize_data(legacy)
    supabase_upsert(url, key, SUPABASE_IDEAS_TABLE, [idea_to_supabase_row(idea) for idea in normalized["ideas"]])
    supabase_upsert(url, key, SUPABASE_TASKS_TABLE, [task_to_supabase_row(task) for task in normalized["tasks"]])
    vote_rows = []
    history_rows = []
    for idea in normalized["ideas"]:
        vote_rows.extend({"idea_id": idea["id"], "voter_token": token} for token in idea.get("voters", []))
        history = idea.get("history") or [
            {
                "from_status": "",
                "to_status": idea["status"],
                "response": "反馈已从旧数据迁移。",
                "owner": idea.get("owner", "待确认"),
                "actor": "系统迁移",
                "created_at": idea["created_at"],
            }
        ]
        history_rows.extend({"idea_id": idea["id"], **event} for event in history)
    supabase_upsert(url, key, SUPABASE_VOTES_TABLE, vote_rows)
    supabase_upsert(url, key, SUPABASE_HISTORY_TABLE, history_rows)


def load_normalized_remote_data(url: str, key: str) -> dict:
    idea_rows = supabase_get(url, key, SUPABASE_IDEAS_TABLE, {"select": "*", "order": "created_at.desc"})
    if not idea_rows:
        legacy = load_legacy_remote_data(url, key)
        if legacy and (legacy.get("ideas") or legacy.get("tasks")):
            migrate_legacy_to_normalized(url, key, legacy)
            idea_rows = supabase_get(url, key, SUPABASE_IDEAS_TABLE, {"select": "*", "order": "created_at.desc"})
    task_rows = supabase_get(url, key, SUPABASE_TASKS_TABLE, {"select": "*", "order": "created_at.desc"})
    history_rows = supabase_get(url, key, SUPABASE_HISTORY_TABLE, {"select": "*", "order": "created_at.asc"})
    token = get_session_token()
    own_votes = supabase_get(
        url,
        key,
        SUPABASE_VOTES_TABLE,
        {"select": "idea_id", "voter_token": f"eq.{token}"},
    )
    voted_ids = {row["idea_id"] for row in own_votes}
    history_by_idea: dict[str, list[dict]] = {}
    for row in history_rows:
        history_by_idea.setdefault(str(row["idea_id"]), []).append(row)
    ideas = []
    for row in idea_rows:
        idea = normalize_idea(row)
        idea["history"] = history_by_idea.get(idea["id"], [])
        idea["voters"] = [token] if idea["id"] in voted_ids else []
        ideas.append(idea)
    legacy = load_legacy_remote_data(url, key) or DEFAULT_DATA
    return normalize_data(
        {
            **legacy,
            "ideas": ideas,
            "tasks": [normalize_task(row) for row in task_rows],
        }
    )


def save_normalized_remote_data(url: str, key: str, data: dict) -> None:
    normalized = normalize_data(data)
    supabase_upsert(url, key, SUPABASE_IDEAS_TABLE, [idea_to_supabase_row(idea) for idea in normalized["ideas"]])
    supabase_upsert(url, key, SUPABASE_TASKS_TABLE, [task_to_supabase_row(task) for task in normalized["tasks"]])


def get_normalized_supabase_config() -> tuple[str, str] | None:
    url, key = get_supabase_config()
    if url and key and normalized_supabase_available(url, key):
        return url, key
    return None


def supabase_patch(url: str, key: str, table: str, filters: dict, payload: dict) -> None:
    response = requests.patch(
        f"{url}/rest/v1/{table}",
        headers={**supabase_headers(key), "Prefer": "return=minimal"},
        params=filters,
        json=payload,
        timeout=12,
    )
    response.raise_for_status()


def supabase_delete(url: str, key: str, table: str, filters: dict) -> None:
    response = requests.delete(
        f"{url}/rest/v1/{table}",
        headers=supabase_headers(key),
        params=filters,
        timeout=12,
    )
    response.raise_for_status()


def persist_new_idea(data: dict, idea: dict) -> None:
    config = get_normalized_supabase_config()
    if not config:
        save_data(data)
        return
    url, key = config
    supabase_upsert(url, key, SUPABASE_IDEAS_TABLE, idea_to_supabase_row(idea))
    initial_event = (idea.get("history") or [{}])[0]
    supabase_upsert(
        url,
        key,
        SUPABASE_HISTORY_TABLE,
        {
            "idea_id": idea["id"],
            "from_status": initial_event.get("from_status", ""),
            "to_status": initial_event.get("to_status", idea["status"]),
            "response": initial_event.get("response", "反馈已提交，等待确认。"),
            "owner": initial_event.get("owner", idea.get("owner", "待确认")),
            "actor": initial_event.get("actor", "系统"),
            "created_at": initial_event.get("created_at", idea["created_at"]),
        },
    )


def persist_vote(idea: dict, token: str, data: dict) -> bool:
    config = get_normalized_supabase_config()
    if not config:
        voters = set(idea.get("voters") or [])
        if token in voters:
            return False
        idea["votes"] = max(0, int(idea.get("votes", 0) or 0)) + 1
        voters.add(token)
        idea["voters"] = list(voters)
        refresh_idea_heat_scores(data["ideas"])
        save_data(data)
        return True
    url, key = config
    response = requests.post(
        f"{url}/rest/v1/rpc/stellar_cast_vote",
        headers=supabase_headers(key),
        json={"p_idea_id": idea["id"], "p_voter_token": token},
        timeout=12,
    )
    response.raise_for_status()
    return bool(response.json())


def persist_delete_idea(data: dict, idea_id: str) -> None:
    config = get_normalized_supabase_config()
    if config:
        url, key = config
        supabase_delete(url, key, SUPABASE_IDEAS_TABLE, {"id": f"eq.{idea_id}"})
    else:
        data["ideas"] = [idea for idea in data["ideas"] if idea["id"] != idea_id]
        save_data(data)


def persist_management_update(idea: dict, event: dict, data: dict) -> None:
    config = get_normalized_supabase_config()
    if config:
        url, key = config
        supabase_patch(
            url,
            key,
            SUPABASE_IDEAS_TABLE,
            {"id": f"eq.{idea['id']}"},
            {
                "status": idea["status"],
                "owner": idea["owner"],
                "management_response": idea["management_response"],
                "next_update_at": idea["next_update_at"],
                "merged_into_id": idea.get("merged_into_id") or None,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        supabase_upsert(url, key, SUPABASE_HISTORY_TABLE, {"idea_id": idea["id"], **event})
    else:
        idea.setdefault("history", []).append(event)
        save_data(data)


def normalize_data(data: dict | None) -> dict:
    normalized = json.loads(json.dumps(DEFAULT_DATA))
    if isinstance(data, dict):
        for key, value in data.items():
            normalized[key] = value
    normalized["ideas"] = refresh_idea_heat_scores([normalize_idea(idea) for idea in normalized.get("ideas", [])])
    normalized["tasks"] = [normalize_task(task) for task in normalized.get("tasks", [])]
    if "council" not in normalized or not isinstance(normalized["council"], dict):
        normalized["council"] = json.loads(json.dumps(DEFAULT_DATA["council"]))
    return normalized


def load_remote_data() -> dict | None:
    url, key = get_supabase_config()
    if not url or not key:
        return None
    try:
        if normalized_supabase_available(url, key):
            return load_normalized_remote_data(url, key)
        legacy = load_legacy_remote_data(url, key)
        if legacy is not None:
            return legacy
        data = normalize_data(DEFAULT_DATA)
        save_remote_data(data)
        return data
    except Exception as exc:
        st.warning(f"远程数据读取失败，已使用本地数据：{exc}")
        return None


def save_remote_data(data: dict) -> bool:
    url, key = get_supabase_config()
    if not url or not key:
        return False
    try:
        if normalized_supabase_available(url, key):
            save_normalized_remote_data(url, key, data)
            return True
        response = requests.post(
            f"{url}/rest/v1/{SUPABASE_TABLE}",
            headers={**supabase_headers(key), "Prefer": "resolution=merge-duplicates,return=minimal"},
            json={"id": "main", "data": normalize_data(data), "updated_at": datetime.now(timezone.utc).isoformat()},
            timeout=12,
        )
        response.raise_for_status()
        return True
    except Exception as exc:
        st.warning(f"远程数据保存失败，已保存到本地：{exc}")
        return False


@st.cache_data(show_spinner=False)
def image_data_uri(path: Path) -> str:
    if not path.exists():
        return ""
    suffix = path.suffix.lower().replace(".", "")
    mime = "jpeg" if suffix in {"jpg", "jpeg"} else suffix or "png"
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:image/{mime};base64,{encoded}"


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def classify_text(text: str) -> tuple[str, str, int]:
    rules = [
        ("活动", "文化活动", "中"),
        ("团建", "文化活动", "中"),
        ("加班", "权益激励", "高"),
        ("补贴", "权益激励", "高"),
        ("奖金", "权益激励", "高"),
        ("沟通", "沟通协同", "高"),
        ("信息", "沟通协同", "高"),
        ("流程", "流程规范", "中"),
        ("负责人", "流程规范", "中"),
        ("晋升", "成长发展", "中"),
        ("培训", "成长发展", "中"),
    ]
    for keyword, category, priority in rules:
        if keyword in text:
            heat = 88 if priority == "高" else 72
            return category, priority, heat
    return "综合建议", "中", 64


def polish_text(text: str) -> str:
    compact = " ".join(text.strip().split())
    if not compact:
        return "建议补充更具体的背景和预期结果。"
    return (
        "建议将该问题作为可跟踪事项处理：先明确影响范围与负责人，"
        f'再围绕“{compact[:42]}”形成执行动作，并在固定周期内反馈处理进展。'
    )


def get_secret_value(name: str) -> str:
    value = os.getenv(name, "")
    if value:
        return value
    try:
        return str(st.secrets.get(name, "") or "")
    except Exception:
        return ""


def get_gemini_config() -> tuple[str, str]:
    api_key = get_secret_value("GEMINI_API_KEY") or get_secret_value("GOOGLE_API_KEY")
    model = get_secret_value("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
    if model in LEGACY_GEMINI_MODELS:
        model = DEFAULT_GEMINI_MODEL
    return api_key, model


def get_deepseek_config() -> tuple[str, str]:
    api_key = get_secret_value("DEEPSEEK_API_KEY")
    model = get_secret_value("DEEPSEEK_MODEL") or DEFAULT_DEEPSEEK_MODEL
    return api_key, model


def extract_json_object(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def call_gemini_translation_once(text: str, target: str, model: str, api_key: str) -> dict:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=25000))
    try:
        response = client.models.generate_content(model=model, contents=build_translation_prompt(text, target))
        output_text = getattr(response, "text", "") or str(response)
    finally:
        client.close()

    return normalize_ai_result(output_text, text, f"Gemini API · {model}")


def call_gemini_translation(text: str, target: str, model: str, api_key: str) -> dict:
    tried: list[str] = []
    errors: list[str] = []
    for candidate in [model, *GEMINI_FALLBACK_MODELS]:
        if candidate in tried:
            continue
        tried.append(candidate)
        try:
            result = call_gemini_translation_once(text, target, candidate, api_key)
            result["source"] = f"Gemini API · {candidate}"
            return result
        except Exception as exc:
            errors.append(f"{candidate}: {type(exc).__name__} {str(exc)[:220]}")
            if "RESOURCE_EXHAUSTED" in str(exc):
                break
    if errors:
        raise RuntimeError("；".join(errors))
    raise RuntimeError("Gemini 调用失败：没有可用模型")


def build_translation_prompt(text: str, target: str) -> str:
    output_language = tx("简体中文", "日本語")
    return f"""
把员工反馈转成公司内部可执行建议。只输出 JSON，不要 Markdown。
除 category 和 priority 的枚举值外，其他文本字段使用{output_language}。
字段：
category 从 ["沟通协同","权益激励","流程规范","文化活动","成长发展","综合建议"] 选一项；
priority 从 ["高","中","低"] 选一项；
heat 为 0-100 整数；
tone 为表达风格；
title 28 字内；
translated 120 字内，保留问题本质，弱化攻击性，不编造事实；
next_step 60 字内。
对象：{target}
反馈：{text[:500]}
""".strip()


def normalize_ai_result(output_text: str, text: str, source: str) -> dict:
    parsed = extract_json_object(output_text)
    category = parsed.get("category") or "综合建议"
    priority = parsed.get("priority") or "中"
    heat = int(parsed.get("heat") or 64)
    return {
        "category": category,
        "priority": priority,
        "heat": max(0, min(100, heat)),
        "tone": parsed.get("tone") or tx("正式、克制、聚焦行动", "丁寧で節度があり、行動に焦点を当てる"),
        "title": parsed.get("title") or tx(f"{category}优化建议", f"{ui_value(category)}の改善提案"),
        "translated": parsed.get("translated") or polish_text(text),
        "next_step": parsed.get("next_step") or tx("先整理共性样本，再确认负责人和反馈周期。", "まず共通する事例を整理し、担当者と回答サイクルを決めます。"),
        "source": source,
    }


def call_deepseek_translation(text: str, target: str, model: str, api_key: str) -> dict:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你是公司内部反馈整理助手，只输出合法 JSON。"},
            {"role": "user", "content": build_translation_prompt(text, target)},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.2,
        "max_tokens": 420,
    }
    response = requests.post(
        "https://api.deepseek.com/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    output_text = data["choices"][0]["message"]["content"]
    return normalize_ai_result(output_text, text, f"DeepSeek API · {model}")


def call_ai_translation(provider: str, text: str, target: str, model: str) -> dict:
    if provider == "DeepSeek":
        api_key, default_model = get_deepseek_config()
        if not api_key:
            raise RuntimeError("未配置 DEEPSEEK_API_KEY")
        return call_deepseek_translation(text, target, model or default_model, api_key)
    if provider == "Gemini":
        api_key, default_model = get_gemini_config()
        if not api_key:
            raise RuntimeError("未配置 GEMINI_API_KEY")
        return call_gemini_translation(text, target, model or default_model, api_key)
    return translate_emotion(text, target)


def translate_emotion(text: str, target: str) -> dict:
    compact = " ".join(text.strip().split())
    category, priority, heat = classify_text(compact)
    topic_hint = compact[:48] or "该反馈"
    tone_map = {
        "给管理层": "正式、克制、聚焦组织效率",
        "给员工协同小组": "真实、具体、便于整理和追踪",
        "给协同负责人": "真实、具体、便于整理和追踪",
        "给活动负责人": "协作式、重视资源和分工",
    }
    action_map = {
        "沟通协同": "建立统一同步入口，明确事项负责人、更新时间和反馈节点。",
        "权益激励": "确认额外公共事务的补贴、调休或贡献记录方式。",
        "流程规范": "梳理标准流程，把发起、审批、执行、复盘拆成明确步骤。",
        "文化活动": "将活动改为投票、认领、预算确认和复盘反馈的共创流程。",
        "成长发展": "明确培训、晋升或成长反馈机制，并设置固定沟通周期。",
    }
    core_action = action_map.get(category, "先收集更多样本，再形成可执行事项和反馈节奏。")
    if current_language() == "ja":
        action_map_ja = {
            "沟通协同": "情報共有の窓口を統一し、担当者、更新日、回答の節目を明確にします。",
            "权益激励": "通常業務外の公共的な仕事に対する手当、代休、貢献記録を明確にします。",
            "流程规范": "発起、承認、実行、振り返りを明確な手順に整理します。",
            "文化活动": "投票、担当希望、予算確認、振り返りを含む共創プロセスにします。",
            "成长发展": "研修、昇進、成長フィードバックの仕組みと定期的な対話を設けます。",
        }
        core_action = action_map_ja.get(category, "まず事例を集め、実行可能な案件と回答のリズムを整えます。")
        return {
            "category": category,
            "priority": priority,
            "heat": heat,
            "tone": "丁寧で具体的、行動に焦点を当てた表現",
            "translated": f"「{topic_hint}」に関する声から、{ui_value(category)}に改善の余地があることが分かります。{core_action}対応中の進捗と結果も共有し、課題が口頭のまま残らないようにします。",
            "title": f"{ui_value(category)}の改善提案：{topic_hint}",
            "next_step": core_action,
            "source": "ローカルルール",
        }
    translated = (
        f'当前围绕"{topic_hint}"的反馈，反映出公司在{category}方面存在可优化空间。'
        f"建议将其作为{priority}优先级事项处理：{core_action}"
        "同时建议在处理过程中同步进展和结果，避免问题长期停留在口头沟通层面。"
    )
    return {
        "category": category,
        "priority": priority,
        "heat": heat,
        "tone": tone_map.get(target, "正式、克制、聚焦行动"),
        "translated": translated,
        "title": f"{category}优化建议：{topic_hint}",
        "next_step": core_action,
        "source": "本地规则",
    }


def status_color(status: str) -> str:
    return {
        "待确认": "#f7c948",
        "已受理": "#5eead4",
        "推进中": "#7c8cff",
        "已完成": "#60d394",
        "暂缓": "#ff6b6b",
        "已合并": "#cbd5e1",
    }.get(status, "#94a3b8")


def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@900&display=swap');

        :root {
            --bg: #080b16;
            --panel: rgba(18, 25, 47, 0.78);
            --panel-2: rgba(255, 255, 255, 0.07);
            --text: #f7fbff;
            --muted: #9aa7bd;
            --line: rgba(255, 255, 255, 0.12);
            --cyan: #5eead4;
            --pink: #ff5ea8;
            --amber: #f7c948;
            --violet: #7c8cff;
        }

        .stApp {
            background:
                radial-gradient(circle at 18% 12%, rgba(94, 234, 212, 0.16), transparent 25%),
                radial-gradient(circle at 82% 8%, rgba(255, 94, 168, 0.15), transparent 24%),
                linear-gradient(135deg, #070913 0%, #101827 46%, #11161f 100%);
            color: var(--text);
        }

        [data-testid="stSidebar"] {
            background: rgba(8, 11, 22, 0.88);
            border-right: 1px solid var(--line);
        }

        [data-testid="stHeader"] {
            background: rgba(8, 11, 22, 0);
        }

        .block-container {
            padding-top: 2.2rem;
            max-width: 1280px;
            animation: stellarPageIn 0.45s ease-out both;
        }

        /* opacity-only: transform on this ancestor would break the
           position:fixed full-page shells (landing/star/echo/postcard). */
        @keyframes stellarPageIn {
            from { opacity: 0; }
            to   { opacity: 1; }
        }

        @media (prefers-reduced-motion: reduce) {
            .block-container { animation: none; }
        }

        /* Slide-up on the page-level containers. transform here is safe:
           on the fixed shells it applies to the element itself (not an
           ancestor), so it doesn't disturb their viewport positioning. */
        .landing-shell, .star-shell, .echo-shell, .pc-shell, .hero {
            animation: stellarRise 0.55s cubic-bezier(0.22, 1, 0.36, 1) both;
        }

        @keyframes stellarRise {
            from { transform: translateY(18px); }
            to   { transform: translateY(0); }
        }

        @media (prefers-reduced-motion: reduce) {
            .landing-shell, .star-shell, .echo-shell, .pc-shell, .hero { animation: none; }
        }

        h1, h2, h3, p, label, span, div {
            letter-spacing: 0 !important;
        }

        .hero {
            position: relative;
            overflow: hidden;
            padding: 28px;
            border: 1px solid rgba(255, 255, 255, 0.14);
            background:
                linear-gradient(135deg, rgba(94, 234, 212, 0.13), rgba(124, 140, 255, 0.11)),
                rgba(11, 16, 31, 0.8);
            box-shadow: 0 24px 80px rgba(0, 0, 0, 0.28);
            border-radius: 18px;
        }

        .hero:before {
            content: "";
            position: absolute;
            inset: -2px;
            background-image:
                linear-gradient(rgba(255,255,255,0.05) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255,255,255,0.05) 1px, transparent 1px);
            background-size: 34px 34px;
            mask-image: linear-gradient(90deg, black, transparent 72%);
            pointer-events: none;
        }

        .hero h1 {
            position: relative;
            font-size: 48px;
            line-height: 1.02;
            margin: 0 0 14px;
        }

        .hero p {
            position: relative;
            max-width: 820px;
            color: #cbd5e1;
            font-size: 17px;
            margin: 0;
        }

        .pulse-dot {
            display: inline-flex;
            width: 10px;
            height: 10px;
            border-radius: 999px;
            margin-right: 8px;
            background: var(--cyan);
            box-shadow: 0 0 0 0 rgba(94, 234, 212, 0.7);
            animation: pulse 1.8s infinite;
        }

        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(94, 234, 212, 0.6); }
            72% { box-shadow: 0 0 0 15px rgba(94, 234, 212, 0); }
            100% { box-shadow: 0 0 0 0 rgba(94, 234, 212, 0); }
        }

        .metric-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 14px;
            margin: 18px 0 6px;
        }

        .metric-card, .glass-card, .idea-card, .task-card {
            border: 1px solid var(--line);
            background: var(--panel);
            border-radius: 16px;
            box-shadow: 0 18px 52px rgba(0, 0, 0, 0.22);
        }

        .metric-card {
            padding: 18px;
        }

        .metric-card .label {
            color: var(--muted);
            font-size: 13px;
        }

        .metric-card .value {
            font-size: 30px;
            font-weight: 800;
            margin-top: 6px;
        }

        .metric-card .hint {
            color: #b9c6d8;
            font-size: 12px;
            margin-top: 4px;
        }

        .section-title {
            display: flex;
            align-items: center;
            gap: 10px;
            margin: 22px 0 12px;
            font-size: 22px;
            font-weight: 800;
        }

        .glass-card {
            padding: 20px;
            margin-bottom: 14px;
        }

        .idea-card {
            padding: 18px;
            margin-bottom: 12px;
        }

        .idea-head {
            display: flex;
            justify-content: space-between;
            gap: 12px;
            align-items: flex-start;
        }

        .idea-title {
            font-size: 18px;
            font-weight: 800;
            margin-bottom: 7px;
        }

        .tag {
            display: inline-flex;
            align-items: center;
            min-height: 25px;
            padding: 3px 9px;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.08);
            color: #e2e8f0;
            font-size: 12px;
            border: 1px solid rgba(255, 255, 255, 0.12);
            margin-right: 6px;
            margin-bottom: 6px;
        }

        .heat {
            min-width: 72px;
            text-align: center;
            border-radius: 14px;
            padding: 8px 10px;
            background: linear-gradient(135deg, rgba(255, 94, 168, 0.25), rgba(247, 201, 72, 0.22));
            border: 1px solid rgba(255, 255, 255, 0.12);
            font-weight: 800;
        }

        .heat-breakdown {
            display: flex;
            flex-wrap: wrap;
            gap: 4px;
            margin-top: 10px;
        }

        .heat-breakdown .tag {
            margin-right: 0;
            background: rgba(255, 255, 255, 0.055);
            color: #cbd5e1;
        }

        .management-response {
            margin-top: 12px;
            padding: 12px 14px;
            border-left: 3px solid var(--cyan);
            border-radius: 0 8px 8px 0;
            background: rgba(94, 234, 212, 0.08);
            color: #dffdfa;
            line-height: 1.65;
            font-size: 13px;
        }

        .status-timeline {
            position: relative;
            margin: -2px 0 18px 20px;
            padding-left: 20px;
            border-left: 1px solid rgba(94, 234, 212, 0.26);
        }

        .timeline-entry {
            position: relative;
            padding: 0 0 14px;
        }

        .timeline-entry:last-child {
            padding-bottom: 0;
        }

        .timeline-entry::before {
            content: "";
            position: absolute;
            left: -25px;
            top: 5px;
            width: 9px;
            height: 9px;
            border-radius: 999px;
            background: var(--cyan);
            box-shadow: 0 0 12px rgba(94, 234, 212, 0.7);
        }

        .timeline-entry strong {
            color: #f7fbff;
            font-size: 13px;
        }

        .timeline-entry span {
            display: block;
            color: var(--muted);
            font-size: 12px;
            margin-top: 3px;
            line-height: 1.55;
        }

        .idea-actions {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 14px;
        }

        .idea-actions form {
            margin: 0;
        }

        .idea-action-button {
            appearance: none;
            min-height: 36px;
            padding: 0 14px;
            border-radius: 999px;
            border: 1px solid rgba(94, 234, 212, 0.34);
            background: rgba(94, 234, 212, 0.10);
            color: #eafffb;
            font: inherit;
            font-size: 13px;
            font-weight: 800;
            cursor: pointer;
            backdrop-filter: blur(10px);
        }

        .idea-action-button:hover {
            border-color: rgba(94, 234, 212, 0.72);
            background: rgba(94, 234, 212, 0.18);
        }

        .idea-action-button.is-liked {
            border-color: rgba(247, 201, 72, 0.52);
            background: rgba(247, 201, 72, 0.16);
            color: #fff4c2;
        }

        .idea-delete-button {
            appearance: none;
            min-height: 36px;
            padding: 0 14px;
            border-radius: 999px;
            border: 1px solid rgba(255, 107, 107, 0.42);
            background: rgba(255, 107, 107, 0.10);
            color: #ffd5d5;
            font: inherit;
            font-size: 13px;
            font-weight: 800;
            cursor: pointer;
            backdrop-filter: blur(10px);
        }

        .idea-delete-button:hover {
            border-color: rgba(255, 107, 107, 0.74);
            background: rgba(255, 107, 107, 0.18);
            color: #fff;
        }

        .muted {
            color: var(--muted);
        }

        .task-card {
            padding: 16px;
            margin-bottom: 12px;
        }

        .progress-shell {
            height: 10px;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.08);
            overflow: hidden;
            margin: 12px 0 10px;
        }

        .progress-bar {
            height: 10px;
            border-radius: 999px;
            background: linear-gradient(90deg, var(--cyan), var(--violet), var(--pink));
        }

        .event-grid {
            display: grid;
            grid-template-columns: 1.05fr 0.95fr;
            gap: 16px;
        }

        .slot {
            display: flex;
            justify-content: space-between;
            gap: 12px;
            padding: 12px;
            border: 1px solid var(--line);
            background: rgba(255, 255, 255, 0.06);
            border-radius: 12px;
            margin-bottom: 8px;
        }

        .slot-done {
            color: var(--cyan);
            font-weight: 800;
        }

        .slot-open {
            color: var(--amber);
            font-weight: 800;
        }

        .stButton > button, .stDownloadButton > button {
            border-radius: 12px;
            border: 1px solid rgba(94, 234, 212, 0.35);
            background: linear-gradient(135deg, rgba(94, 234, 212, 0.18), rgba(124, 140, 255, 0.20));
            color: #f8fafc;
            font-weight: 750;
        }

        .stButton > button:hover, .stDownloadButton > button:hover {
            border-color: rgba(255, 94, 168, 0.6);
            color: white;
        }

        div[data-baseweb="tab-list"] {
            gap: 8px;
        }

        button[data-baseweb="tab"] {
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.06);
            padding: 8px 14px;
        }

        .landing-shell {
            position: fixed;
            inset: 0;
            width: 100vw;
            height: 100dvh;
            margin: 0;
            overflow: hidden;
        }

        .landing-hero {
            position: relative;
            width: 100%;
            height: 100%;
            min-height: 100dvh;
            overflow: hidden;
            background-size: cover;
            background-position: center;
        }

        .landing-hero:before {
            content: "";
            position: absolute;
            inset: 0;
            background:
                linear-gradient(90deg, rgba(5, 8, 18, 0.76) 0%, rgba(5, 8, 18, 0.38) 38%, rgba(5, 8, 18, 0.04) 100%),
                linear-gradient(0deg, rgba(5, 8, 18, 0.58) 0%, rgba(5, 8, 18, 0.04) 60%);
        }

        .landing-hero:after {
            content: none;
            pointer-events: none;
        }

        .weather-atmosphere {
            position: absolute;
            inset: 0;
            z-index: 0;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.5s ease;
        }

        .weather-fog .weather-atmosphere {
            opacity: 0.78;
            background:
                linear-gradient(100deg, transparent 10%, rgba(210, 222, 232, 0.28) 42%, transparent 72%),
                linear-gradient(80deg, transparent 26%, rgba(255, 255, 255, 0.18) 58%, transparent 86%);
            animation: weatherDrift 12s ease-in-out infinite alternate;
        }

        .weather-cloudy .weather-atmosphere {
            opacity: 0.52;
            background: linear-gradient(105deg, transparent 24%, rgba(175, 191, 205, 0.28) 55%, transparent 82%);
            animation: weatherDrift 16s ease-in-out infinite alternate;
        }

        .weather-light .weather-atmosphere {
            opacity: 0.28;
            background: linear-gradient(100deg, transparent 38%, rgba(220, 234, 242, 0.22) 64%, transparent 88%);
            animation: weatherDrift 18s ease-in-out infinite alternate;
        }

        .weather-clear .weather-atmosphere {
            opacity: 0.22;
            background: linear-gradient(0deg, rgba(94, 234, 212, 0.10), transparent 44%);
        }

        @keyframes weatherDrift {
            from { transform: translateX(-3%); }
            to { transform: translateX(3%); }
        }

        .landing-weather {
            position: absolute;
            top: 32px;
            right: 36px;
            z-index: 3;
            width: min(310px, calc(100vw - 72px));
            padding: 14px 16px;
            border: 1px solid rgba(255, 255, 255, 0.20);
            border-radius: 8px;
            background: rgba(8, 13, 26, 0.30);
            backdrop-filter: blur(14px);
            color: #f7fbff;
        }

        .landing-weather span,
        .landing-weather small {
            display: block;
            color: #a9bbcc;
            font-size: 11px;
        }

        .landing-weather strong {
            display: block;
            margin: 4px 0 5px;
            color: #ffffff;
            font-size: 19px;
        }

        .landing-weather p {
            margin: 0 0 6px;
            color: #dbe7f3;
            font-size: 12px;
            line-height: 1.55;
        }

        .landing-content {
            position: absolute;
            left: 50%;
            bottom: 9vh;
            transform: translateX(-50%);
            z-index: 1;
            width: min(760px, 92%);
            padding: 0 24px;
            text-align: center;
        }

        .landing-kicker {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            border: 1px solid rgba(94, 234, 212, 0.42);
            background: rgba(94, 234, 212, 0.12);
            color: #d7fffb;
            border-radius: 999px;
            padding: 8px 12px;
            font-size: 13px;
            font-weight: 750;
            margin-bottom: 20px;
        }

        .landing-title {
            font-family: 'Playfair Display', Georgia, serif;
            font-size: clamp(56px, 9vw, 118px);
            line-height: 0.98;
            font-weight: 900;
            font-style: italic;
            margin-bottom: 16px;
            text-wrap: balance;
            letter-spacing: -0.02em !important;
            text-shadow: 0 8px 38px rgba(0, 0, 0, 0.58);
        }

        .landing-copy {
            max-width: 620px;
            color: #d5deed;
            font-size: 18px;
            line-height: 1.72;
            margin: 0 auto 22px;
            text-shadow: 0 2px 22px rgba(0, 0, 0, 0.45);
        }

        .landing-actions {
            position: relative;
            z-index: 2;
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 12px;
            margin-top: 26px;
        }

        .landing-actions form {
            margin: 0;
        }

        .landing-cta,
        .landing-ghost {
            appearance: none;
            font: inherit;
            cursor: pointer;
        }

        .landing-cta {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-height: 46px;
            padding: 0 24px;
            border-radius: 999px;
            border: 1px solid rgba(255, 255, 255, 0.38);
            background: rgba(8, 13, 26, 0.20);
            color: #ffffff !important;
            font-weight: 850;
            text-decoration: none !important;
            backdrop-filter: blur(14px);
            box-shadow: 0 16px 46px rgba(0, 0, 0, 0.18);
        }

        .landing-cta:hover {
            background: rgba(255, 255, 255, 0.13);
            border-color: rgba(255, 255, 255, 0.72);
        }

        .landing-ghost {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-height: 46px;
            padding: 0 22px;
            border-radius: 999px;
            border: 1px solid rgba(255, 255, 255, 0.26);
            background: rgba(8, 13, 26, 0.14);
            color: #e5edf8 !important;
            font-weight: 750;
            text-decoration: none !important;
            backdrop-filter: blur(14px);
        }

        .landing-ghost:hover {
            background: rgba(255, 255, 255, 0.12);
            border-color: rgba(255, 255, 255, 0.34);
        }

        .star-map {
            position: relative;
            min-height: calc(100vh - 150px);
            overflow: hidden;
            border-radius: 18px;
            border: 1px solid rgba(255, 255, 255, 0.12);
            background-size: cover;
            background-position: center bottom;
            box-shadow: 0 24px 70px rgba(0, 0, 0, 0.28);
        }

        .star-shell {
            min-height: 100vh;
            margin: -1.2rem calc(50% - 50vw) 0;
        }

        .star-page {
            position: relative;
            min-height: 100vh;
            overflow: hidden;
            background-size: cover;
            background-position: center;
        }

        .star-page:before {
            content: "";
            position: absolute;
            inset: 0;
            background:
                linear-gradient(90deg, rgba(5,8,18,0.72) 0%, rgba(5,8,18,0.28) 52%, rgba(5,8,18,0.06) 100%),
                linear-gradient(0deg, rgba(5,8,18,0.62) 0%, rgba(5,8,18,0.04) 55%);
            pointer-events: none;
        }

        .star-page-title {
            position: absolute;
            left: 40px;
            top: 36px;
            z-index: 2;
            max-width: 460px;
        }

        .star-page-title h2 {
            margin: 0 0 10px;
            font-size: 36px;
        }

        .star-page-title p {
            margin: 0;
            color: #d5deed;
            font-size: 15px;
        }

        .star-page-back {
            position: absolute;
            right: 36px;
            top: 36px;
            z-index: 10;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 9px 18px;
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,0.22);
            background: rgba(8,13,26,0.55);
            backdrop-filter: blur(10px);
            color: #f0f6ff;
            font-size: 13px;
            font-weight: 700;
            cursor: pointer;
            text-decoration: none;
        }

        .star-page-back:hover {
            background: rgba(94,234,212,0.18);
            border-color: rgba(94,234,212,0.5);
            color: #5eead4;
        }

        .star-page-detail {
            position: absolute;
            bottom: 32px;
            left: 40px;
            right: 40px;
            z-index: 10;
            border: 1px solid rgba(255,255,255,0.16);
            border-radius: 16px;
            background: rgba(8, 13, 26, 0.78);
            backdrop-filter: blur(18px);
            padding: 20px 24px;
        }

        .star-map:before {
            content: "";
            position: absolute;
            inset: 0;
            background: linear-gradient(90deg, rgba(5,8,18,0.64), rgba(5,8,18,0.16) 58%, rgba(5,8,18,0.04));
            pointer-events: none;
        }

        .star-map-title {
            position: absolute;
            left: 28px;
            top: 26px;
            z-index: 2;
            max-width: 420px;
        }

        .star-map-title h2 {
            margin: 0 0 8px;
            font-size: 34px;
        }

        .star-map-title p {
            margin: 0;
            color: #d5deed;
        }

        .star-link {
            position: absolute;
            z-index: 3;
            width: 14px;
            height: 14px;
            border-radius: 999px;
            background: #f8fbff;
            box-shadow: 0 0 10px rgba(255,255,255,0.95), 0 0 24px rgba(94,234,212,0.65);
            border: 1px solid rgba(255,255,255,0.9);
            transform: translate(-50%, -50%);
            overflow: hidden;
            text-indent: -9999px;
            font-size: 0;
        }

        .star-link:hover {
            width: 20px;
            height: 20px;
            background: #5eead4;
        }

        .star-detail {
            border: 1px solid rgba(255,255,255,0.16);
            border-radius: 16px;
            background: rgba(8, 13, 26, 0.72);
            backdrop-filter: blur(16px);
            padding: 18px;
            margin-top: 14px;
        }

        .landing-card-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 16px;
        }

        .landing-card {
            min-height: 178px;
            border-radius: 18px;
            border: 1px solid rgba(255, 255, 255, 0.13);
            background:
                linear-gradient(135deg, rgba(255, 255, 255, 0.09), rgba(255, 255, 255, 0.035)),
                rgba(11, 16, 31, 0.72);
            box-shadow: 0 24px 70px rgba(0, 0, 0, 0.24);
            padding: 22px;
        }

        .landing-card .num {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 34px;
            height: 34px;
            border-radius: 999px;
            background: linear-gradient(135deg, rgba(94, 234, 212, 0.30), rgba(255, 94, 168, 0.26));
            margin-bottom: 16px;
            font-weight: 900;
        }

        .landing-card h3 {
            font-size: 20px;
            margin: 0 0 10px;
        }

        .landing-card p {
            margin: 0;
            color: #b9c6d8;
            line-height: 1.65;
        }

        .landing-route {
            display: grid;
            grid-template-columns: 0.8fr 1.2fr;
            gap: 16px;
            align-items: stretch;
        }

        .landing-route-main {
            border-radius: 18px;
            border: 1px solid rgba(94, 234, 212, 0.22);
            background: linear-gradient(135deg, rgba(94, 234, 212, 0.14), rgba(124, 140, 255, 0.10));
            padding: 24px;
        }

        .timeline {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 10px;
        }

        .timeline-step {
            border-radius: 14px;
            border: 1px solid rgba(255, 255, 255, 0.13);
            background: rgba(255, 255, 255, 0.06);
            padding: 14px;
        }

        .timeline-step strong {
            display: block;
            margin-bottom: 8px;
            color: #f7c948;
        }

        @media (max-width: 900px) {
            .metric-grid, .event-grid, .landing-card-grid, .landing-route, .timeline {
                grid-template-columns: 1fr;
            }
            .hero h1 {
                font-size: 36px;
            }
            .landing-hero {
                min-height: 100dvh;
                background-position: 58% center;
            }
            .landing-content {
                bottom: 7vh;
                padding: 0 20px;
            }
            .landing-weather {
                top: 20px;
                right: 20px;
                width: min(280px, calc(100vw - 40px));
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hide_sidebar_for_landing() -> None:
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            display: none;
        }
        [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"], header {
            display: none !important;
        }
        html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
            height: 100dvh;
            overflow: hidden;
        }
        .block-container {
            max-width: none;
            padding: 0;
            height: 100dvh;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def organization_weather(data: dict) -> dict:
    ideas = [idea for idea in data["ideas"] if idea.get("status") != "已合并"]
    active = [idea for idea in ideas if idea.get("status") not in {"已完成", "已合并"}]
    unanswered = [idea for idea in active if not idea.get("management_response")]
    high_unanswered = [idea for idea in unanswered if int(idea.get("heat", 0) or 0) >= 70]
    responded = [idea for idea in ideas if idea.get("management_response")]
    completed = sum(1 for idea in ideas if idea.get("status") == "已完成") + sum(
        1 for task in data["tasks"] if task.get("status") == "已完成"
    )
    category_counter = Counter(idea.get("category", "综合建议") for idea in active)
    top_category = category_counter.most_common(1)[0][0] if category_counter else "暂无集中议题"
    response_rate = round(len(responded) / max(len(ideas), 1) * 100)

    if len(high_unanswered) >= 2:
        return {
            "class": "weather-fog",
            "label": tx("山间浓雾", "山間に濃霧"),
            "summary": tx(
                f"{len(high_unanswered)} 条高热反馈仍待回应，建议优先明确负责人。",
                f"注目度の高い{len(high_unanswered)}件が未回答です。まず担当者を明確にしましょう。",
            ),
            "meta": tx(
                f"回应率 {response_rate}% · 关注 {top_category}",
                f"回答率 {response_rate}% · 注目 {ui_value(top_category)}",
            ),
        }
    if top_category == "沟通协同" and unanswered:
        return {
            "class": "weather-cloudy",
            "label": tx("流云偏多", "流れ雲多め"),
            "summary": tx("沟通协同声音正在聚集，信息需要更早抵达相关人员。", "連携に関する声が集まっています。必要な人へより早く情報を届けましょう。"),
            "meta": tx(f"待回应 {len(unanswered)} 条 · 回应率 {response_rate}%", f"未回答 {len(unanswered)}件 · 回答率 {response_rate}%"),
        }
    if completed and response_rate >= 60:
        return {
            "class": "weather-clear",
            "label": tx("富士晴朗", "富士快晴"),
            "summary": tx(f"已有 {completed} 个议题完成闭环，组织回应正在形成稳定节奏。", f"{completed}件の課題が完了し、組織の回答リズムが整いつつあります。"),
            "meta": tx(f"回应率 {response_rate}% · 已闭环 {completed} 项", f"回答率 {response_rate}% · 完了 {completed}件"),
        }
    return {
        "class": "weather-light",
        "label": tx("薄云待晴", "薄曇り、晴れ待ち"),
        "summary": tx("目前没有明显风暴，但仍有声音等待第一次正式回应。", "大きな嵐はありませんが、まだ最初の正式回答を待つ声があります。"),
        "meta": tx(f"待回应 {len(unanswered)} 条 · 关注 {top_category}", f"未回答 {len(unanswered)}件 · 注目 {ui_value(top_category)}"),
    }


def render_landing(data: dict) -> None:
    hide_sidebar_for_landing()
    hero_uri = image_data_uri(HERO_IMAGE)
    weather = organization_weather(data)

    st.markdown(
        f"""
        <style>
        .landing-weather-head {{ display:flex; align-items:center; justify-content:space-between; gap:12px; }}
        .landing-language {{ display:flex; gap:4px; flex:0 0 auto; }}
        .landing-language form {{ margin:0; }}
        .landing-language button {{ border:0; background:transparent; color:#a9bbcc; padding:3px 6px; border-radius:4px; font-size:10px; line-height:1.2; cursor:pointer; }}
        .landing-language button:hover {{ color:#ffffff; background:rgba(255,255,255,.08); }}
        .landing-language button.active {{ color:#5eead4; background:rgba(94,234,212,.10); }}
        </style>
        <div class="landing-shell">
            <section class="landing-hero {weather['class']}" style="background-image: url('{hero_uri}');">
                <div class="weather-atmosphere"></div>
                <div class="landing-weather">
                    <div class="landing-weather-head">
                        <span>{tx("富士山组织天气", "富士山 組織の天気")}</span>
                        <div class="landing-language">
                            <form method="get"><input type="hidden" name="lang" value="zh"><button class="{'active' if current_language() == 'zh' else ''}" type="submit">中文</button></form>
                            <form method="get"><input type="hidden" name="lang" value="ja"><button class="{'active' if current_language() == 'ja' else ''}" type="submit">日本語</button></form>
                        </div>
                    </div>
                    <strong>{html.escape(weather['label'])}</strong>
                    <p>{html.escape(weather['summary'])}</p>
                    <small>{html.escape(weather['meta'])}</small>
                </div>
                <div class="landing-content">
                    <div class="landing-title">Stellar</div>
                    <div class="landing-copy">
                        {tx("让每一个想法被看见，让每一次反馈有回声。", "すべての想いを見える形に。すべての声に、応答を。")}
                    </div>
                    <div class="landing-actions">
                        <form method="get">
                            <input type="hidden" name="view" value="workspace">
                            <input type="hidden" name="page" value="submit">
                            {language_query_field()}
                            <button type="submit" class="landing-cta">{tx("提交反馈", "声を届ける")}</button>
                        </form>
                        <form method="get">
                            <input type="hidden" name="view" value="workspace">
                            <input type="hidden" name="page" value="progress">
                            {language_query_field()}
                            <button type="submit" class="landing-ghost">{tx("查看进度", "進捗を見る")}</button>
                        </form>
                        <form method="get">
                            <input type="hidden" name="view" value="stars">
                            {language_query_field()}
                            <button type="submit" class="landing-ghost">{tx("星空意见图", "星空ボイスマップ")}</button>
                        </form>
                        <form method="get">
                            <input type="hidden" name="view" value="echoes">
                            {language_query_field()}
                            <button type="submit" class="landing-ghost">{tx("回声墙", "エコーウォール")}</button>
                        </form>
                    </div>
                </div>
            </section>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero(data: dict) -> None:
    ideas = data["ideas"]
    tasks = data["tasks"]
    open_tasks = sum(1 for t in tasks if t["status"] in {"待确认", "已受理", "推进中"})
    avg_heat = round(sum(i["heat"] for i in ideas) / max(len(ideas), 1))
    completed = sum(1 for t in tasks if t["status"] == "已完成")
    categories = len({i["category"] for i in ideas})

    st.markdown(
        f"""
        <div class="hero">
            <h1><span class="pulse-dot"></span>{APP_TITLE}</h1>
            <p>{tx("统一提交反馈，公开查看进度。让问题有人看见，也有机会被跟进。", "声をひとつの場所に集め、進捗を公開。課題を見える形にし、次の行動へつなげます。")}</p>
        </div>
        <div class="metric-grid">
            <div class="metric-card"><div class="label">{tx("反馈数量", "声の数")}</div><div class="value">{len(ideas)}</div><div class="hint">{tx("已提交的员工反馈", "届けられたフィードバック")}</div></div>
            <div class="metric-card"><div class="label">{tx("处理中事项", "対応中の案件")}</div><div class="value">{open_tasks}</div><div class="hint">{tx("待确认、已受理或推进中", "確認待ち・受付済み・進行中")}</div></div>
            <div class="metric-card"><div class="label">{tx("平均热度", "平均注目度")}</div><div class="value">{avg_heat}%</div><div class="hint">{tx("影响、紧急、共鸣、重复、可执行", "影響・緊急性・共感・重複・実行性")}</div></div>
            <div class="metric-card"><div class="label">{tx("反馈类型", "カテゴリ")}</div><div class="value">{categories}</div><div class="hint">{tx("自动分类统计", "自動分類の集計")}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if completed:
        st.toast(tx(f"已有 {completed} 个事项完成闭环", f"{completed}件の案件が完了しました"))


def query_form_fields(**params: str) -> str:
    fields = dict(params)
    fields.setdefault("lang", current_language())
    return "\n".join(
        f'<input type="hidden" name="{html.escape(str(key))}" value="{html.escape(str(value), quote=True)}">'
        for key, value in fields.items()
    )


def liked_idea_ids() -> set[str]:
    if "liked_idea_ids" not in st.session_state:
        st.session_state["liked_idea_ids"] = set()
    return st.session_state["liked_idea_ids"]


def get_session_token() -> str:
    if "session_token" not in st.session_state:
        st.session_state["session_token"] = uuid4().hex
    return st.session_state["session_token"]


def render_idea_card(idea: dict, show_actions: bool = False, allow_delete: bool = False) -> None:
    color = status_color(idea["status"])
    idea_id = str(idea["id"])
    factors = idea.get("heat_factors") or calculate_heat_factors(idea)
    factor_html = "".join(
        f'<span class="tag">{label} {int(factors.get(key, 0))}</span>'
        for key, label in [
            ("impact", tx("影响", "影響")),
            ("urgency", tx("紧急", "緊急性")),
            ("resonance", tx("共鸣", "共感")),
            ("duplication", tx("重复", "重複")),
            ("actionability", tx("可执行", "実行性")),
        ]
    )
    is_liked = idea_id in liked_idea_ids()
    like_class = "idea-action-button is-liked" if is_liked else "idea-action-button"
    like_text = tx("已赞同", "共感済み") if is_liked else tx("赞同这个反馈", "この声に共感")
    action_html = ""
    if show_actions:
        action_html = f"""
            <div class="idea-actions">
                <form method="get">
                    {query_form_fields(view="workspace", page="progress", op="like", idea=idea_id)}
                    <button class="{like_class}" type="submit">{like_text}</button>
                </form>
        """
        if allow_delete and (idea.get("delete_code_hash") or idea.get("delete_code")):
            action_html += f"""
                <form method="get">
                    {query_form_fields(view="workspace", page="progress", op="delete", idea=idea_id)}
                    <button class="idea-delete-button" type="submit">{tx("删除反馈", "削除")}</button>
                </form>
            """
        action_html += "</div>"
    response_html = ""
    if idea.get("management_response"):
        response_html = (
            f'<div class="management-response"><strong>{tx("管理层回应", "管理層からの回答")}</strong><br>'
            f'{html.escape(idea["management_response"])}'
            f'<br><span class="muted">{tx("负责人", "担当者")}：{html.escape(ui_value(idea.get("owner", "待确认")))}'
            f' · {tx("下次更新", "次回更新")}：{html.escape(ui_value(idea.get("next_update_at", "待定") or "待定"))}</span></div>'
        )
    merged_html = ""
    if idea.get("merged_into_id"):
        merged_html = f'<div class="management-response"><strong>{tx("已合并到主议题", "主要テーマに統合済み")}</strong><br>{tx("原始反馈仍被保留，后续进展请查看主议题。", "元の声は保存されています。今後の進捗は主要テーマをご覧ください。")}</div>'

    st.html(
        f"""
        <div class="idea-card">
            <div class="idea-head">
                <div>
                    <div class="idea-title">{html.escape(idea["title"])}</div>
                    <span class="tag">{html.escape(ui_value(idea["category"]))}</span>
                    <span class="tag" style="border-color:{color}; color:{color};">{html.escape(ui_value(idea["status"]))}</span>
                    <span class="tag">{tx("来自", "投稿者")}：{html.escape(ui_value(idea["author"]))}</span>
                </div>
                <div class="heat">{idea["heat"]}%<br><span style="font-size:11px;color:#dbeafe;">{tx("热度", "注目度")}</span></div>
            </div>
            <p>{html.escape(idea["content"])}</p>
            <p class="muted">{tx("希望如何改进", "期待する改善")}：{html.escape(idea["impact"])}</p>
            <span class="tag">{tx("赞同", "共感")} {idea["votes"]}</span>
            <span class="tag">{html.escape(idea["created_at"])}</span>
            <div class="heat-breakdown">{factor_html}</div>
            {response_html}
            {merged_html}
            {action_html}
        </div>
        """
    )


def render_status_timeline(idea: dict) -> None:
    history = idea.get("history") or [
        {
            "to_status": idea.get("status", "待确认"),
            "response": "反馈已提交，等待确认。",
            "owner": idea.get("owner", "待确认"),
            "actor": "系统",
            "created_at": idea.get("created_at", ""),
        }
    ]
    entries = []
    for event in history:
        status = html.escape(ui_value(event.get("to_status") or tx("状态更新", "ステータス更新")))
        actor = html.escape(ui_value(event.get("actor") or "系统"))
        created_at = html.escape(str(event.get("created_at") or ""))
        response = html.escape(ui_system_text(event.get("response") or ""))
        owner = html.escape(str(event.get("owner") or ""))
        detail = response or (f'{tx("负责人", "担当者")}：{owner}' if owner else "")
        entries.append(
            f'<div class="timeline-entry"><strong>{status} · {actor}</strong>'
            f'<span>{created_at}{(" · " + detail) if detail else ""}</span></div>'
        )
    st.html(f'<div class="status-timeline">{"".join(entries)}</div>')


def finalize_idea_submission(data: dict, idea: dict) -> None:
    data["ideas"].insert(0, idea)
    persist_new_idea(data, idea)
    st.session_state.pop("pending_similar_submission", None)
    st.session_state["pending_toast"] = tx("反馈已提交。", "声を受け付けました。")
    st.query_params["view"] = "workspace"
    st.query_params["page"] = "progress"
    st.rerun()


def render_submit_form(data: dict) -> None:
    st.markdown(f'<div class="section-title">{tx("填写反馈", "声を書く")}</div>', unsafe_allow_html=True)
    pending = st.session_state.get("pending_similar_submission")
    if pending:
        candidate = normalize_idea(pending["idea"])
        match_ids = pending.get("match_ids", [])
        matches = [idea for idea in data["ideas"] if idea["id"] in match_ids]
        st.warning(tx("发现可能相似的已有议题。你可以直接支持它，避免同一个问题被分散。", "似た内容のテーマがあります。新規投稿の代わりに共感を送ることもできます。"))
        for match in matches:
            score = round(idea_similarity(candidate, match) * 100)
            st.html(
                f'<div class="glass-card"><span class="tag">{tx("相似度", "類似度")} {score}%</span>'
                f'<span class="tag">{tx("热度", "注目度")} {match["heat"]}%</span>'
                f'<div class="idea-title">{html.escape(match["title"])}</div>'
                f'<p class="muted">{html.escape(match["content"][:180])}</p></div>'
            )
            if st.button(tx(f'支持已有议题「{match["title"][:18]}」', f'「{match["title"][:18]}」に共感する'), key=f'join_{match["id"]}', use_container_width=True):
                if persist_vote(match, get_session_token(), data):
                    liked_idea_ids().add(match["id"])
                    message = tx("已加入已有议题，并增加一份共鸣。", "既存テーマに共感を送りました。")
                else:
                    message = tx("你已经支持过这个议题。", "このテーマには共感済みです。")
                st.session_state.pop("pending_similar_submission", None)
                st.session_state["pending_toast"] = message
                st.query_params["view"] = "workspace"
                st.query_params["page"] = "progress"
                st.rerun()
        left, right = st.columns(2)
        if left.button(tx("仍然提交为新反馈", "新しい声として投稿"), type="primary", use_container_width=True):
            finalize_idea_submission(data, candidate)
        if right.button(tx("取消本次提交", "キャンセル"), use_container_width=True):
            st.session_state.pop("pending_similar_submission", None)
            st.rerun()
        st.divider()
        return

    with st.form("idea_form", clear_on_submit=True):
        col1, col2 = st.columns([1.1, 0.9])
        with col1:
            title = st.text_input(tx("标题", "タイトル"), placeholder=tx("例如：希望建立固定的信息同步机制", "例：定期的な情報共有の仕組みがほしい"))
            content = st.text_area(tx("反馈内容", "内容"), placeholder=tx("描述发生了什么、影响了谁、为什么值得处理", "何が起きたか、誰に影響したか、なぜ改善したいかを書いてください"))
            impact = st.text_area(tx("希望如何改进", "期待する改善"), placeholder=tx("写下你期待的处理方式或建议", "期待する対応や提案を書いてください"))
        with col2:
            author = st.text_input(tx("署名", "お名前"), placeholder=tx("可填写昵称", "ニックネームでも可"))
            anonymous = st.toggle(tx("匿名提交", "匿名で投稿"), value=True)
            delete_code = st.text_input(tx("删除码（4位数字，记住它才能删除自己的反馈）*", "削除コード（4桁の数字、投稿の削除に必要）*"), max_chars=4, placeholder=tx("如：1234", "例：1234"))
            submitted = st.form_submit_button(tx("提交反馈", "声を届ける"))

    if submitted:
        if not title.strip() or not content.strip():
            st.warning(tx("标题和问题描述需要先写一下。", "タイトルと内容を入力してください。"))
            return
        if not delete_code.strip() or not delete_code.strip().isdigit() or len(delete_code.strip()) != 4:
            st.warning(tx("删除码必须是4位数字（提交后无法修改，请记住它）。", "削除コードは4桁の数字で入力してください。投稿後は変更できません。"))
            return
        category, _priority, heat = classify_text(f"{title} {content} {impact}")
        idea_id = f"idea-{uuid4().hex[:8]}"
        created_at = now_str()
        new_idea = normalize_idea(
            {
                "id": idea_id,
                "title": title.strip(),
                "category": category,
                "author": "匿名" if anonymous else (author.strip() or "未署名同事"),
                "anonymous": anonymous,
                "content": content.strip(),
                "impact": impact.strip() or polish_text(content),
                "status": "待确认",
                "base_heat": heat,
                "heat": heat,
                "votes": 1,
                "created_at": created_at,
                "delete_code_hash": hash_delete_code(delete_code.strip(), idea_id),
                "history": [
                    {
                        "from_status": "",
                        "to_status": "待确认",
                        "response": "反馈已提交，等待协同小组确认。",
                        "owner": "待确认",
                        "actor": "系统",
                        "created_at": created_at,
                    }
                ],
            }
        )
        similar = find_similar_ideas(new_idea, data["ideas"])
        if similar:
            st.session_state["pending_similar_submission"] = {
                "idea": new_idea,
                "match_ids": [idea["id"] for idea, _score in similar],
            }
            st.rerun()
        finalize_idea_submission(data, new_idea)


def render_ideas(data: dict) -> None:
    render_submit_form(data)
    st.markdown('<div class="section-title">想法广场</div>', unsafe_allow_html=True)
    categories = ["全部"] + sorted({i["category"] for i in data["ideas"]})
    selected = st.segmented_control("筛选类型", categories, default="全部")
    for idea in data["ideas"]:
        if selected != "全部" and idea["category"] != selected:
            continue
        render_idea_card(idea)


def render_translator(data: dict) -> None:
    st.markdown(f'<div class="section-title">{tx("AI 整理表达", "AIで表現を整える")}</div>', unsafe_allow_html=True)
    st.caption(tx("把原始想法整理成更清楚、可处理的反馈。", "そのままの思いを、明確で対応しやすい表現に整えます。"))
    deepseek_key, deepseek_default = get_deepseek_config()
    gemini_key, gemini_default = get_gemini_config()

    provider_options = ["DeepSeek", "Gemini", "本地规则"]

    with st.form("translator_form"):
        raw = st.text_area(
            tx("想说的话", "伝えたいこと"),
            value="",
            height=130,
            placeholder=tx("直接写真实想法即可，例如：部门之间信息不同步，经常不知道事情推进到哪里。", "そのままの言葉で大丈夫です。例：部門間の情報共有が遅く、仕事がどこまで進んでいるか分からない。"),
        )
        target = st.radio(
            tx("整理用途", "宛先"),
            ["给管理层", "给协同负责人", "给活动负责人"],
            horizontal=True,
        )
        provider = st.selectbox(
            tx("AI 服务", "AIサービス"),
            provider_options,
            index=0 if deepseek_key else 2,
            help=f"DeepSeek：{'已配置' if deepseek_key else '未配置'}；Gemini：{'已配置' if gemini_key else '未配置'}",
        )
        if provider == "DeepSeek":
            default_index = DEEPSEEK_MODELS.index(deepseek_default) if deepseek_default in DEEPSEEK_MODELS else 0
            model = st.selectbox("模型", DEEPSEEK_MODELS, index=default_index)
        elif provider == "Gemini":
            default_index = GEMINI_MODELS.index(gemini_default) if gemini_default in GEMINI_MODELS else 0
            model = st.selectbox("模型", GEMINI_MODELS, index=default_index)
        else:
            model = "local"
        submitted = st.form_submit_button(tx("整理表达", "表現を整える"))

    if submitted:
        if not raw.strip():
            st.warning(tx("先写一点想反馈的内容。", "まず伝えたいことを書いてください。"))
            return
        if provider in {"DeepSeek", "Gemini"}:
            with st.spinner(f"{provider} 正在整理..."):
                try:
                    result = call_ai_translation(provider, raw, target, model)
                except Exception as exc:
                    st.warning(f"{provider} 调用失败，已回退本地规则：{exc}")
                    result = translate_emotion(raw, target)
        else:
            result = translate_emotion(raw, target)
        st.session_state["translator_result"] = result
        st.session_state["translator_raw"] = raw
        st.session_state["translator_target"] = target

    result = st.session_state.get("translator_result")
    if result:
        preview_idea = normalize_idea(
            {
                "title": result["title"],
                "category": result["category"],
                "author": "AI 转译",
                "content": result["translated"],
                "impact": result["next_step"],
                "status": "待确认",
                "votes": 1,
                "created_at": now_str(),
            }
        )
        preview_factors = preview_idea.get("heat_factors", {})
        factor_preview = " · ".join(
            f"{label}{int(preview_factors.get(key, 0))}"
            for key, label in [
                ("impact", "影响"),
                ("urgency", "紧急"),
                ("resonance", "共鸣"),
                ("duplication", "重复"),
                ("actionability", "可执行"),
            ]
        )
        left, right = st.columns([1.05, 0.95])
        with left:
            st.markdown(
                f"""
                <div class="glass-card">
                    <span class="tag">来源：{result.get("source", "本地规则")}</span>
                    <span class="tag">{result["category"]}</span>
                    <span class="tag">优先级：{result["priority"]}</span>
                    <span class="tag">预计热度：{preview_idea["heat"]}%</span>
                    <div class="idea-title">{result["title"]}</div>
                    <p>{result["translated"]}</p>
                    <p class="muted">建议下一步：{result["next_step"]}</p>
                    <p class="muted">热度构成：{factor_preview}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with right:
            st.markdown(
                f"""
                <div class="glass-card">
                    <div class="idea-title">处理建议</div>
                    <p class="muted">表达风格：{result["tone"]}</p>
                    <p class="muted">建议把这条反馈提交到反馈列表，由负责人确认是否需要转成事项。</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if st.button("把转译结果送入查看进度", key="send_translator_result", use_container_width=True):
            created_at = now_str()
            translated_idea = normalize_idea(
                {
                    "id": f"idea-{uuid4().hex[:8]}",
                    "title": result["title"],
                    "category": result["category"],
                    "author": "AI 转译",
                    "anonymous": True,
                    "content": result["translated"],
                    "impact": result["next_step"],
                    "status": "待确认",
                    "base_heat": preview_idea["heat"],
                    "heat": preview_idea["heat"],
                    "votes": 1,
                    "created_at": created_at,
                    "history": [
                        {
                            "from_status": "",
                            "to_status": "待确认",
                            "response": "AI 已协助整理表达，等待协同小组确认。",
                            "owner": "待确认",
                            "actor": "系统",
                            "created_at": created_at,
                        }
                    ],
                }
            )
            data["ideas"].insert(0, translated_idea)
            persist_new_idea(data, translated_idea)
            st.session_state.pop("translator_result", None)
            st.session_state.pop("translator_raw", None)
            st.session_state.pop("translator_target", None)
            st.session_state["pending_toast"] = "已送入查看进度列表。"
            st.query_params["view"] = "workspace"
            st.query_params["page"] = "progress"
            st.rerun()


def render_task_card(task: dict) -> None:
    color = status_color(task["status"])
    members = html.escape(" / ".join(task["members"]))
    plan_html = f'<p class="muted">{tx("实施方案", "実施計画")}：{html.escape(task["plan"])}</p>' if task.get("plan") else ""
    st.html(
        f"""
        <div class="task-card">
            <div class="idea-head">
                <div>
                    <div class="idea-title">{html.escape(task["name"])}</div>
                    <span class="tag" style="border-color:{color}; color:{color};">{html.escape(ui_value(task["status"]))}</span>
                    <span class="tag">{tx("优先级", "優先度")}：{html.escape(ui_value(task["priority"]))}</span>
                    <span class="tag">{tx("截止", "期限")}：{html.escape(ui_value(task["due"]))}</span>
                </div>
                <div class="heat">{task["progress"]}%<br><span style="font-size:11px;color:#dbeafe;">{tx("进度", "進捗")}</span></div>
            </div>
            <div class="progress-shell"><div class="progress-bar" style="width:{task["progress"]}%;"></div></div>
            <p>{tx("负责人", "担当者")}：{html.escape(ui_value(task["owner"]))}</p>
            <p class="muted">{tx("参与方", "参加者")}：{members}</p>
            <p class="muted">{tx("下一步", "次のアクション")}：{html.escape(task["next_step"])}</p>
            {plan_html}
            <span class="tag">{tx("激励", "インセンティブ")}：{html.escape(task["reward"])}</span>
        </div>
        """
    )


def render_tasks(data: dict) -> None:
    st.markdown(f'<div class="section-title">{tx("事项看板", "案件ボード")}</div>', unsafe_allow_html=True)
    st.caption(tx('把“有人提了但没人接”的事情变成有状态、有负责人、有下一步的协作任务。', '「声は上がったのに誰も拾わない」課題を、ステータス、担当者、次の行動が見える案件に変えます。'))

    statuses = ["待确认", "已受理", "推进中", "已完成", "暂缓", "已合并"]
    cols = st.columns(len(statuses))
    for col, status in zip(cols, statuses):
        count = sum(1 for task in data["tasks"] if task["status"] == status)
        col.metric(ui_value(status), count)

    if is_admin():
        with st.expander("创建新事项", expanded=False):
            with st.form("task_form", clear_on_submit=True):
                c1, c2, c3 = st.columns(3)
                name = c1.text_input("事项名称")
                owner = c2.text_input("负责人", value="待确认")
                due = c3.date_input("截止日期")
                next_step = st.text_area("下一步动作")
                priority = st.selectbox("优先级", ["高", "中", "低"])
                reward = st.text_input("建议激励", value="纳入试点贡献记录")
                create = st.form_submit_button("生成事项卡")
            if create and name.strip():
                data["tasks"].insert(
                    0,
                    {
                        "id": f"task-{uuid4().hex[:8]}",
                        "name": name.strip(),
                        "owner": owner.strip() or "待确认",
                        "status": "待确认",
                        "priority": priority,
                        "progress": 8,
                        "due": str(due),
                        "reward": reward.strip() or "待确认",
                        "members": ["员工协同小组"],
                        "next_step": next_step.strip() or "等待负责人确认",
                    },
                )
                save_data(data)
                st.success("事项卡已生成。")
                st.rerun()

    for task in data["tasks"]:
        render_task_card(task)
        if not is_admin():
            continue
        plan = task.get("plan", "")
        with st.expander("实施方案" + ("  ✓" if plan else ""), expanded=False):
            new_plan = st.text_area(
                "具体实施方案",
                value=plan,
                placeholder="描述如何推进：分几步、谁负责哪块、预计时间节点……",
                key=f"plan_{task['id']}",
                height=120,
            )
            status_options = ["待确认", "已受理", "推进中", "已完成", "暂缓"]
            current_idx = status_options.index(task["status"]) if task["status"] in status_options else 0
            new_status = st.selectbox("更新状态", status_options, index=current_idx, key=f"status_{task['id']}")
            if st.button("保存", key=f"save_{task['id']}"):
                if new_status == "已完成" and not new_plan.strip():
                    st.error("标记为「已完成」前请先填写实施方案。")
                else:
                    task["plan"] = new_plan.strip()
                    task["status"] = new_status
                    if new_status == "已完成":
                        task["progress"] = 100
                    elif new_status == "推进中" and task["progress"] < 10:
                        task["progress"] = 30
                    save_data(data)
                    st.toast("已保存。")
                    st.rerun()


def render_event(data: dict) -> None:
    if not data.get("events"):
        st.info("暂无活动。")
        return
    event = data["events"][0]
    slots = event.get("slots") or []
    done = sum(1 for slot in slots if slot.get("done"))
    total = len(slots) or 1
    progress = round(done / total * 100)
    st.markdown('<div class="section-title">活动共创实验室</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="glass-card">
            <div class="idea-head">
                <div>
                    <div class="idea-title">{event["name"]}</div>
                    <span class="tag">{event["status"]}</span>
                    <span class="tag">任务完成 {done}/{total}</span>
                    <span class="tag">建议：组织补贴 + 复盘报告</span>
                </div>
                <div class="heat">{progress}%<br><span style="font-size:11px;color:#dbeafe;">筹备</span></div>
            </div>
            <div class="progress-shell"><div class="progress-bar" style="width:{progress}%;"></div></div>
            <p class="muted">示例场景：同样是烧烤活动，过去靠临时摊派；现在可以让员工投票、认领任务、申请资源、活动后复盘。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.05, 0.95])
    with left:
        st.markdown("#### 共创任务")
        for slot in event["slots"]:
            mark = "完成" if slot["done"] else "待认领"
            cls = "slot-done" if slot["done"] else "slot-open"
            st.markdown(
                f"""
                <div class="slot">
                    <div>{slot["name"]}<br><span class="muted">负责人：{slot["owner"]}</span></div>
                    <div class="{cls}">{mark}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with right:
        st.markdown("#### 时间投票")
        vote_df = pd.DataFrame(
            [{"选项": k, "票数": v} for k, v in event["votes"].items()]
        )
        st.bar_chart(vote_df, x="选项", y="票数", color="#5eead4")
        st.markdown("#### 口味偏好")
        pref_df = pd.DataFrame(
            [{"偏好": k, "热度": v} for k, v in event["preferences"].items()]
        )
        st.bar_chart(pref_df, x="偏好", y="热度", color="#ff5ea8")


def build_brief(data: dict) -> str:
    ideas = data["ideas"]
    tasks = data["tasks"]
    category_counter = Counter(i["category"] for i in ideas)
    hot_ideas = sorted(ideas, key=lambda item: item["heat"], reverse=True)[:3]
    open_tasks = [t for t in tasks if t["status"] in {"待确认", "已受理", "推进中"}]
    category_text = "、".join(f"{k} {v} 项" for k, v in category_counter.most_common())
    hot_text = "\n".join(
        f"- {idea['title']}（{idea['category']}，热度 {idea['heat']}%）"
        for idea in hot_ideas
    )
    task_text = "\n".join(
        f"- {task['name']}：{task['status']}，负责人 {task['owner']}，下一步：{task['next_step']}"
        for task in open_tasks[:5]
    )
    return f"""# Pulse Hub 员工脉冲简报

生成时间：{now_str()}

## 本期概览
- 收集想法：{len(ideas)} 项
- 开放事项：{len(open_tasks)} 项
- 议题分布：{category_text or "暂无"}

## 高频关注
{hot_text or "- 暂无高频议题"}

## 当前推进事项
{task_text or "- 暂无开放事项"}

## AI 建议
1. 优先处理热度高且影响范围广的沟通协同问题，避免信息差继续扩大。
2. 对公共事务组织建立补贴、调休或贡献记录，防止"临时有空的人"持续承担隐性成本。
3. 以烧烤活动作为第一个共创样板，跑通投票、认领、预算、执行、复盘的完整闭环。
"""


def render_report(data: dict) -> None:
    st.markdown('<div class="section-title">AI 周报与提案素材</div>', unsafe_allow_html=True)
    st.caption("这里的 AI 先用本地规则模拟，方便无成本部署；后续可以接入你们自己的 Agent 能力。")
    brief = build_brief(data)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown(brief)
    st.markdown("</div>", unsafe_allow_html=True)
    st.download_button(
        "下载本期简报 Markdown",
        data=brief,
        file_name=f"pulse_brief_{datetime.now().strftime('%Y%m%d')}.md",
        mime="text/markdown",
    )

    st.markdown("#### 可放进计划书的核心表述")
    st.info(
        "本试点通过员工协同小组与 Pulse Hub 工具，把分散意见、临时安排和公共事务转化为可记录、可分配、可反馈、可复盘的事项闭环。"
    )


def render_management_dashboard(data: dict) -> None:
    st.markdown('<div class="section-title">管理层看板</div>', unsafe_allow_html=True)
    st.caption("给上层看的不是零散意见，而是组织风险、决策事项和推进状态。")

    ideas = data["ideas"]
    tasks = data["tasks"]
    open_tasks = [t for t in tasks if t["status"] in {"待确认", "已受理", "推进中"}]
    blocked_tasks = [t for t in open_tasks if t["progress"] < 35]
    hot_ideas = sorted(ideas, key=lambda item: item["heat"], reverse=True)[:5]
    category_counter = Counter(i["category"] for i in ideas)
    status_counter = Counter(t["status"] for t in tasks)
    avg_progress = round(sum(t["progress"] for t in tasks) / max(len(tasks), 1))
    decision_items = [
        task
        for task in open_tasks
        if "预算" in task["next_step"] or "确认" in task["next_step"] or task["priority"] == "高"
    ][:4]

    st.markdown(
        f"""
        <div class="metric-grid">
            <div class="metric-card"><div class="label">需关注议题</div><div class="value">{len(hot_ideas)}</div><div class="hint">按热度和影响范围排序</div></div>
            <div class="metric-card"><div class="label">待决策事项</div><div class="value">{len(decision_items)}</div><div class="hint">需要管理层确认资源或方向</div></div>
            <div class="metric-card"><div class="label">推进均值</div><div class="value">{avg_progress}%</div><div class="hint">事项看板平均进度</div></div>
            <div class="metric-card"><div class="label">低进度事项</div><div class="value">{len(blocked_tasks)}</div><div class="hint">可能需要补负责人或资源</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1, 1])
    with left:
        st.markdown("#### 高频议题分布")
        category_df = pd.DataFrame(
            [{"议题": key, "数量": value} for key, value in category_counter.items()]
        )
        if not category_df.empty:
            st.bar_chart(category_df, x="议题", y="数量", color="#5eead4")
        st.markdown("#### 管理层本周应看")
        for idea in hot_ideas[:3]:
            st.markdown(
                f"""
                <div class="idea-card">
                    <span class="tag">{idea["category"]}</span>
                    <span class="tag">热度 {idea["heat"]}%</span>
                    <div class="idea-title">{idea["title"]}</div>
                    <p class="muted">{idea["impact"]}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with right:
        st.markdown("#### 事项状态")
        status_df = pd.DataFrame(
            [{"状态": status, "数量": count} for status, count in status_counter.items()]
        )
        if not status_df.empty:
            st.bar_chart(status_df, x="状态", y="数量", color="#ff5ea8")
        st.markdown("#### 需要拍板")
        for task in decision_items:
            st.markdown(
                f"""
                <div class="task-card">
                    <span class="tag">优先级：{task["priority"]}</span>
                    <span class="tag">{task["status"]}</span>
                    <div class="idea-title">{task["name"]}</div>
                    <p class="muted">需要确认：{task["next_step"]}</p>
                    <p class="muted">建议激励：{task["reward"]}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("#### 管理层摘要")
    st.info(
        "当前最值得优先处理的是沟通链路、公共事务激励和员工协同机制。建议批准 30 天试点，并明确一个公司侧对接人，避免员工协同小组只有责任没有资源。"
    )


def render_council(data: dict) -> None:
    st.markdown('<div class="section-title">员工协同小组</div>', unsafe_allow_html=True)
    st.caption('把“地下自发”转成公开、透明、有边界的员工事务协同机制。')

    council = data.get("council", DEFAULT_DATA["council"])
    st.markdown(
        f"""
        <div class="glass-card">
            <span class="tag">{council["cycle"]}</span>
            <span class="tag">{council["cadence"]}</span>
            <div class="idea-title">定位：员工与公司之间的协同层、反馈层、共创层</div>
            <p class="muted">小组不替代管理层决策，也不制造对立；它负责把分散意见整理成共性问题，把临时事务转成可分工、可复盘的行动。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### 成员与分工")
    cols = st.columns(2)
    for index, member in enumerate(council["members"]):
        with cols[index % 2]:
            st.markdown(
                f"""
                <div class="idea-card">
                    <span class="tag">{member["role"]}</span>
                    <div class="idea-title">{member["name"]}</div>
                    <p class="muted">{member["scope"]}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    left, right = st.columns([0.95, 1.05])
    with left:
        st.markdown("#### 工作边界")
        for principle in council["principles"]:
            st.markdown(
                f"""
                <div class="slot">
                    <div>{principle}</div>
                    <div class="slot-done">边界</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    with right:
        st.markdown("#### 标准流转")
        flow = [
            ("收集", "员工提交真实反馈、活动想法或协作问题"),
            ("整理", "AI 转译 + 小组合并重复议题"),
            ("反馈", "形成周报和需管理层确认事项"),
            ("协同", "明确负责人、资源、激励和时间节点"),
            ("复盘", "公开处理结果，沉淀下次流程"),
        ]
        for step, body in flow:
            st.markdown(
                f"""
                <div class="slot">
                    <div><strong>{step}</strong><br><span class="muted">{body}</span></div>
                    <div class="slot-open">Pulse</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with st.expander("新增协同小组成员", expanded=False):
        with st.form("council_form", clear_on_submit=True):
            name = st.text_input("成员名称 / 昵称")
            role = st.text_input("角色", placeholder="例如：设计代表 / 新人代表 / 行政对接")
            scope = st.text_area("负责范围", placeholder="描述这个成员主要收集或推进什么问题")
            add_member = st.form_submit_button("加入小组名单")
        if add_member and name.strip() and role.strip():
            data.setdefault("council", json.loads(json.dumps(DEFAULT_DATA["council"])))
            data["council"]["members"].append(
                {
                    "name": name.strip(),
                    "role": role.strip(),
                    "scope": scope.strip() or "待补充负责范围",
                }
            )
            save_data(data)
            st.success("已加入员工协同小组名单。")
            st.rerun()


def render_proposal() -> None:
    st.markdown('<div class="section-title">30 天落地路线</div>', unsafe_allow_html=True)
    phases = [
        ("第 1 周", "小范围调研与分类体系", "收集 10-20 条真实员工反馈，确定议题分类、状态流转和代表机制。"),
        ("第 2 周", "Pulse Hub 试用", "选 5-10 名员工试用提交、投票、事项看板和周报功能。"),
        ("第 3 周", "真实场景接入", "以烧烤活动或一次内部沟通事项作为样板，跑通任务认领和资源确认。"),
        ("第 4 周", "复盘汇报", "输出数据、案例、员工反馈与下一阶段建议，向管理层申请正式试点。"),
    ]
    for phase, title, body in phases:
        st.markdown(
            f"""
            <div class="glass-card">
                <span class="tag">{phase}</span>
                <div class="idea-title">{title}</div>
                <p class="muted">{body}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### 角色设计")
    st.table(
        pd.DataFrame(
            [
                {"角色": "普通员工", "职责": "提交建议、投票、补充信息、查看处理进度"},
                {"角色": "员工协同小组", "职责": "合并共性问题、协助活动组织、整理反馈周报"},
                {"角色": "公司负责人/行政", "职责": "确认资源、指定责任人、反馈处理结果"},
                {"角色": "AI Agent", "职责": "分类摘要、生成周报、把情绪表达转译成建设性建议"},
            ]
        )
    )


def render_pulse_space(data: dict) -> None:
    st.markdown('<div class="section-title">脉冲广场</div>', unsafe_allow_html=True)
    st.caption("先收集真实声音；需要更正式表达时，再展开 AI 转译。")
    with st.expander("AI 情绪转译", expanded=False):
        render_translator(data)
    render_ideas(data)


def render_coordination_space(data: dict) -> None:
    st.markdown('<div class="section-title">推进看板</div>', unsafe_allow_html=True)
    st.caption("把员工反馈变成有人接、有边界、有下一步的协作事项。")
    render_tasks(data)
    with st.expander("员工协同小组机制", expanded=False):
        render_council(data)


def render_briefing_space(data: dict) -> None:
    st.markdown('<div class="section-title">汇报中心</div>', unsafe_allow_html=True)
    st.caption("给上层看的内容集中在这里：风险、决策、简报和试点路线。")
    render_management_dashboard(data)
    with st.expander("AI 周报与提案素材", expanded=False):
        render_report(data)
    with st.expander("30 天试点方案", expanded=False):
        render_proposal()


def render_submit_feedback(data: dict) -> None:
    st.markdown(f'<div class="section-title">{tx("提交反馈", "声を届ける")}</div>', unsafe_allow_html=True)
    st.caption(tx("提交后会进入公开列表，便于集中整理和跟进。", "投稿後は公開リストに表示され、整理と進捗確認に活用されます。"))
    with st.expander(tx("需要 AI 帮你整理表达？", "AIで表現を整えますか？"), expanded=False):
        render_translator(data)
    render_submit_form(data)


def is_admin() -> bool:
    return bool(st.session_state.get("admin_authenticated"))


def render_admin_management(data: dict) -> None:
    if not is_admin() or not data["ideas"]:
        return
    st.markdown(f'<div class="section-title">{tx("管理层处理台", "管理層対応デスク")}</div>', unsafe_allow_html=True)
    st.caption(tx("更新状态、负责人和正式回应；保存后会进入员工可见的时间线。", "ステータス、担当者、正式回答を更新します。保存後は公開タイムラインに反映されます。"))
    idea_options = {f'{idea["title"]} · {idea["status"]}': idea["id"] for idea in data["ideas"]}
    selected_label = st.selectbox(tx("选择反馈", "対象の声"), list(idea_options), key="admin_selected_idea")
    selected_id = idea_options[selected_label]
    idea = next(item for item in data["ideas"] if item["id"] == selected_id)
    statuses = ["待确认", "已受理", "推进中", "已完成", "暂缓"]
    current_index = statuses.index(idea["status"]) if idea["status"] in statuses else 0
    with st.form(f'admin_manage_{idea["id"]}'):
        left, right = st.columns(2)
        new_status = left.selectbox(tx("状态", "ステータス"), statuses, index=current_index, format_func=ui_value)
        new_owner = right.text_input(tx("负责人", "担当者"), value=idea.get("owner", "待确认"))
        response = st.text_area(
            tx("管理层回应", "管理層からの回答"),
            value=idea.get("management_response", ""),
            placeholder=tx("说明是否受理、为什么、接下来由谁处理。", "受付の有無、理由、次の担当者を説明してください。"),
            height=100,
        )
        next_update = st.text_input(
            tx("下次更新时间", "次回更新日"),
            value=idea.get("next_update_at", ""),
            placeholder=tx("例如：2026-07-17 或 本周五", "例：2026-07-17 または 今週金曜日"),
        )
        merge_candidates = {
            "不合并": "",
            **{
                f'{candidate["title"]} · {candidate["category"]}': candidate["id"]
                for candidate in data["ideas"]
                if candidate["id"] != idea["id"] and not candidate.get("merged_into_id")
            },
        }
        current_merge_label = next(
            (label for label, value in merge_candidates.items() if value == idea.get("merged_into_id")),
            "不合并",
        )
        merge_into_label = st.selectbox(
            tx("合并到主议题（可选）", "主要テーマに統合（任意）"),
            list(merge_candidates),
            index=list(merge_candidates).index(current_merge_label),
        )
        save_update = st.form_submit_button(tx("发布回应与状态更新", "回答とステータスを公開"), use_container_width=True)
    if save_update:
        merge_into_id = merge_candidates[merge_into_label]
        if merge_into_id:
            new_status = "已合并"
        if new_status != idea["status"] and not response.strip():
            st.error("状态发生变化时，请同时填写一段管理层回应。")
            return
        old_status = idea["status"]
        idea["status"] = new_status
        idea["owner"] = new_owner.strip() or "待确认"
        idea["management_response"] = response.strip()
        idea["next_update_at"] = next_update.strip()
        idea["merged_into_id"] = merge_into_id
        event = {
            "from_status": old_status,
            "to_status": new_status,
            "response": response.strip(),
            "owner": idea["owner"],
            "actor": get_secret_value("ADMIN_NAME") or "管理层",
            "created_at": now_str(),
        }
        persist_management_update(idea, event, data)
        st.session_state["pending_toast"] = "管理层回应已发布。"
        st.rerun()


def render_feedback_progress(data: dict) -> None:
    st.markdown(f'<div class="section-title">{tx("查看进度", "進捗を見る")}</div>', unsafe_allow_html=True)
    st.caption(tx("这里展示已经提交的反馈和正在推进的事项。", "届けられた声と、現在進んでいる案件を確認できます。"))

    pending_id = st.session_state.get("pending_delete_id", "")
    if pending_id:
        pending_idea = next((i for i in data["ideas"] if i["id"] == pending_id), None)
        if pending_idea:
            st.warning(tx(f"请输入「{pending_idea['title']}」的删除码以确认删除", f"「{pending_idea['title']}」を削除するため、削除コードを入力してください"))
            col1, col2, col3 = st.columns([1, 0.4, 0.4])
            entered = col1.text_input(tx("删除码", "削除コード"), max_chars=4, label_visibility="collapsed", placeholder=tx("输入提交时设置的4位删除码", "投稿時に設定した4桁のコード"))
            if col2.button(tx("确认删除", "削除する"), type="primary"):
                if verify_delete_code(pending_idea, entered.strip()):
                    persist_delete_idea(data, pending_id)
                    liked_idea_ids().discard(pending_id)
                    st.session_state.pop("pending_delete_id", None)
                    st.toast(tx("已删除这条反馈。", "この声を削除しました。"))
                    st.rerun()
                else:
                    st.error(tx("删除码不正确。", "削除コードが正しくありません。"))
            if col3.button(tx("取消", "キャンセル")):
                st.session_state.pop("pending_delete_id", None)
                st.rerun()
            st.divider()

    if not data["ideas"]:
        st.info(tx('还没有反馈。可以先到“提交反馈”写下第一条。', 'まだ声がありません。「声を届ける」から最初の投稿を書いてみましょう。'))
        return
    render_admin_management(data)
    categories = ["全部"] + sorted({i["category"] for i in data["ideas"]})
    selected = st.segmented_control(tx("反馈类型", "カテゴリ"), categories, default="全部", format_func=lambda value: tx("全部", "すべて") if value == "全部" else ui_value(value))
    for idea in data["ideas"]:
        if selected != "全部" and idea["category"] != selected:
            continue
        render_idea_card(idea, show_actions=True, allow_delete=True)
        render_status_timeline(idea)

    with st.expander(tx("事项处理进度", "案件の進捗"), expanded=False):
        render_tasks(data)



def star_position(index: int) -> tuple[int, int]:
    positions = [
        (64, 18), (72, 28), (55, 24), (82, 18), (46, 31),
        (68, 40), (37, 22), (76, 48), (58, 12), (88, 34),
        (49, 45), (62, 56), (34, 37), (79, 60), (91, 22),
    ]
    return positions[index % len(positions)]


CONSTELLATION_NAMES = {
    "沟通协同": "信息流星座",
    "权益激励": "温度星座",
    "流程规范": "秩序星座",
    "文化活动": "共创星座",
    "成长发展": "成长星座",
    "综合建议": "微光星座",
}

CONSTELLATION_NAMES_JA = {
    "沟通协同": "情報の流れ座",
    "权益激励": "ぬくもり座",
    "流程规范": "秩序座",
    "文化活动": "共創座",
    "成长发展": "成長座",
    "综合建议": "微光座",
}


def constellation_name(category: str) -> str:
    if current_language() == "ja":
        return CONSTELLATION_NAMES_JA.get(category, f"{ui_value(category)}座")
    return CONSTELLATION_NAMES.get(category, f"{category}星座")


CONSTELLATION_COLORS = {
    "沟通协同": "#5eead4",
    "权益激励": "#f7c948",
    "流程规范": "#7c8cff",
    "文化活动": "#ff5ea8",
    "成长发展": "#60d394",
    "综合建议": "#cbd5e1",
}


def build_constellations(ideas: list[dict]) -> list[dict]:
    groups: dict[str, list[tuple[int, dict]]] = {}
    for index, idea in enumerate(ideas):
        category = str(idea.get("category") or "综合建议")
        groups.setdefault(category, []).append((index, idea))

    constellations = []
    for category, items in groups.items():
        if len(items) < 2:
            continue
        points = []
        for index, idea in items:
            x, y = star_position(index)
            text = f"{idea.get('title', '')} {idea.get('content', '')} {idea.get('impact', '')} {category}"
            points.append(
                {
                    "index": index,
                    "x": x,
                    "y": y,
                    "tokens": extract_topic_tokens(text),
                    "heat": int(idea.get("heat", 0) or 0),
                }
            )

        connected_pairs = []
        for current, following in zip(points, points[1:]):
            connected_pairs.append((current, following))
        for left_index, left in enumerate(points):
            for right in points[left_index + 1 :]:
                if left["tokens"] & right["tokens"]:
                    pair = (left, right)
                    reverse_pair = (right, left)
                    if pair not in connected_pairs and reverse_pair not in connected_pairs:
                        connected_pairs.append(pair)

        center_x = round(sum(point["x"] for point in points) / len(points), 1)
        center_y = round(sum(point["y"] for point in points) / len(points), 1)
        avg_heat = round(sum(point["heat"] for point in points) / len(points))
        constellations.append(
            {
                "category": category,
                "name": constellation_name(category),
                "color": CONSTELLATION_COLORS.get(category, "#cbd5e1"),
                "count": len(points),
                "avg_heat": avg_heat,
                "x": center_x,
                "y": max(12, center_y - 8),
                "pairs": connected_pairs,
            }
        )
    return constellations


def render_star_page(data: dict) -> None:
    hide_sidebar_for_landing()
    ideas = [idea for idea in data["ideas"] if not idea.get("merged_into_id")]
    hero_uri = image_data_uri(NIGHT_HERO_IMAGE)

    star_items = []
    detail_cards = []
    for index, idea in enumerate(ideas):
        x, y = star_position(index)
        detail_id = f"star-detail-{index}"
        title = html.escape(idea["title"])
        category = html.escape(idea["category"])
        status = html.escape(idea["status"])
        content = html.escape(idea["content"])
        impact = html.escape(idea["impact"])
        created_at = html.escape(idea["created_at"])
        color = status_color(idea["status"])
        star_items.append(
            f'<a class="sp-star" href="#{detail_id}" style="left:{x}%; top:{y}%;" title="{title}">{title}</a>'
        )
        detail_cards.append(
            f"""
            <div class="star-page-detail" id="{detail_id}">
                <span class="sp-tag">{html.escape(ui_value(idea["category"]))}</span>
                <span class="sp-tag" style="border-color:{color}; color:{color};">{html.escape(ui_value(idea["status"]))}</span>
                <span class="sp-tag">{tx("热度", "注目度")} {idea["heat"]}%</span>
                <a class="sp-close" href="#star-field">{tx("关闭", "閉じる")}</a>
                <div class="sp-idea-title">{title}</div>
                <p style="margin:6px 0 4px; color:#d5deed;">{content}</p>
                <p style="margin:0; color:#9aa7bd; font-size:13px;">{tx("希望改进", "期待する改善")}：{impact} · {created_at}</p>
            </div>
            """
        )

    constellations = build_constellations(ideas)
    constellation_categories = {constellation["category"] for constellation in constellations}
    constellation_lines = []
    constellation_labels = []
    constellation_seed_labels = []
    constellation_panel_items = []
    for constellation in constellations:
        color = constellation["color"]
        for left, right in constellation["pairs"]:
            constellation_lines.append(
                f'<line x1="{left["x"]}%" y1="{left["y"]}%" x2="{right["x"]}%" y2="{right["y"]}%" '
                f'stroke="{color}" stroke-width="1.4" stroke-linecap="round" />'
            )
        constellation_labels.append(
            f"""
            <div class="sp-constellation-label" style="left:{constellation["x"]}%; top:{constellation["y"]}%; border-color:{color}; color:{color};">
                {html.escape(constellation["name"])}
            </div>
            """
        )
        constellation_panel_items.append(
            f"""
            <div class="sp-constellation-item">
                <span class="sp-constellation-dot" style="background:{color}; box-shadow:0 0 14px {color};"></span>
                <div>
                    <strong>{html.escape(constellation["name"])}</strong>
                    <span>{tx(f'{constellation["count"]} 颗星 · 平均热度 {constellation["avg_heat"]}%', f'{constellation["count"]}個の星 · 平均注目度 {constellation["avg_heat"]}%')}</span>
                </div>
            </div>
            """
        )

    for index, idea in enumerate(ideas):
        category = str(idea.get("category") or "综合建议")
        if category in constellation_categories:
            continue
        x, y = star_position(index)
        seed_name = constellation_name(category)
        seed_color = CONSTELLATION_COLORS.get(category, "#cbd5e1")
        constellation_seed_labels.append(
            f"""
            <div class="sp-constellation-seed" style="left:{x}%; top:{max(10, y + 6)}%; border-color:{seed_color}; color:{seed_color};">
                {html.escape(seed_name)} · {tx("待连接", "接続待ち")}
            </div>
            """
        )

    constellation_svg = ""
    if constellation_lines:
        constellation_svg = f"""
        <svg class="sp-constellation-lines" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
            <defs>
                <filter id="constellation-glow">
                    <feGaussianBlur stdDeviation="0.55" result="blur" />
                    <feMerge>
                        <feMergeNode in="blur" />
                        <feMergeNode in="SourceGraphic" />
                    </feMerge>
                </filter>
            </defs>
            <g filter="url(#constellation-glow)">
                {''.join(constellation_lines)}
            </g>
        </svg>
        """

    constellation_panel = ""
    if constellations:
        constellation_panel = f"""
        <div class="sp-constellation-panel">
            <div class="sp-panel-title">{tx("意见星座", "声の星座")}</div>
            {''.join(constellation_panel_items)}
        </div>
        """
    elif ideas:
        constellation_panel = f"""
        <div class="sp-constellation-panel">
            <div class="sp-panel-title">{tx("意见星座", "声の星座")}</div>
            <p>{tx("第一颗星已经出现，等待更多同频意见连成星座。", "最初の星が現れました。同じ想いが集まり、星座になるのを待っています。")}</p>
        </div>
        """

    empty_text = "" if ideas else f"<p style='color:#9aa7bd;margin:12px 0 0;'>{tx('还没有反馈，提交第一条后这里会出现第一颗星。', 'まだ声はありません。最初の投稿がここで最初の星になります。')}</p>"

    star_page_html = f"""
        <style>
        html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"], .main {{
            width: 100vw !important;
            height: 100dvh !important;
            min-height: 100dvh !important;
            overflow: hidden !important;
            background: #040914 !important;
        }}
        .block-container {{
            width: 100vw !important;
            height: 100dvh !important;
            min-height: 100dvh !important;
            max-width: none !important;
            padding: 0 !important;
            margin: 0 !important;
            overflow: hidden !important;
            background: #040914 !important;
        }}
        [data-testid="stVerticalBlock"],
        [data-testid="stElementContainer"],
        [data-testid="stHtml"],
        [data-testid="stMarkdownContainer"] {{
            width: 100vw !important;
            max-width: none !important;
            margin: 0 !important;
            padding: 0 !important;
        }}
        .star-shell {{
            position: fixed;
            inset: 0;
            width: 100vw;
            height: 100dvh;
            min-height: 100dvh;
            overflow: hidden;
            background: #040914;
            z-index: 999;
        }}
        [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"], footer, header {{
            display: none !important;
        }}
        .star-page {{
            position: relative;
            width: 100%;
            height: 100%;
            min-height: 100dvh;
            overflow: hidden;
            background-size: cover;
            background-position: center center;
            background-repeat: no-repeat;
            color: #f7fbff;
            font-family: inherit;
        }}
        .star-page::before {{
            content: "";
            position: absolute;
            inset: 0;
            background:
                linear-gradient(90deg, rgba(5,8,18,0.72) 0%, rgba(5,8,18,0.28) 52%, rgba(5,8,18,0.06) 100%),
                linear-gradient(0deg, rgba(5,8,18,0.62) 0%, rgba(5,8,18,0.04) 55%);
            pointer-events: none;
        }}
        .star-page-title {{
            position: absolute;
            left: 40px;
            top: 74px;
            z-index: 2;
            max-width: 460px;
        }}
        .star-page-title h2 {{
            margin: 0 0 10px;
            font-size: 36px;
            font-weight: 900;
            color: #f7fbff;
        }}
        .star-page-title p {{
            margin: 0;
            color: #d5deed;
            font-size: 15px;
        }}
        .sp-back-form {{
            position: absolute;
            top: 28px;
            left: 36px;
            z-index: 30;
            margin: 0;
            width: auto;
            height: auto;
        }}
        .sp-view-tools {{
            position: absolute;
            top: 28px;
            right: 36px;
            z-index: 30;
            display: flex;
            gap: 10px;
            align-items: center;
        }}
        .sp-back-button {{
            appearance: none;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            width: auto;
            height: auto;
            min-width: 0;
            min-height: 0;
            padding: 9px 16px;
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,0.22);
            background: rgba(8,13,26,0.34);
            backdrop-filter: blur(10px);
            color: #f0f6ff;
            font-size: 13px;
            font-weight: 700;
            text-decoration: none;
            line-height: 1;
            white-space: nowrap;
            cursor: pointer;
        }}
        .sp-layer-toggle {{
            display: inline-flex;
            align-items: center;
            min-height: 33px;
            padding: 0 14px;
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,0.22);
            background: rgba(8,13,26,0.34);
            backdrop-filter: blur(10px);
            color: #f0f6ff;
            font-size: 13px;
            font-weight: 700;
            text-decoration: none;
            line-height: 1;
            white-space: nowrap;
        }}
        .sp-hide-notes {{
            display: none;
        }}
        #constellation-notes:target ~ .sp-view-tools .sp-show-notes {{
            display: none;
        }}
        #constellation-notes:target ~ .sp-view-tools .sp-hide-notes {{
            display: inline-flex;
        }}
        .sp-back-button:hover {{
            background: rgba(94,234,212,0.18);
            border-color: rgba(94,234,212,0.5);
            color: #5eead4;
        }}
        .sp-layer-toggle:hover {{
            background: rgba(94,234,212,0.18);
            border-color: rgba(94,234,212,0.5);
            color: #5eead4;
        }}
        .sp-star {{
            position: absolute;
            z-index: 3;
            width: 14px;
            height: 14px;
            border-radius: 999px;
            background: #f8fbff;
            box-shadow: 0 0 10px rgba(255,255,255,0.95), 0 0 24px rgba(94,234,212,0.65);
            border: 1px solid rgba(255,255,255,0.9);
            transform: translate(-50%, -50%);
            overflow: hidden;
            font-size: 0;
            text-indent: -9999px;
        }}
        .sp-star:hover {{
            width: 20px;
            height: 20px;
            background: #5eead4;
        }}
        .sp-constellation-lines {{
            position: absolute;
            inset: 0;
            z-index: 2;
            width: 100%;
            height: 100%;
            opacity: 0.62;
            pointer-events: none;
        }}
        .sp-constellation-label {{
            position: absolute;
            z-index: 4;
            transform: translate(-50%, -50%);
            padding: 4px 10px;
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,0.22);
            background: rgba(5,8,18,0.42);
            backdrop-filter: blur(10px);
            font-size: 12px;
            font-weight: 850;
            white-space: nowrap;
            pointer-events: none;
            text-shadow: 0 0 12px rgba(0,0,0,0.6);
        }}
        .sp-constellation-seed {{
            position: absolute;
            z-index: 4;
            transform: translate(-50%, -50%);
            padding: 3px 9px;
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,0.18);
            background: rgba(5,8,18,0.32);
            backdrop-filter: blur(8px);
            font-size: 11px;
            font-weight: 750;
            white-space: nowrap;
            pointer-events: none;
            opacity: 0.82;
        }}
        .sp-constellation-panel {{
            position: absolute;
            left: 40px;
            bottom: 32px;
            z-index: 8;
            width: min(320px, calc(100vw - 80px));
            border: 1px solid rgba(255,255,255,0.14);
            border-radius: 16px;
            background: rgba(8,13,26,0.52);
            backdrop-filter: blur(16px);
            padding: 14px 16px;
            color: #d5deed;
        }}
        .sp-notes-layer {{
            display: none;
        }}
        #constellation-notes:target ~ .sp-notes-layer {{
            display: block;
        }}
        .sp-anchor {{
            position: absolute;
            inset: 0 auto auto 0;
            width: 1px;
            height: 1px;
            pointer-events: none;
            opacity: 0;
        }}
        .sp-panel-title {{
            color: #f7fbff;
            font-weight: 900;
            margin-bottom: 10px;
        }}
        .sp-constellation-panel p {{
            margin: 0;
            color: #9aa7bd;
            font-size: 13px;
            line-height: 1.55;
        }}
        .sp-constellation-item {{
            display: flex;
            gap: 10px;
            align-items: center;
            margin-top: 9px;
        }}
        .sp-constellation-item strong {{
            display: block;
            color: #f7fbff;
            font-size: 13px;
            line-height: 1.2;
        }}
        .sp-constellation-item span:last-child {{
            display: block;
            color: #9aa7bd;
            font-size: 12px;
            margin-top: 2px;
        }}
        .sp-constellation-dot {{
            width: 9px;
            height: 9px;
            border-radius: 999px;
            flex: 0 0 auto;
        }}
        .star-page-detail {{
            position: absolute;
            bottom: 32px;
            left: 40px;
            right: 40px;
            z-index: 20;
            border: 1px solid rgba(255,255,255,0.16);
            border-radius: 16px;
            background: rgba(8,13,26,0.78);
            backdrop-filter: blur(18px);
            padding: 20px 24px;
            display: none;
        }}
        .star-page-detail:target {{
            display: block;
        }}
        .star-page-detail:target ~ .sp-constellation-panel {{
            display: none;
        }}
        .sp-tag {{
            display: inline-block;
            padding: 3px 10px;
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,0.22);
            color: #d5deed;
            font-size: 12px;
            margin-right: 6px;
            margin-bottom: 10px;
        }}
        .sp-idea-title {{
            font-size: 18px;
            font-weight: 800;
            color: #f7fbff;
            margin: 4px 0 8px;
        }}
        .sp-close {{
            float: right;
            display: inline-flex;
            align-items: center;
            min-height: 28px;
            padding: 0 11px;
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,0.18);
            color: #cbd5e1;
            font-size: 12px;
            font-weight: 750;
            cursor: pointer;
            background: rgba(255,255,255,0.06);
            text-decoration: none;
        }}
        .sp-close:hover {{
            color: #5eead4;
            border-color: rgba(94,234,212,0.42);
        }}
        @media (max-width: 700px) {{
            .star-page-title {{
                left: 22px;
                top: 70px;
                right: 22px;
            }}
            .star-page-title h2 {{
                font-size: 28px;
            }}
            .sp-back-form {{
                left: 22px;
                top: 22px;
            }}
            .sp-view-tools {{
                right: 22px;
                top: 22px;
            }}
            .sp-layer-toggle {{
                font-size: 12px;
                padding: 0 11px;
            }}
            .star-page-detail {{
                left: 22px;
                right: 22px;
                bottom: 22px;
            }}
            .sp-constellation-panel {{
                left: 22px;
                right: 22px;
                bottom: 22px;
                width: auto;
            }}
            .sp-constellation-label {{
                font-size: 11px;
            }}
        }}
        </style>
        <div class="star-shell">
            <div class="star-page" id="star-field" style="background-image: url('{hero_uri}');">
                <span class="sp-anchor" id="constellation-notes"></span>
                <div class="star-page-title">
                    <h2>{tx("意见像星星一样被看见", "声が星のように見える")}</h2>
                    <p>{tx("把分散的想法放在同一片天空里，方便大家查看和跟进。", "ばらばらの想いをひとつの夜空に集め、みんなで見守ります。")}</p>
                    {empty_text}
                </div>
                <form class="sp-back-form" method="get">
                    <input type="hidden" name="view" value="landing">
                    {language_query_field()}
                    <button class="sp-back-button" type="submit">← {tx("返回", "戻る")}</button>
                </form>
                <div class="sp-view-tools">
                    <a class="sp-layer-toggle sp-show-notes" href="#constellation-notes">{tx("显示星座说明", "星座の説明を表示")}</a>
                    <a class="sp-layer-toggle sp-hide-notes" href="#star-field">{tx("隐藏星座说明", "星座の説明を隠す")}</a>
                </div>
                {constellation_svg}
                <div class="sp-notes-layer">
                    {''.join(constellation_labels)}
                    {''.join(constellation_seed_labels)}
                    {constellation_panel}
                </div>
                {''.join(star_items)}
                {''.join(detail_cards)}
            </div>
        </div>
        """
    st.markdown("\n".join(line.strip() for line in star_page_html.splitlines()), unsafe_allow_html=True)


def make_echo_text(idea: dict) -> str:
    category = str(idea.get("category") or "综合建议")
    title = str(idea.get("title") or "这条反馈")
    templates = {
        "沟通协同": "这不是一句抱怨，而是在提醒我们：信息需要更早抵达每一个正在努力的人。",
        "权益激励": "这份声音在说：被照顾到的感受，也会成为继续投入的力量。",
        "流程规范": "这条反馈想让事情少一些临时补救，多一些清楚可循的路径。",
        "文化活动": "这里藏着一个愿望：让活动不只是被安排，而是被大家一起点亮。",
        "成长发展": "这份期待指向更长远的东西：让努力被看见，也让成长有方向。",
        "综合建议": "这是一颗小小的信号，提醒我们把模糊的不舒服变成可以讨论的改进。",
    }
    if current_language() == "ja":
        templates = {
            "沟通协同": "これは不満ではなく、情報をより早く、努力しているすべての人に届けてほしいというサインです。",
            "权益激励": "配慮されているという実感が、次の一歩につながります。",
            "流程规范": "その場しのぎを減らし、誰もが迷わず進める道筋を求める声です。",
            "文化活动": "イベントを一方的に決めるのではなく、みんなでつくりたいという願いです。",
            "成长发展": "努力を見つけ、成長の方向を一緒に描いてほしいという期待です。",
            "综合建议": "小さなサインが、まだ言葉になっていない違和感を改善の対話へ変えてくれます。",
        }
        return templates.get(category, f'「{title[:18]}」に関する声が、丁寧な回答を待っています。')
    return templates.get(category, f'关于"{title[:18]}"的声音，正在等待一次认真回应。')


def render_echo_wall(data: dict) -> None:
    hide_sidebar_for_landing()
    ideas = [idea for idea in data["ideas"] if not idea.get("merged_into_id")]
    hero_uri = image_data_uri(NIGHT_HERO_IMAGE)
    positions = [
        (38, 46), (68, 38), (82, 56), (24, 64), (54, 62),
        (76, 72), (32, 78), (50, 76), (66, 82), (84, 68),
    ]
    echo_cards = []
    for index, idea in enumerate(ideas[:10]):
        x, y = positions[index % len(positions)]
        delay = round((index % 5) * 0.7, 1)
        category = html.escape(ui_value(idea.get("category") or "综合建议"))
        title = html.escape(str(idea.get("title") or "未命名反馈"))
        echo = html.escape(make_echo_text(idea))
        heat = int(idea.get("heat", 0) or 0)
        echo_cards.append(
            f"""
            <div class="echo-card" style="left:{x}%; top:{y}%; animation-delay:{delay}s;">
                <span>{category} · {tx("热度", "注目度")} {heat}%</span>
                <strong>{title}</strong>
                <p>{echo}</p>
            </div>
            """
        )

    empty_text = ""
    if not ideas:
        empty_text = f"""
        <div class="echo-empty">
            {tx("还没有可以回响的反馈。第一条真实声音，会成为这里的第一道回声。", "まだ響き合う声はありません。最初の率直な声が、ここで最初のエコーになります。")}
        </div>
        """

    echo_html = f"""
        <style>
        html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"], .main {{
            width: 100vw !important;
            height: 100dvh !important;
            min-height: 100dvh !important;
            overflow: hidden !important;
            background: #040914 !important;
        }}
        .block-container {{
            width: 100vw !important;
            height: 100dvh !important;
            min-height: 100dvh !important;
            max-width: none !important;
            padding: 0 !important;
            margin: 0 !important;
            overflow: hidden !important;
            background: #040914 !important;
        }}
        [data-testid="stVerticalBlock"],
        [data-testid="stElementContainer"],
        [data-testid="stMarkdownContainer"] {{
            width: 100vw !important;
            max-width: none !important;
            margin: 0 !important;
            padding: 0 !important;
        }}
        [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"], footer, header {{
            display: none !important;
        }}
        .echo-shell {{
            position: fixed;
            inset: 0;
            width: 100vw;
            height: 100dvh;
            overflow: hidden;
            background: #040914;
            z-index: 999;
        }}
        .echo-wall {{
            position: relative;
            width: 100%;
            height: 100%;
            overflow: hidden;
            background-image:
                linear-gradient(90deg, rgba(5,8,18,0.72), rgba(5,8,18,0.18) 58%, rgba(5,8,18,0.06)),
                url('{hero_uri}');
            background-size: cover;
            background-position: center center;
            color: #f7fbff;
        }}
        .echo-wall::after {{
            content: "";
            position: absolute;
            inset: 0;
            background: radial-gradient(circle at 50% 44%, rgba(94,234,212,0.12), transparent 32%);
            pointer-events: none;
        }}
        .echo-back-form {{
            position: absolute;
            top: 28px;
            left: 36px;
            z-index: 30;
            margin: 0;
        }}
        .echo-back-button,
        .echo-progress-link {{
            appearance: none;
            display: inline-flex;
            align-items: center;
            min-height: 33px;
            padding: 0 14px;
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,0.22);
            background: rgba(8,13,26,0.34);
            color: #f0f6ff;
            font: inherit;
            font-size: 13px;
            font-weight: 700;
            text-decoration: none;
            backdrop-filter: blur(10px);
            cursor: pointer;
        }}
        .echo-progress-link {{
            position: absolute;
            top: 28px;
            right: 36px;
            z-index: 30;
        }}
        .echo-back-button:hover,
        .echo-progress-link:hover {{
            background: rgba(94,234,212,0.18);
            border-color: rgba(94,234,212,0.5);
            color: #5eead4;
        }}
        .echo-title {{
            position: absolute;
            left: 40px;
            top: 78px;
            z-index: 8;
            max-width: 520px;
        }}
        .echo-title h2 {{
            margin: 0 0 10px;
            font-size: 42px;
            line-height: 1.05;
            color: #f7fbff;
        }}
        .echo-title p {{
            margin: 0;
            color: #d5deed;
            line-height: 1.7;
            font-size: 15px;
        }}
        .echo-card {{
            position: absolute;
            z-index: 5;
            width: min(310px, 34vw);
            min-height: 118px;
            transform: translate(-50%, -50%);
            border: 1px solid rgba(255,255,255,0.16);
            border-radius: 16px;
            background: rgba(8,13,26,0.50);
            backdrop-filter: blur(16px);
            box-shadow: 0 22px 70px rgba(0,0,0,0.28);
            padding: 16px 18px;
            animation: echo-float 7s ease-in-out infinite;
        }}
        .echo-card span {{
            display: inline-flex;
            color: #5eead4;
            font-size: 12px;
            font-weight: 800;
            margin-bottom: 8px;
        }}
        .echo-card strong {{
            display: block;
            color: #f7fbff;
            font-size: 16px;
            margin-bottom: 8px;
            line-height: 1.35;
        }}
        .echo-card p {{
            margin: 0;
            color: #d5deed;
            font-size: 13px;
            line-height: 1.65;
        }}
        .echo-empty {{
            position: absolute;
            left: 40px;
            bottom: 42px;
            z-index: 5;
            max-width: 420px;
            border: 1px solid rgba(255,255,255,0.14);
            border-radius: 16px;
            background: rgba(8,13,26,0.54);
            color: #d5deed;
            padding: 18px 20px;
            backdrop-filter: blur(16px);
        }}
        @keyframes echo-float {{
            0%, 100% {{ transform: translate(-50%, -50%) translateY(0); }}
            50% {{ transform: translate(-50%, -50%) translateY(-12px); }}
        }}
        @media (max-width: 760px) {{
            .echo-title {{
                left: 22px;
                right: 22px;
                top: 74px;
            }}
            .echo-title h2 {{
                font-size: 32px;
            }}
            .echo-card {{
                width: min(300px, 78vw);
            }}
            .echo-card:nth-of-type(n+5) {{
                display: none;
            }}
            .echo-back-form {{
                left: 22px;
                top: 22px;
            }}
            .echo-progress-link {{
                right: 22px;
                top: 22px;
            }}
        }}
        </style>
        <div class="echo-shell">
            <div class="echo-wall">
                <form class="echo-back-form" method="get">
                    <input type="hidden" name="view" value="landing">
                    {language_query_field()}
                    <button class="echo-back-button" type="submit">← {tx("返回", "戻る")}</button>
                </form>
                <form method="get" style="margin:0;">
                    <input type="hidden" name="view" value="workspace">
                    <input type="hidden" name="page" value="progress">
                    {language_query_field()}
                    <button class="echo-progress-link" type="submit">{tx("查看进度", "進捗を見る")}</button>
                </form>
                <div class="echo-title">
                    <h2>{tx("回声墙", "エコーウォール")}</h2>
                    <p>{tx("每一条反馈都会留下一个更柔和的回响。这里不展示抱怨，而展示那些值得被认真听见的提醒。", "ひとつひとつの声が、やわらかな響きとして残ります。ここにあるのは不満ではなく、丁寧に耳を傾けたい大切な気づきです。")}</p>
                </div>
                {''.join(echo_cards)}
                {empty_text}
            </div>
        </div>
    """
    st.markdown("\n".join(line.strip() for line in echo_html.splitlines()), unsafe_allow_html=True)


def postcard_summary_line(data: dict) -> str:
    ideas = data["ideas"]
    if not ideas:
        return tx("本周还没有新的员工反馈，建议先用一个轻量入口收集第一批真实声音。", "今週はまだ新しい声がありません。まずは気軽な入口から、最初の率直な声を集めましょう。")
    top_category, count = Counter(idea["category"] for idea in ideas).most_common(1)[0]
    avg_heat = round(sum(idea["heat"] for idea in ideas) / max(len(ideas), 1))
    return tx(
        f"本期共收到 {len(ideas)} 条反馈，最集中的议题是「{top_category}」{count} 条，平均热度 {avg_heat}%。",
        f"今期は{len(ideas)}件の声が届きました。最も多いテーマは「{ui_value(top_category)}」の{count}件、平均注目度は{avg_heat}%です。",
    )


def render_management_postcard(data: dict) -> None:
    hide_sidebar_for_landing()
    ideas = [idea for idea in data["ideas"] if not idea.get("merged_into_id")]
    tasks = data["tasks"]
    hero_uri = image_data_uri(HERO_IMAGE)
    hot_ideas = sorted(ideas, key=lambda item: item["heat"], reverse=True)[:3]
    constellations = sorted(build_constellations(ideas), key=lambda item: item["count"], reverse=True)
    main_constellation = constellations[0] if constellations else None
    open_tasks = [task for task in tasks if task["status"] in {"待确认", "已受理", "推进中"}]
    completed = sum(1 for task in tasks if task["status"] == "已完成")
    response_idea = hot_ideas[0] if hot_ideas else None
    response_line = make_echo_text(response_idea) if response_idea else tx("请给员工一个能被看见、能被回应的固定入口。", "従業員の声が見える形で届き、回答を得られる定着した入口をつくりましょう。")

    hot_items = []
    for idea in hot_ideas:
        hot_items.append(
            f"""
            <div class="pc-hot-item">
                <span>{html.escape(ui_value(idea.get("category", "综合建议")))} · {tx("热度", "注目度")} {int(idea.get("heat", 0) or 0)}%</span>
                <strong>{html.escape(str(idea.get("title", "未命名反馈")))}</strong>
            </div>
            """
        )
    if not hot_items:
        hot_items.append(
            f"""
            <div class="pc-hot-item">
                <span>{tx("等待第一条反馈", "最初の声を待っています")}</span>
                <strong>{tx("先让真实声音有一个落点。", "率直な声が届く場所を、ここから。")}</strong>
            </div>
            """
        )

    constellation_name = tx("尚未形成星座", "まだ星座はありません")
    constellation_detail = tx("当同类反馈达到 2 条以上，会自动连成共性议题。", "同じ種類の声が2件以上集まると、共通テーマの星座になります。")
    constellation_color = "#cbd5e1"
    if main_constellation:
        constellation_name = html.escape(str(main_constellation["name"]))
        constellation_detail = tx(f'{main_constellation["count"]} 颗星 · 平均热度 {main_constellation["avg_heat"]}%', f'{main_constellation["count"]}個の星 · 平均注目度 {main_constellation["avg_heat"]}%')
        constellation_color = str(main_constellation["color"])

    summary = html.escape(postcard_summary_line(data))
    response_line = html.escape(response_line)
    date_text = datetime.now().strftime("%Y.%m.%d")

    postcard_html = f"""
        <style>
        html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"], .main {{
            width: 100vw !important;
            height: 100dvh !important;
            min-height: 100dvh !important;
            overflow: hidden !important;
            background: #07101a !important;
        }}
        .block-container {{
            width: 100vw !important;
            height: 100dvh !important;
            min-height: 100dvh !important;
            max-width: none !important;
            padding: 0 !important;
            margin: 0 !important;
            overflow: hidden !important;
            background: #07101a !important;
        }}
        [data-testid="stVerticalBlock"],
        [data-testid="stElementContainer"],
        [data-testid="stMarkdownContainer"] {{
            width: 100vw !important;
            max-width: none !important;
            margin: 0 !important;
            padding: 0 !important;
        }}
        [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"], footer, header {{
            display: none !important;
        }}
        .pc-shell {{
            position: fixed;
            inset: 0;
            width: 100vw;
            height: 100dvh;
            overflow: hidden;
            z-index: 999;
            background:
                linear-gradient(90deg, rgba(5,8,18,0.72), rgba(5,8,18,0.18) 58%, rgba(5,8,18,0.04)),
                url('{hero_uri}');
            background-size: cover;
            background-position: center center;
            color: #172033;
        }}
        .pc-back-form {{
            position: absolute;
            top: 28px;
            left: 36px;
            z-index: 30;
            margin: 0;
        }}
        .pc-back-button,
        .pc-progress-link {{
            appearance: none;
            display: inline-flex;
            align-items: center;
            min-height: 33px;
            padding: 0 14px;
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,0.22);
            background: rgba(8,13,26,0.34);
            color: #f0f6ff;
            font: inherit;
            font-size: 13px;
            font-weight: 700;
            text-decoration: none;
            backdrop-filter: blur(10px);
            cursor: pointer;
        }}
        .pc-progress-link {{
            position: absolute;
            top: 28px;
            right: 36px;
            z-index: 30;
        }}
        .pc-back-button:hover,
        .pc-progress-link:hover {{
            background: rgba(94,234,212,0.18);
            border-color: rgba(94,234,212,0.5);
            color: #5eead4;
        }}
        .pc-card {{
            position: absolute;
            left: 50%;
            top: 53%;
            transform: translate(-50%, -50%) rotate(-1.2deg);
            width: min(1040px, calc(100vw - 96px));
            height: min(680px, calc(100dvh - 92px));
            border-radius: 8px;
            background:
                linear-gradient(135deg, rgba(255,255,255,0.96), rgba(232,241,247,0.92)),
                #f7fbff;
            box-shadow: 0 34px 110px rgba(0,0,0,0.44);
            overflow: hidden;
            display: grid;
            grid-template-columns: 0.92fr 1.08fr;
        }}
        .pc-photo {{
            position: relative;
            min-height: 100%;
            background:
                linear-gradient(0deg, rgba(5,8,18,0.62), rgba(5,8,18,0.12)),
                url('{hero_uri}');
            background-size: cover;
            background-position: center center;
            color: #f7fbff;
            padding: 28px;
            display: flex;
            flex-direction: column;
            justify-content: flex-end;
        }}
        .pc-photo::before {{
            content: "";
            position: absolute;
            inset: 18px;
            border: 1px solid rgba(255,255,255,0.25);
            pointer-events: none;
        }}
        .pc-photo h2 {{
            position: relative;
            margin: 0 0 10px;
            font-size: 34px;
            line-height: 1.05;
            color: #f7fbff;
        }}
        .pc-photo p {{
            position: relative;
            margin: 0;
            color: #dbeafe;
            line-height: 1.65;
            font-size: 13px;
        }}
        .pc-content {{
            position: relative;
            padding: 24px 30px 24px;
            background-image:
                linear-gradient(rgba(15,23,42,0.055) 1px, transparent 1px),
                linear-gradient(90deg, rgba(15,23,42,0.04) 1px, transparent 1px);
            background-size: 28px 28px;
        }}
        .pc-stamp {{
            position: absolute;
            right: 34px;
            top: 24px;
            width: 92px;
            height: 72px;
            border: 2px dashed rgba(15,23,42,0.35);
            border-radius: 8px;
            display: grid;
            place-items: center;
            color: #334155;
            font-weight: 900;
            font-size: 13px;
            text-align: center;
            transform: rotate(3deg);
        }}
        .pc-kicker {{
            color: #64748b;
            font-size: 13px;
            font-weight: 900;
            text-transform: uppercase;
            letter-spacing: 0.08em !important;
            margin-bottom: 8px;
        }}
        .pc-content h1 {{
            color: #0f172a;
            margin: 0 108px 12px 0;
            font-size: 31px;
            line-height: 1.08;
        }}
        .pc-summary {{
            color: #334155;
            font-size: 13px;
            line-height: 1.62;
            margin: 0 0 12px;
            max-width: 620px;
        }}
        .pc-metrics {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 8px;
            margin: 12px 0;
        }}
        .pc-metric {{
            border: 1px solid rgba(15,23,42,0.12);
            border-radius: 8px;
            background: rgba(255,255,255,0.58);
            padding: 9px 10px;
        }}
        .pc-metric span {{
            display: block;
            color: #64748b;
            font-size: 12px;
            margin-bottom: 4px;
        }}
        .pc-metric strong {{
            color: #0f172a;
            font-size: 22px;
        }}
        .pc-section-title {{
            color: #0f172a;
            font-size: 15px;
            font-weight: 900;
            margin: 12px 0 7px;
        }}
        .pc-hot-list {{
            display: grid;
            gap: 6px;
        }}
        .pc-hot-item {{
            border-left: 3px solid #5eead4;
            padding: 7px 10px;
            background: rgba(15,23,42,0.045);
            border-radius: 0 8px 8px 0;
        }}
        .pc-hot-item span {{
            display: block;
            color: #64748b;
            font-size: 12px;
            margin-bottom: 2px;
        }}
        .pc-hot-item strong {{
            color: #172033;
            font-size: 13px;
            line-height: 1.35;
        }}
        .pc-constellation {{
            display: flex;
            align-items: center;
            gap: 12px;
            border: 1px solid rgba(15,23,42,0.12);
            background: rgba(255,255,255,0.62);
            border-radius: 8px;
            padding: 10px 12px;
        }}
        .pc-constellation-dot {{
            width: 14px;
            height: 14px;
            border-radius: 999px;
            background: {constellation_color};
            box-shadow: 0 0 16px {constellation_color};
            flex: 0 0 auto;
        }}
        .pc-constellation strong {{
            display: block;
            color: #0f172a;
            font-size: 15px;
        }}
        .pc-constellation span {{
            display: block;
            color: #64748b;
            font-size: 12px;
            margin-top: 2px;
        }}
        .pc-response {{
            margin-top: 10px;
            padding: 10px 12px;
            border-radius: 8px;
            background: rgba(94,234,212,0.13);
            color: #0f172a;
            line-height: 1.55;
            font-weight: 750;
            font-size: 13px;
        }}
        .pc-bottom-grid {{
            display: grid;
            grid-template-columns: 0.9fr 1.1fr;
            gap: 10px;
            align-items: stretch;
            margin-top: 12px;
        }}
        .pc-bottom-grid .pc-section-title {{
            margin-top: 0;
        }}
        .pc-bottom-grid .pc-response {{
            margin-top: 0;
            height: calc(100% - 30px);
        }}
        .pc-signoff {{
            position: absolute;
            right: 38px;
            bottom: 18px;
            color: #64748b;
            font-size: 12px;
            text-align: right;
        }}
        @media (max-width: 820px) {{
            .pc-card {{
                top: 54%;
                width: calc(100vw - 36px);
                min-height: calc(100dvh - 110px);
                grid-template-columns: 1fr;
                overflow: auto;
                transform: translate(-50%, -50%);
            }}
            .pc-photo {{
                min-height: 210px;
            }}
            .pc-content {{
                padding: 24px;
            }}
            .pc-content h1 {{
                margin-right: 0;
                font-size: 30px;
            }}
            .pc-stamp {{
                display: none;
            }}
            .pc-metrics {{
                grid-template-columns: 1fr;
            }}
            .pc-signoff {{
                position: static;
                margin-top: 18px;
                text-align: left;
            }}
        }}
        </style>
        <div class="pc-shell">
            <form class="pc-back-form" method="get">
                <input type="hidden" name="view" value="landing">
                {language_query_field()}
                <button class="pc-back-button" type="submit">← {tx("返回", "戻る")}</button>
            </form>
            <form method="get" style="margin:0;">
                <input type="hidden" name="view" value="workspace">
                <input type="hidden" name="page" value="progress">
                {language_query_field()}
                <button class="pc-progress-link" type="submit">{tx("查看进度", "進捗を見る")}</button>
            </form>
            <section class="pc-card">
                <div class="pc-photo">
                    <h2>From Fuji</h2>
                    <p>{tx("把员工反馈整理成一张能被快速阅读、截图转发、用于决策沟通的明信片。", "従業員の声を、すぐに読めて共有でき、意思決定の対話に使える一枚のポストカードへ。")}</p>
                </div>
                <div class="pc-content">
                    <div class="pc-stamp">STELLAR<br>{date_text}</div>
                    <div class="pc-kicker">To Management</div>
                    <h1>{tx("本周员工声音明信片", "今週の従業員ボイス・ポストカード")}</h1>
                    <p class="pc-summary">{summary}</p>
                    <div class="pc-metrics">
                        <div class="pc-metric"><span>{tx("反馈总数", "声の合計")}</span><strong>{len(ideas)}</strong></div>
                        <div class="pc-metric"><span>{tx("开放事项", "対応中")}</span><strong>{len(open_tasks)}</strong></div>
                        <div class="pc-metric"><span>{tx("已完成", "完了")}</span><strong>{completed}</strong></div>
                    </div>
                    <div class="pc-section-title">{tx("最亮的 3 颗星", "最も輝く3つの星")}</div>
                    <div class="pc-hot-list">{''.join(hot_items)}</div>
                    <div class="pc-bottom-grid">
                        <div>
                            <div class="pc-section-title">{tx("最大星座", "最大の星座")}</div>
                            <div class="pc-constellation">
                                <span class="pc-constellation-dot"></span>
                                <div><strong>{constellation_name}</strong><span>{constellation_detail}</span></div>
                            </div>
                        </div>
                        <div>
                            <div class="pc-section-title">{tx("最需要回应的一句话", "今、最も回答が必要な一言")}</div>
                            <div class="pc-response">{response_line}</div>
                        </div>
                    </div>
                    <div class="pc-signoff">Stellar · {tx("员工反馈与协同", "従業員の声と連携")}<br>{now_str()}</div>
                </div>
            </section>
        </div>
    """
    st.markdown("\n".join(line.strip() for line in postcard_html.splitlines()), unsafe_allow_html=True)


def render_settings_panel() -> None:
    st.markdown("**AI 设置**")
    deepseek_key, deepseek_model = get_deepseek_config()
    gemini_key, gemini_model = get_gemini_config()
    supabase_url, supabase_key = get_supabase_config()
    st.caption(f"DeepSeek：{'已配置' if deepseek_key else '未配置'}")
    st.caption(f"DeepSeek 默认模型：{deepseek_model}")
    st.caption(f"Gemini：{'已配置' if gemini_key else '未配置'}")
    st.caption(f"Gemini 默认模型：{gemini_model}")
    st.markdown("**数据存储**")
    st.caption(f"Supabase：{'已配置' if supabase_url and supabase_key else '未配置，当前使用本地 JSON'}")


def render_admin_login() -> None:
    admin_password = get_secret_value("ADMIN_PASSWORD")
    with st.expander(tx("管理入口", "管理者メニュー"), expanded=is_admin()):
        if is_admin():
            st.success(tx("管理员模式已开启", "管理者モードです"))
            if st.button(tx("退出管理员模式", "管理者モードを終了"), use_container_width=True):
                st.session_state.pop("admin_authenticated", None)
                st.rerun()
            return
        if not admin_password:
            st.caption(tx("尚未配置 ADMIN_PASSWORD。", "ADMIN_PASSWORD が設定されていません。"))
            return
        with st.form("admin_login_form"):
            entered = st.text_input(tx("管理员密码", "管理者パスワード"), type="password")
            login = st.form_submit_button(tx("进入管理模式", "管理者モードへ"), use_container_width=True)
        if login:
            if hmac.compare_digest(entered, admin_password):
                st.session_state["admin_authenticated"] = True
                st.rerun()
            else:
                st.error(tx("管理员密码不正确。", "管理者パスワードが正しくありません。"))


def sidebar(data: dict) -> None:
    with st.sidebar:
        st.markdown("## Stellar")
        st.caption(tx("反馈收集 · 进度公开", "声の収集 · 進捗の公開"))
        if st.button(tx("返回入口页", "トップへ戻る"), use_container_width=True):
            st.session_state["view"] = "landing"
            st.query_params.clear()
            st.query_params["lang"] = current_language()
            st.rerun()
        st.divider()
        st.metric(tx("想法总数", "声の合計"), len(data["ideas"]))
        st.metric(tx("事项总数", "案件の合計"), len(data["tasks"]))
        st.divider()
        st.markdown(f'**{tx("共鸣空间", "共感スペース")}**')
        if st.button(tx("星空意见图", "星空ボイスマップ"), key="sidebar_stars", use_container_width=True):
            st.session_state["view"] = "stars"
            st.query_params.clear()
            st.query_params["view"] = "stars"
            st.query_params["lang"] = current_language()
            st.rerun()
        if st.button(tx("回声墙", "エコーウォール"), key="sidebar_echoes", use_container_width=True):
            st.session_state["view"] = "echoes"
            st.query_params.clear()
            st.query_params["view"] = "echoes"
            st.query_params["lang"] = current_language()
            st.rerun()
        st.divider()
        render_admin_login()
        if is_admin():
            if st.button(tx("管理层明信片", "管理層へのポストカード"), key="sidebar_postcard", use_container_width=True):
                st.session_state["view"] = "postcard"
                st.query_params.clear()
                st.query_params["view"] = "postcard"
                st.query_params["lang"] = current_language()
                st.rerun()
            st.divider()
            st.download_button(
                "下载数据备份",
                data=json.dumps(data, ensure_ascii=False, indent=2),
                file_name=f"stellar_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True,
            )
            uploaded_backup = st.file_uploader("恢复数据备份", type=["json"], label_visibility="collapsed")
            if uploaded_backup is not None:
                try:
                    restored = normalize_data(json.loads(uploaded_backup.getvalue().decode("utf-8")))
                    save_data(restored)
                    st.success("数据已恢复。")
                    st.rerun()
                except Exception as exc:
                    st.error(f"恢复失败：{exc}")
            st.divider()
            render_settings_panel()
        st.divider()
        st.caption("语言 / Language")
        language_label = "日本語" if current_language() == "ja" else "中文"
        st.session_state.setdefault("language_switch", language_label)
        selected_language = st.segmented_control(
            "Language",
            list(LANGUAGE_OPTIONS),
            key="language_switch",
            label_visibility="collapsed",
        )
        if LANGUAGE_OPTIONS[selected_language] != current_language():
            sync_language_from_widget("language_switch")
            st.rerun()


def handle_feedback_actions(data: dict) -> None:
    action = st.query_params.get("op", "") or st.query_params.get("action", "")
    idea_id = st.query_params.get("idea", "")
    if action not in {"like", "delete"} or not idea_id:
        return

    idea = next((item for item in data["ideas"] if item["id"] == idea_id), None)
    if not idea:
        st.query_params.clear()
        st.query_params["view"] = "workspace"
        st.query_params["page"] = "progress"
        st.rerun()

    if action == "like":
        token = get_session_token()
        if persist_vote(idea, token, data):
            liked_idea_ids().add(idea_id)
            st.toast("已赞同，这条反馈的热度已更新。")
        else:
            liked_idea_ids().add(idea_id)
            st.toast("你已经赞同过这条反馈了。")

    if action == "delete":
        if idea.get("delete_code_hash") or idea.get("delete_code"):
            st.session_state["pending_delete_id"] = idea_id
            st.query_params.clear()
            st.query_params["view"] = "workspace"
            st.query_params["page"] = "progress"
            st.rerun()
        else:
            st.toast("该反馈未设置删除码，无法删除。")

    st.query_params.clear()
    st.query_params["view"] = "workspace"
    st.query_params["page"] = "progress"
    st.rerun()


def main() -> None:
    st.set_page_config(
        page_title=f"{APP_TITLE} · 反馈与跟进",
        page_icon="✨",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    query_language = st.query_params.get("lang", "")
    if query_language in LANGUAGE_OPTIONS.values():
        st.session_state["language"] = query_language
        query_label = next(label for label, code in LANGUAGE_OPTIONS.items() if code == query_language)
        for widget_key in ("language_switch",):
            if widget_key in st.session_state:
                st.session_state[widget_key] = query_label
    elif "language" not in st.session_state:
        st.session_state["language"] = "zh"
    inject_css()
    data = load_data()
    token = get_session_token()
    for idea in data.get("ideas", []):
        if token in (idea.get("voters") or []):
            liked_idea_ids().add(str(idea["id"]))
    handle_feedback_actions(data)
    pending_toast = st.session_state.pop("pending_toast", "")
    if pending_toast:
        st.toast(pending_toast)
    if "view" not in st.session_state:
        st.session_state["view"] = "landing"
    qv = st.query_params.get("view")
    if qv == "workspace":
        st.session_state["view"] = "workspace"
    elif qv == "stars":
        st.session_state["view"] = "stars"
    elif qv == "echoes":
        st.session_state["view"] = "echoes"
    elif qv == "postcard":
        st.session_state["view"] = "postcard"
    elif qv == "landing":
        st.session_state["view"] = "landing"
        st.query_params.clear()
        st.query_params["lang"] = current_language()

    if st.session_state["view"] == "landing":
        render_landing(data)
        return

    if st.session_state["view"] == "stars" or st.query_params.get("page") == "stars":
        render_star_page(data)
        return

    if st.session_state["view"] == "echoes" or st.query_params.get("page") == "echoes":
        render_echo_wall(data)
        return

    if st.session_state["view"] == "postcard" or st.query_params.get("page") == "postcard":
        if is_admin():
            render_management_postcard(data)
            return
        st.session_state["view"] = "workspace"
        st.query_params.clear()
        st.query_params["view"] = "workspace"
        st.query_params["page"] = "progress"
        st.session_state["pending_toast"] = tx("请先从侧边栏进入管理员模式。", "先にサイドバーから管理者モードに入ってください。")
        st.rerun()

    sidebar(data)
    page_param = st.query_params.get("page")
    default_page = {"progress": "progress"}.get(page_param, "submit")
    page = st.segmented_control(
        tx("页面", "ページ"),
        ["submit", "progress"],
        default=default_page,
        format_func=lambda value: tx("提交反馈", "声を届ける") if value == "submit" else tx("查看进度", "進捗を見る"),
    )
    render_hero(data)
    if page == "submit":
        render_submit_feedback(data)
    else:
        render_feedback_progress(data)


if __name__ == "__main__":
    main()
