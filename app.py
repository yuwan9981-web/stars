from __future__ import annotations

import base64
import json
import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd
import requests
import streamlit as st


APP_TITLE = "群星"
DATA_FILE = Path("pulse_data.json")
HERO_IMAGE = Path("assets/fuji-hero.png")
NIGHT_HERO_IMAGE = Path("assets/fuji-night-stars.png")
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_FALLBACK_MODELS = ["gemini-2.0-flash"]
LEGACY_GEMINI_MODELS = {"gemini-3.5-flash"}
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"
DEEPSEEK_MODELS = ["deepseek-v4-flash", "deepseek-v4-pro", "deepseek-chat", "deepseek-reasoner"]
GEMINI_MODELS = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-3-flash-preview"]


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


def load_data() -> dict:
    if not DATA_FILE.exists():
        save_data(DEFAULT_DATA)
        return json.loads(json.dumps(DEFAULT_DATA))
    with DATA_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "council" not in data:
        data["council"] = json.loads(json.dumps(DEFAULT_DATA["council"]))
        save_data(data)
    return data


def save_data(data: dict) -> None:
    with DATA_FILE.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


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
        f"再围绕“{compact[:42]}”形成执行动作，并在固定周期内反馈处理进展。"
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

    prompt = f"""
把员工反馈转成公司内部可执行建议。只输出 JSON，不要 Markdown。
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

    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=25000))
    try:
        response = client.models.generate_content(model=model, contents=prompt)
        output_text = getattr(response, "text", "") or str(response)
    finally:
        client.close()

    parsed = extract_json_object(output_text)
    category = parsed.get("category") or "综合建议"
    priority = parsed.get("priority") or "中"
    heat = int(parsed.get("heat") or 64)
    return {
        "category": category,
        "priority": priority,
        "heat": max(0, min(100, heat)),
        "tone": parsed.get("tone") or "正式、克制、聚焦行动",
        "title": parsed.get("title") or f"{category}优化建议",
        "translated": parsed.get("translated") or polish_text(text),
        "next_step": parsed.get("next_step") or "先由员工协同小组整理样本，再提交管理层确认负责人和反馈周期。",
        "source": "Gemini API",
    }


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
    return f"""
把员工反馈转成公司内部可执行建议。只输出 JSON，不要 Markdown。
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
        "tone": parsed.get("tone") or "正式、克制、聚焦行动",
        "title": parsed.get("title") or f"{category}优化建议",
        "translated": parsed.get("translated") or polish_text(text),
        "next_step": parsed.get("next_step") or "先整理共性样本，再确认负责人和反馈周期。",
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
    translated = (
        f"当前围绕“{topic_hint}”的反馈，反映出公司在{category}方面存在可优化空间。"
        f"建议将其作为{priority}优先级事项处理：{core_action}"
        "同时建议在处理过程中同步进展和结果，避免问题长期停留在口头沟通层面。"
    )
    return {
        "category": category,
        "priority": priority,
        "heat": heat,
        "tone": tone_map[target],
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
            min-height: 100vh;
            margin: -1.2rem calc(50% - 50vw) 0;
        }

        .landing-hero {
            position: relative;
            min-height: 100vh;
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
                min-height: 100vh;
                background-position: 58% center;
            }
            .landing-content {
                bottom: 7vh;
                padding: 0 20px;
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
        .block-container {
            max-width: 1440px;
            padding-top: 1.2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_landing(data: dict) -> None:
    hide_sidebar_for_landing()
    hero_uri = image_data_uri(HERO_IMAGE)

    st.markdown(
        f"""
        <div class="landing-shell">
            <section class="landing-hero" style="background-image: url('{hero_uri}');">
                <div class="landing-content">
                    <div class="landing-title">Stellar</div>
                    <div class="landing-copy">
                        让每一个想法被看见，让每一次反馈有回声。
                    </div>
                    <div class="landing-actions">
                        <form method="get">
                            <input type="hidden" name="view" value="workspace">
                            <input type="hidden" name="page" value="submit">
                            <button type="submit" class="landing-cta">提交反馈</button>
                        </form>
                        <form method="get">
                            <input type="hidden" name="view" value="workspace">
                            <input type="hidden" name="page" value="progress">
                            <button type="submit" class="landing-ghost">查看进度</button>
                        </form>
                        <form method="get">
                            <input type="hidden" name="view" value="stars">
                            <button type="submit" class="landing-ghost">星空意见图</button>
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
            <p>统一提交反馈，公开查看进度。让问题有人看见，也有机会被跟进。</p>
        </div>
        <div class="metric-grid">
            <div class="metric-card"><div class="label">反馈数量</div><div class="value">{len(ideas)}</div><div class="hint">已提交的员工反馈</div></div>
            <div class="metric-card"><div class="label">处理中事项</div><div class="value">{open_tasks}</div><div class="hint">待确认、已受理或推进中</div></div>
            <div class="metric-card"><div class="label">平均热度</div><div class="value">{avg_heat}%</div><div class="hint">根据投票与影响范围估算</div></div>
            <div class="metric-card"><div class="label">反馈类型</div><div class="value">{categories}</div><div class="hint">自动分类统计</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if completed:
        st.toast(f"已有 {completed} 个事项完成闭环")


def render_idea_card(idea: dict) -> None:
    color = status_color(idea["status"])
    st.markdown(
        f"""
        <div class="idea-card">
            <div class="idea-head">
                <div>
                    <div class="idea-title">{idea["title"]}</div>
                    <span class="tag">{idea["category"]}</span>
                    <span class="tag" style="border-color:{color}; color:{color};">{idea["status"]}</span>
                    <span class="tag">来自：{idea["author"]}</span>
                </div>
                <div class="heat">{idea["heat"]}%<br><span style="font-size:11px;color:#dbeafe;">热度</span></div>
            </div>
            <p>{idea["content"]}</p>
            <p class="muted">预期影响：{idea["impact"]}</p>
            <span class="tag">赞同 {idea["votes"]}</span>
            <span class="tag">{idea["created_at"]}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_submit_form(data: dict) -> None:
    st.markdown('<div class="section-title">填写反馈</div>', unsafe_allow_html=True)
    with st.form("idea_form", clear_on_submit=True):
        col1, col2 = st.columns([1.1, 0.9])
        with col1:
            title = st.text_input("标题", placeholder="例如：希望建立固定的信息同步机制")
            content = st.text_area("反馈内容", placeholder="描述发生了什么、影响了谁、为什么值得处理")
            impact = st.text_area("希望如何改进", placeholder="写下你期待的处理方式或建议")
        with col2:
            author = st.text_input("署名", placeholder="可填写昵称")
            anonymous = st.toggle("匿名提交", value=True)
            submitted = st.form_submit_button("提交反馈")

    if submitted:
        if not title.strip() or not content.strip():
            st.warning("标题和问题描述需要先写一下。")
            return
        category, _priority, heat = classify_text(f"{title} {content} {impact}")
        data["ideas"].insert(
            0,
            {
                "id": f"idea-{uuid4().hex[:8]}",
                "title": title.strip(),
                "category": category,
                "author": "匿名" if anonymous else (author.strip() or "未署名同事"),
                "anonymous": anonymous,
                "content": content.strip(),
                "impact": impact.strip() or polish_text(content),
                "status": "待确认",
                "heat": heat,
                "votes": 1,
                "created_at": now_str(),
            },
        )
        save_data(data)
        st.success("已提交！正在跳转到进度页面……")
        st.query_params["view"] = "workspace"
        st.query_params["page"] = "progress"
        st.rerun()


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
    st.markdown('<div class="section-title">AI 整理表达</div>', unsafe_allow_html=True)
    st.caption("把原始想法整理成更清楚、可处理的反馈。")
    deepseek_key, deepseek_default = get_deepseek_config()
    gemini_key, gemini_default = get_gemini_config()

    provider_options = ["DeepSeek", "Gemini", "本地规则"]

    with st.form("translator_form"):
        raw = st.text_area(
            "想说的话",
            value="",
            height=130,
            placeholder="直接写真实想法即可，例如：部门之间信息不同步，经常不知道事情推进到哪里。",
        )
        target = st.radio(
            "整理用途",
            ["给管理层", "给协同负责人", "给活动负责人"],
            horizontal=True,
        )
        provider = st.selectbox(
            "AI 服务",
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
        submitted = st.form_submit_button("整理表达")

    if submitted:
        if not raw.strip():
            st.warning("先写一点想反馈的内容。")
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
        left, right = st.columns([1.05, 0.95])
        with left:
            st.markdown(
                f"""
                <div class="glass-card">
                    <span class="tag">来源：{result.get("source", "本地规则")}</span>
                    <span class="tag">{result["category"]}</span>
                    <span class="tag">优先级：{result["priority"]}</span>
                    <span class="tag">热度：{result["heat"]}%</span>
                    <div class="idea-title">{result["title"]}</div>
                    <p>{result["translated"]}</p>
                    <p class="muted">建议下一步：{result["next_step"]}</p>
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

        if st.button("把转译结果送入想法广场", use_container_width=True):
            data["ideas"].insert(
                0,
                {
                    "id": f"idea-{uuid4().hex[:8]}",
                    "title": result["title"],
                    "category": result["category"],
                    "author": "AI 转译",
                    "anonymous": True,
                    "content": result["translated"],
                    "impact": result["next_step"],
                    "status": "待确认",
                    "heat": result["heat"],
                    "votes": 1,
                    "created_at": now_str(),
                },
            )
            save_data(data)
            st.success("已送入想法广场，协同小组可以继续合并、筛选和推进。")
            st.rerun()


def render_task_card(task: dict) -> None:
    color = status_color(task["status"])
    members = " / ".join(task["members"])
    st.markdown(
        f"""
        <div class="task-card">
            <div class="idea-head">
                <div>
                    <div class="idea-title">{task["name"]}</div>
                    <span class="tag" style="border-color:{color}; color:{color};">{task["status"]}</span>
                    <span class="tag">优先级：{task["priority"]}</span>
                    <span class="tag">截止：{task["due"]}</span>
                </div>
                <div class="heat">{task["progress"]}%<br><span style="font-size:11px;color:#dbeafe;">进度</span></div>
            </div>
            <div class="progress-shell"><div class="progress-bar" style="width:{task["progress"]}%;"></div></div>
            <p>负责人：{task["owner"]}</p>
            <p class="muted">参与方：{members}</p>
            <p class="muted">下一步：{task["next_step"]}</p>
            <span class="tag">激励：{task["reward"]}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_tasks(data: dict) -> None:
    st.markdown('<div class="section-title">事项看板</div>', unsafe_allow_html=True)
    st.caption("把“有人提了但没人接”的事情变成有状态、有负责人、有下一步的协作任务。")

    statuses = ["待确认", "已受理", "推进中", "已完成", "暂缓"]
    cols = st.columns(len(statuses))
    for col, status in zip(cols, statuses):
        count = sum(1 for task in data["tasks"] if task["status"] == status)
        col.metric(status, count)

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


def render_event(data: dict) -> None:
    event = data["events"][0]
    done = sum(1 for slot in event["slots"] if slot["done"])
    total = len(event["slots"])
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
2. 对公共事务组织建立补贴、调休或贡献记录，防止“临时有空的人”持续承担隐性成本。
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
    st.caption("把“地下自发”转成公开、透明、有边界的员工事务协同机制。")

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
    st.markdown('<div class="section-title">提交反馈</div>', unsafe_allow_html=True)
    st.caption("提交后会进入公开列表，便于集中整理和跟进。")
    with st.expander("需要 AI 帮你整理表达？", expanded=False):
        render_translator(data)
    render_submit_form(data)


def render_feedback_progress(data: dict) -> None:
    st.markdown('<div class="section-title">查看进度</div>', unsafe_allow_html=True)
    st.caption("这里展示已经提交的反馈和正在推进的事项。")
    if not data["ideas"]:
        st.info("还没有反馈。可以先到“提交反馈”写下第一条。")
        return
    categories = ["全部"] + sorted({i["category"] for i in data["ideas"]})
    selected = st.segmented_control("反馈类型", categories, default="全部")
    for idea in data["ideas"]:
        if selected != "全部" and idea["category"] != selected:
            continue
        render_idea_card(idea)
        if st.button("删除", key=f"del_{idea['id']}", type="tertiary"):
            data["ideas"] = [i for i in data["ideas"] if i["id"] != idea["id"]]
            save_data(data)
            st.rerun()

    with st.expander("事项处理进度", expanded=False):
        render_tasks(data)



def star_position(index: int) -> tuple[int, int]:
    positions = [
        (64, 18), (72, 28), (55, 24), (82, 18), (46, 31),
        (68, 40), (37, 22), (76, 48), (58, 12), (88, 34),
        (49, 45), (62, 56), (34, 37), (79, 60), (91, 22),
    ]
    return positions[index % len(positions)]


def render_star_map(data: dict) -> None:
    st.markdown('<div class="section-title">星空意见图</div>', unsafe_allow_html=True)
    st.caption("每颗星代表一条员工反馈。点开星星查看详情。")
    ideas = data["ideas"]
    hero_uri = image_data_uri(NIGHT_HERO_IMAGE)
    selected_id = st.query_params.get("idea", "")

    star_links = []
    for index, idea in enumerate(ideas):
        x, y = star_position(index)
        title = idea["title"].replace('"', "&quot;")
        star_links.append(
            f'<a class="star-link" style="left:{x}%; top:{y}%;" '
            f'href="?view=workspace&page=stars&idea={idea["id"]}" title="{title}">{title}</a>'
        )

    empty_text = "" if ideas else "<p class='muted'>还没有反馈。提交第一条后，这里会出现第一颗星。</p>"
    st.markdown(
        f"""
        <div class="star-map" style="background-image: url('{hero_uri}');">
            <div class="star-map-title">
                <h2>意见像星星一样被看见</h2>
                <p>把分散的想法放在同一片天空里，方便大家查看和跟进。</p>
                {empty_text}
            </div>
            {''.join(star_links)}
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected = next((idea for idea in ideas if idea["id"] == selected_id), None)
    if selected:
        color = status_color(selected["status"])
        st.markdown(
            f"""
            <div class="star-detail">
                <span class="tag">{selected["category"]}</span>
                <span class="tag" style="border-color:{color}; color:{color};">{selected["status"]}</span>
                <span class="tag">热度 {selected["heat"]}%</span>
                <div class="idea-title">{selected["title"]}</div>
                <p>{selected["content"]}</p>
                <p class="muted">希望改进：{selected["impact"]}</p>
                <p class="muted">提交时间：{selected["created_at"]}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_star_page(data: dict) -> None:
    ideas = data["ideas"]
    hero_uri = image_data_uri(NIGHT_HERO_IMAGE)
    selected_id = st.query_params.get("idea", "")

    star_links = []
    for index, idea in enumerate(ideas):
        x, y = star_position(index)
        title = idea["title"].replace('"', "&quot;")
        star_links.append(
            f'<a class="sp-star" style="left:{x}%; top:{y}%;" '
            f'href="?view=stars&idea={idea["id"]}" title="{title}">{title}</a>'
        )

    selected = next((idea for idea in ideas if idea["id"] == selected_id), None)
    detail_html = ""
    if selected:
        color = status_color(selected["status"])
        detail_html = f"""
        <div class="star-page-detail">
            <span class="sp-tag">{selected["category"]}</span>
            <span class="sp-tag" style="border-color:{color}; color:{color};">{selected["status"]}</span>
            <span class="sp-tag">热度 {selected["heat"]}%</span>
            <div class="sp-idea-title">{selected["title"]}</div>
            <p style="margin:6px 0 4px; color:#d5deed;">{selected["content"]}</p>
            <p style="margin:0; color:#9aa7bd; font-size:13px;">希望改进：{selected["impact"]} · {selected["created_at"]}</p>
        </div>
        """

    empty_text = "" if ideas else "<p style='color:#9aa7bd;margin:12px 0 0;'>还没有反馈，提交第一条后这里会出现第一颗星。</p>"

    st.html(
        f"""
        <style>
        .star-shell {{
            min-height: 100vh;
            margin: -1.2rem calc(50% - 50vw) 0;
        }}
        .star-page {{
            position: relative;
            min-height: 100vh;
            overflow: hidden;
            background-size: cover;
            background-position: center;
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
            top: 36px;
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
        .star-page-back {{
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
            text-decoration: none;
        }}
        .star-page-back:hover {{
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
        .star-page-detail {{
            position: absolute;
            bottom: 32px;
            left: 40px;
            right: 40px;
            z-index: 10;
            border: 1px solid rgba(255,255,255,0.16);
            border-radius: 16px;
            background: rgba(8,13,26,0.78);
            backdrop-filter: blur(18px);
            padding: 20px 24px;
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
        </style>
        <div class="star-shell">
            <div class="star-page" style="background-image: url('{hero_uri}');">
                <div class="star-page-title">
                    <h2>意见像星星一样被看见</h2>
                    <p>把分散的想法放在同一片天空里，方便大家查看和跟进。</p>
                    {empty_text}
                </div>
                <a class="star-page-back" href="?view=landing">← 返回</a>
                {''.join(star_links)}
                {detail_html}
            </div>
        </div>
        """
    )


def render_settings_panel() -> None:
    st.markdown("**AI 设置**")
    deepseek_key, deepseek_model = get_deepseek_config()
    gemini_key, gemini_model = get_gemini_config()
    st.caption(f"DeepSeek：{'已配置' if deepseek_key else '未配置'}")
    st.caption(f"DeepSeek 默认模型：{deepseek_model}")
    st.caption(f"Gemini：{'已配置' if gemini_key else '未配置'}")
    st.caption(f"Gemini 默认模型：{gemini_model}")


def sidebar(data: dict) -> None:
    with st.sidebar:
        st.markdown("## 群星")
        st.caption("反馈收集 · 进度公开")
        if st.button("返回入口页", use_container_width=True):
            st.session_state["view"] = "landing"
            st.query_params.clear()
            st.rerun()
        st.divider()
        st.metric("想法总数", len(data["ideas"]))
        st.metric("事项总数", len(data["tasks"]))
        st.divider()
        render_settings_panel()


def main() -> None:
    st.set_page_config(
        page_title=f"{APP_TITLE} · 反馈与跟进",
        page_icon="✨",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()
    data = load_data()
    if "view" not in st.session_state:
        st.session_state["view"] = "landing"
    qv = st.query_params.get("view")
    if qv == "workspace":
        st.session_state["view"] = "workspace"
    elif qv == "stars":
        st.session_state["view"] = "stars"
    elif qv == "landing":
        st.session_state["view"] = "landing"
        st.query_params.clear()

    if st.session_state["view"] == "landing":
        render_landing(data)
        return

    if st.session_state["view"] == "stars":
        render_star_page(data)
        return

    sidebar(data)
    page_param = st.query_params.get("page")
    default_page = {"progress": "查看进度"}.get(page_param, "提交反馈")
    page = st.segmented_control("页面", ["提交反馈", "查看进度"], default=default_page)
    render_hero(data)
    if page == "提交反馈":
        render_submit_feedback(data)
    else:
        render_feedback_progress(data)


if __name__ == "__main__":
    main()
