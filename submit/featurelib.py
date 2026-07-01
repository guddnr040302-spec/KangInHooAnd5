"""
피처 추출 유틸 (오프라인 자립) — 실제 대회 데이터 스키마에 맞춤.

데이터: JSON Lines (한 줄 = 한 샘플 JSON 객체).
각 샘플 = id / session_meta / history / current_prompt.
타깃(action)은 train_labels.csv 에 별도로 있음(id로 연결).

학습 코드(src/baseline_tfidf.py)와 추론 코드(submit/script.py)가 이 파일을 공유해
피처가 어긋나지 않게 한다. submit/ 안에 두어 제출 zip에 함께 들어간다.
"""
import json
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix

# 14개 행동 클래스 (대소문자까지 정확히 일치해야 함)
LABELS = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch",
    "run_bash", "run_tests", "lint_or_typecheck",
    "ask_user", "plan_task", "web_search", "respond_only",
]
LABEL2IDX = {l: i for i, l in enumerate(LABELS)}

USER_TIERS = ["enterprise", "pro", "free"]
LANG_PREFS = ["ko", "en", "mixed"]
CI_STATUS = ["passed", "failed", "none"]


def load_jsonl(path):
    """JSON Lines 파일 → DataFrame (각 줄이 하나의 JSON 객체)."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def _as_dict(x):
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        try:
            v = json.loads(x)
            return v if isinstance(v, dict) else {}
        except Exception:
            return {}
    return {}


def _as_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            v = json.loads(x)
            return v if isinstance(v, list) else []
        except Exception:
            return []
    return []


def build_text(df):
    """TF-IDF/인코더 입력용 텍스트: current_prompt + history 발화·행동·결과 이어붙이기."""
    out = []
    for _, row in df.iterrows():
        prompt = str(row.get("current_prompt", "") or "")
        parts = [prompt]
        for t in _as_list(row.get("history")):
            if not isinstance(t, dict):
                continue
            if t.get("content"):            # 사용자 턴
                parts.append(str(t["content"]))
            if t.get("name"):               # 행동 턴 (14클래스 행동명)
                parts.append(str(t["name"]))
            if t.get("result_summary"):
                parts.append(str(t["result_summary"]))
        out.append(" ".join(parts).strip())
    return pd.Series(out, index=df.index)


def build_meta_features(df):
    """session_meta + workspace + history 통계를 숫자 피처 DataFrame으로."""
    rows = []
    for _, row in df.iterrows():
        prompt = str(row.get("current_prompt", "") or "")
        hist = _as_list(row.get("history"))
        meta = _as_dict(row.get("session_meta"))
        ws = _as_dict(meta.get("workspace"))

        action_turns = [t for t in hist if isinstance(t, dict) and t.get("name")]
        user_turns = [t for t in hist if isinstance(t, dict) and t.get("content")]
        last_action = action_turns[-1]["name"] if action_turns else None

        lang_mix = _as_dict(ws.get("language_mix"))
        open_files = _as_list(ws.get("open_files"))

        feat = {
            "prompt_char_len": len(prompt),
            "prompt_word_len": len(prompt.split()),
            "history_len": len(hist),
            "n_user_turns": len(user_turns),
            "n_action_turns": len(action_turns),
            "last_action_idx": LABEL2IDX.get(last_action, -1),
            "turn_index": float(meta.get("turn_index", 0) or 0),
            "elapsed_session_sec": float(meta.get("elapsed_session_sec", 0) or 0),
            "budget_tokens_remaining": float(meta.get("budget_tokens_remaining", 0) or 0),
            "loc": float(ws.get("loc", 0) or 0),
            "git_dirty": int(bool(ws.get("git_dirty", False))),
            "n_open_files": len(open_files),
            "n_languages": len(lang_mix),
            "max_lang_ratio": max(lang_mix.values()) if lang_mix else 0.0,
        }
        for t in USER_TIERS:
            feat[f"tier_{t}"] = int(meta.get("user_tier") == t)
        for lp in LANG_PREFS:
            feat[f"langpref_{lp}"] = int(meta.get("language_pref") == lp)
        for c in CI_STATUS:
            feat[f"ci_{c}"] = int(ws.get("last_ci_status") == c)
        rows.append(feat)
    return pd.DataFrame(rows, index=df.index).astype(float)


def assemble_features(df, vectorizer, scaler, meta_columns):
    """학습·추론 공통: TF-IDF(text) + 스케일된 메타 피처를 하나의 희소행렬로 결합."""
    X_text = vectorizer.transform(build_text(df))
    meta = build_meta_features(df).reindex(columns=meta_columns, fill_value=0.0)
    X_meta = scaler.transform(meta.values)
    return hstack([X_text, csr_matrix(X_meta)]).tocsr()
