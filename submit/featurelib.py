"""
피처 추출 유틸 (오프라인 자립).

학습 코드(src/baseline_tfidf.py)와 추론 코드(submit/script.py)가 '동일한' 피처를
쓰도록 이 한 파일을 공유한다. submit/ 안에 두어 제출 zip에 함께 들어가게 한다
(추론 시 src/ 는 zip에 없으므로 import 불가).

※ 실제 데이터 컬럼/형식이 공개되면 아래 COLS 와 파싱 로직만 실제에 맞게 고친다.
"""
import json
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix

# 14개 행동 클래스 (임시 placeholder — 데이터 공개 후 실제 클래스명으로 교체)
LABELS = [
    "read_file", "search_files", "edit_file", "create_file", "delete_file",
    "run_shell", "run_test", "ask_user", "web_search", "read_docs",
    "git_command", "install_package", "summarize", "finish",
]
LABEL2IDX = {l: i for i, l in enumerate(LABELS)}

# 입력 컬럼명 (데이터 공개 후 실제 이름으로 교체)
COLS = {
    "prompt": "current_prompt",
    "history": "history",
    "meta": "session_meta",
}
PLANS = ["free", "pro", "max"]


def _loads(x, default):
    """문자열이면 JSON 파싱, 이미 객체면 그대로, 실패하면 default."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return default
    if isinstance(x, (list, dict)):
        return x
    try:
        return json.loads(x)
    except Exception:
        return default


def build_text(df):
    """TF-IDF/인코더 입력용 텍스트: current_prompt + history 내용 이어붙이기."""
    out = []
    for _, row in df.iterrows():
        prompt = str(row.get(COLS["prompt"], "") or "")
        hist = _loads(row.get(COLS["history"]), [])
        hist_text = " ".join(str(t.get("content", "")) for t in hist if isinstance(t, dict))
        out.append((prompt + " " + hist_text).strip())
    return pd.Series(out, index=df.index)


def build_meta_features(df):
    """세션 메타 + 이력 통계를 숫자 피처 DataFrame으로."""
    rows = []
    for _, row in df.iterrows():
        prompt = str(row.get(COLS["prompt"], "") or "")
        hist = _loads(row.get(COLS["history"]), [])
        meta = _loads(row.get(COLS["meta"]), {})
        if not isinstance(hist, list):
            hist = []
        if not isinstance(meta, dict):
            meta = {}

        roles = [t.get("role") for t in hist if isinstance(t, dict)]
        last_action = next(
            (t.get("action") for t in reversed(hist) if isinstance(t, dict) and t.get("action")),
            None,
        )
        feat = {
            "prompt_char_len": len(prompt),
            "prompt_word_len": len(prompt.split()),
            "history_len": len(hist),
            "n_user": roles.count("user"),
            "n_assistant": roles.count("assistant"),
            "n_tool": roles.count("tool"),
            "last_action_idx": LABEL2IDX.get(last_action, -1),
            "remaining_tokens": float(meta.get("remaining_tokens", 0) or 0),
            "workspace_files": float(meta.get("workspace_files", 0) or 0),
            "has_uncommitted_changes": int(bool(meta.get("has_uncommitted_changes", False))),
        }
        for p in PLANS:
            feat[f"plan_{p}"] = int(meta.get("plan") == p)
        rows.append(feat)
    return pd.DataFrame(rows, index=df.index).astype(float)


def assemble_features(df, vectorizer, scaler, meta_columns):
    """학습·추론 공통: TF-IDF(text) + 스케일된 메타 피처를 하나의 희소행렬로 결합.

    vectorizer/scaler 는 학습 때 fit 된 객체를 그대로 넘긴다(추론 시 model 번들에서 로드).
    """
    X_text = vectorizer.transform(build_text(df))
    meta = build_meta_features(df).reindex(columns=meta_columns, fill_value=0.0)
    X_meta = scaler.transform(meta.values)
    return hstack([X_text, csr_matrix(X_meta)]).tocsr()
