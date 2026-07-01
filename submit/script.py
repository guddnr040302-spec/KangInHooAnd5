"""
추론 스크립트 (제출용). 평가 서버에서 `python script.py`로 자동 실행된다.

동작
- data/test.jsonl 에서 평가 입력 로드 (JSON Lines)
- model/baseline_tfidf.joblib 가 있으면 그걸로 추론, 없으면 안전 폴백(최빈 클래스)
- output/submission.csv (컬럼: id, action) 생성 — 어떤 오류가 나도 반드시 생성

규칙: 완전 오프라인, 추론 ≤ 10분, T4 16GB.
제출 action 값은 14개 클래스명과 대소문자까지 정확히 일치해야 한다.
"""
import os
import sys
import traceback

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)  # featurelib import 가능하게
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

import featurelib as fl  # noqa: E402

BUNDLE_PATH = os.path.join(MODEL_DIR, "baseline_tfidf.joblib")
DEFAULT_FALLBACK = "read_file"  # 모델을 전혀 못 쓸 때 최후의 안전 라벨


def load_data():
    return fl.load_jsonl(os.path.join(DATA_DIR, "test.jsonl"))


def load_bundle():
    if not os.path.isfile(BUNDLE_PATH):
        return None
    import joblib
    return joblib.load(BUNDLE_PATH)


def predict(bundle, data):
    if bundle is None:
        return [DEFAULT_FALLBACK] * len(data)
    X = fl.assemble_features(data, bundle["vectorizer"], bundle["scaler"],
                             bundle["meta_columns"])
    return list(bundle["model"].predict(X))


def save_results(data, predictions):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ids = data["id"].values if "id" in data.columns else range(len(predictions))
    out = pd.DataFrame({"id": ids, "action": predictions})
    out.to_csv(os.path.join(OUTPUT_DIR, "submission.csv"), index=False)


if __name__ == "__main__":
    data = load_data()
    try:
        bundle = load_bundle()
        preds = predict(bundle, data)
    except Exception:
        traceback.print_exc()
        # 모델이 있으면 그 최빈 라벨, 없으면 기본값으로라도 제출을 만든다
        fallback = DEFAULT_FALLBACK
        try:
            b = load_bundle()
            if b and b.get("majority_label"):
                fallback = b["majority_label"]
        except Exception:
            pass
        preds = [fallback] * len(data)
    save_results(data, preds)
    print(f"추론 완료: {len(preds)}건 -> output/submission.csv")
