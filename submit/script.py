"""
추론 스크립트 (제출용). 평가 서버에서 `python script.py`로 자동 실행된다.

동작
- data/ 에서 평가 입력 로드
- model/baseline_tfidf.joblib 가 있으면 그걸로 추론, 없으면 안전 폴백(최빈 클래스)
- output/submission.csv 생성 (어떤 경우에도 반드시 생성 — 절대 죽지 않게 예외 처리)

규칙: 완전 오프라인, 추론 ≤ 10분, T4 16GB.
※ 실제 데이터 파일명·컬럼·submission 형식은 샘플 데이터 확인 후 확정한다(아래 TODO).
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
FALLBACK_LABEL = "ask_user"  # 모델 로드 실패 시 임시 라벨 (TODO: 최빈 클래스로)


def load_data():
    # TODO: 실제 평가 데이터 파일명으로 교체
    return pd.read_csv(os.path.join(DATA_DIR, "test.csv"))


def load_bundle():
    if not os.path.isfile(BUNDLE_PATH):
        return None
    import joblib
    return joblib.load(BUNDLE_PATH)


def predict(bundle, data):
    if bundle is None:
        # 안전 폴백: 모델이 없거나 못 읽어도 제출은 만들어진다
        return [FALLBACK_LABEL] * len(data)
    X = fl.assemble_features(data, bundle["vectorizer"], bundle["scaler"],
                             bundle["meta_columns"])
    return list(bundle["model"].predict(X))


def save_results(data, predictions):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # TODO: 실제 submission 형식에 맞춰 컬럼명/ id 포함 여부 확정
    out = pd.DataFrame({"prediction": predictions})
    if "id" in data.columns:
        out.insert(0, "id", data["id"].values)
    out.to_csv(os.path.join(OUTPUT_DIR, "submission.csv"), index=False)


if __name__ == "__main__":
    data = load_data()
    try:
        bundle = load_bundle()
        preds = predict(bundle, data)
    except Exception:
        # 어떤 오류가 나도 빈 제출보다 폴백 제출이 낫다
        traceback.print_exc()
        preds = [FALLBACK_LABEL] * len(data)
    save_results(data, preds)
    print(f"추론 완료: {len(preds)}건 -> output/submission.csv")
