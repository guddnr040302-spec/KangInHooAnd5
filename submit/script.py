"""
추론 스크립트 (제출용).
평가 서버에서 `python script.py`로 자동 실행된다.

규칙
- data/ (읽기전용)에서 평가 데이터 로드
- model/ 에 동봉한 가중치/토크나이저 로드 (완전 오프라인)
- output/submission.csv 를 반드시 생성
- 추론 10분 이내 완료 (T4 16GB / 3 vCPU / 12GB RAM)

※ 실제 데이터 파일명·컬럼·submission 형식은 '샘플 평가 데이터'를 확인한 뒤 아래 TODO를 채워 확정한다.
"""
import os
import pandas as pd

# 평가 서버에서는 script.py / data/ / model/ / output/ 이 같은 위치(형제 디렉터리)에 놓인다.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# 14개 행동 클래스 (TODO: 샘플 데이터 확인 후 실제 클래스명으로 교체)
LABELS = [
    # 예: "read_file", "search", "edit_file", "run_shell", "run_test", "ask_user", ...
]


def load_data():
    """평가 데이터 로드."""
    # TODO: 실제 평가 데이터 파일명으로 교체
    path = os.path.join(DATA_DIR, "test.csv")
    return pd.read_csv(path)


def load_model():
    """model/ 디렉터리에서 로컬 가중치/토크나이저 로드 (오프라인)."""
    # TODO: 실제 모델 로드 코드로 교체
    #   예) from transformers import AutoModelForSequenceClassification, AutoTokenizer
    #       tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    #       model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    return None


def predict(model, data):
    """추론 수행 → 예측 라벨 리스트 반환."""
    # TODO: 실제 추론 로직으로 교체
    # 아래는 제출 파이프라인 검증용 임시 더미 (가장 흔한 클래스로 전부 채움)
    n = len(data)
    return ["ask_user"] * n


def save_results(data, predictions):
    """output/submission.csv 생성 (필수)."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # TODO: 실제 submission 형식(id 컬럼 포함 여부 등)으로 교체
    submission = pd.DataFrame({"prediction": predictions})
    submission.to_csv(os.path.join(OUTPUT_DIR, "submission.csv"), index=False)


if __name__ == "__main__":
    data = load_data()
    model = load_model()
    preds = predict(model, data)
    save_results(data, preds)
    print(f"추론 완료: {len(preds)}건 -> output/submission.csv")
