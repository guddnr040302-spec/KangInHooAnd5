"""
학습 스크립트 (로컬 개발용).
- 실행: python src/train.py
- 산출물: 학습된 가중치/토크나이저를 submit/model/ 에 저장 → 이후 submit.zip 으로 패키징
- 이 파일은 제출 zip에는 포함하지 않는다 (제출에는 추론용 script.py만 필요).

※ 데이터 명세를 확인한 뒤 아래 TODO를 채운다.
"""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 프로젝트 루트
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_OUT_DIR = os.path.join(BASE_DIR, "submit", "model")


def load_train_data():
    # TODO: 학습 데이터 로드 (current_prompt, history, session_meta -> label)
    raise NotImplementedError


def build_features(data):
    # TODO: 텍스트/메타 피처 구성
    raise NotImplementedError


def train(features, labels):
    # TODO: 모델 학습 (경량·고속 우선: 작은 인코더 또는 피처 + 고전 ML)
    raise NotImplementedError


def save_model(model):
    os.makedirs(MODEL_OUT_DIR, exist_ok=True)
    # TODO: 가중치/토크나이저를 MODEL_OUT_DIR 에 저장
    raise NotImplementedError


if __name__ == "__main__":
    data = load_train_data()
    X = build_features(data)
    # y = ...
    # model = train(X, y)
    # save_model(model)
    print("학습 완료 (TODO 구현 필요)")
