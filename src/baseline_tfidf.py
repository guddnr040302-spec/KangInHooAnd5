"""
베이스라인 1: TF-IDF + 메타피처 + 로지스틱 회귀.

- 가볍고 빠르며 추가 패키지가 필요 없다(scikit-learn은 서버 기본 설치).
- 데이터 나온 첫날 '제대로 된' 첫 제출을 만드는 용도.
- 학습 후 번들을 submit/model/baseline_tfidf.joblib 로 저장 → submit/script.py 가 로드.

실행 (저장소 루트, action-decision 환경):
    python tools/make_dummy_data.py          # 더미 데이터 먼저
    python src/baseline_tfidf.py

※ 지금은 더미 데이터(data/test.csv + data/_answer.csv)를 학습에 쓴다.
   실제 학습 데이터가 나오면 load_train() 만 실제 파일로 바꾸면 된다.
"""
import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)  # `src` 패키지 import 가능하게 (python src/... 직접 실행 대비)
sys.path.insert(0, os.path.join(ROOT, "submit"))  # featurelib 공유
import featurelib as fl  # noqa: E402
from src.metrics import macro_f1, print_report  # noqa: E402
from src.cv import make_folds, compute_class_weights  # noqa: E402

MODEL_OUT = os.path.join(ROOT, "submit", "model", "baseline_tfidf.joblib")


def load_train():
    """학습 데이터 로드. (TODO: 실제 학습 파일로 교체)"""
    df = pd.read_csv(os.path.join(ROOT, "data", "test.csv"))
    ans = pd.read_csv(os.path.join(ROOT, "data", "_answer.csv"))
    df = df.merge(ans, on="id")
    return df, df["label"].values


def main():
    df, y = load_train()
    print(f"학습 샘플 수: {len(df)}")

    # 피처 구성요소 fit
    text = fl.build_text(df)
    vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1, 2), max_features=50000)
    vectorizer.fit(text)

    meta = fl.build_meta_features(df)
    meta_columns = list(meta.columns)
    scaler = StandardScaler().fit(meta.values)

    X = fl.assemble_features(df, vectorizer, scaler, meta_columns)

    # 교차검증으로 Macro-F1 추정 (클래스 불균형 → class_weight='balanced')
    from collections import Counter
    min_count = min(Counter(y).values())
    if min_count < 2:
        print(f"[건너뜀] 가장 적은 클래스 표본이 {min_count}개라 교차검증 생략 "
              f"(실제 데이터에선 정상 동작).")
        fold_scores = []
    else:
        n_splits = max(2, min(5, min_count))
        print(f"교차검증 (n_splits={n_splits}):")
        fold_scores = []
        for k, (tr, va) in enumerate(make_folds(y, n_splits=n_splits), 1):
            clf = LogisticRegression(max_iter=1000, class_weight="balanced")
            clf.fit(X[tr], y[tr])
            pred = clf.predict(X[va])
            s = macro_f1(y[va], pred)
            fold_scores.append(s)
            print(f"  fold {k}: Macro-F1 = {s:.4f}")
        print(f"  ---- 평균 Macro-F1 = {np.mean(fold_scores):.4f} (±{np.std(fold_scores):.4f})")

    # 전체 데이터로 최종 학습
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X, y)

    # 번들 저장 (추론 시 script.py 가 이 객체들을 그대로 사용)
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    joblib.dump(
        {"vectorizer": vectorizer, "scaler": scaler, "model": model,
         "meta_columns": meta_columns, "labels": fl.LABELS},
        MODEL_OUT,
    )
    print(f"저장 완료: {MODEL_OUT}")


if __name__ == "__main__":
    main()
