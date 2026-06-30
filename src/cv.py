"""
교차검증(CV) 유틸 — 리더보드 제출을 아끼면서 로컬에서 점수를 안정적으로 추정한다.
(하루 제출 10회 제한이 있으므로 로컬 CV로 거르는 게 중요)

사용 예:
    from src.cv import make_folds, compute_class_weights, run_cv
    folds = make_folds(y, n_splits=5)
    weights = compute_class_weights(y)   # 클래스 불균형 대응
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

from .metrics import macro_f1


def make_folds(y, n_splits=5, seed=42):
    """라벨 분포를 유지한 채 (train_idx, val_idx) 쌍 리스트 반환."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    dummy_X = np.zeros((len(y), 1))
    return list(skf.split(dummy_X, y))


def compute_class_weights(y):
    """'balanced' 클래스 가중치를 {클래스: 가중치} dict로 반환.
    소수 클래스에 큰 가중치를 줘 Macro-F1을 끌어올리는 데 사용."""
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return dict(zip(classes, weights))


def run_cv(fit_predict_fn, X, y, n_splits=5, seed=42, labels=None):
    """범용 OOF(Out-Of-Fold) 교차검증 루프.

    fit_predict_fn(X_tr, y_tr, X_va) -> y_va_pred  형태의 함수를 넘기면
    폴드별 Macro-F1과 OOF 예측을 돌려준다.
    """
    y = np.asarray(y)
    oof = np.empty(len(y), dtype=object)
    fold_scores = []
    for k, (tr, va) in enumerate(make_folds(y, n_splits, seed), 1):
        X_tr = X[tr] if hasattr(X, "__getitem__") else [X[i] for i in tr]
        X_va = X[va] if hasattr(X, "__getitem__") else [X[i] for i in va]
        pred = fit_predict_fn(X_tr, y[tr], X_va)
        oof[va] = pred
        score = macro_f1(y[va], pred)
        fold_scores.append(score)
        print(f"  fold {k}: Macro-F1 = {score:.4f}")
    print(f"  ---- 평균 Macro-F1 = {np.mean(fold_scores):.4f} "
          f"(±{np.std(fold_scores):.4f})")
    return oof, fold_scores
