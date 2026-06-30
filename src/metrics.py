"""
평가 지표 유틸 — 대회 공식 지표는 Macro-F1 (14 classes).

사용 예:
    from src.metrics import macro_f1, print_report
    print(macro_f1(y_true, y_pred))
    print_report(y_true, y_pred, labels=LABELS)
"""
from sklearn.metrics import f1_score, classification_report, confusion_matrix


def macro_f1(y_true, y_pred):
    """대회 공식 지표. 클래스별 F1의 단순 평균(클래스 불균형에 민감)."""
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def per_class_f1(y_true, y_pred, labels=None):
    """클래스별 F1을 dict로 반환 — 어떤 클래스가 약한지 진단용."""
    scores = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    keys = labels if labels is not None else sorted(set(y_true) | set(y_pred))
    return dict(zip(keys, scores))


def print_report(y_true, y_pred, labels=None):
    """클래스별 precision/recall/f1 + Macro-F1 출력."""
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))
    print(f"Macro-F1 = {macro_f1(y_true, y_pred):.4f}")


def confusion(y_true, y_pred, labels=None):
    """혼동행렬 반환 — 어떤 클래스끼리 헷갈리는지 확인용."""
    return confusion_matrix(y_true, y_pred, labels=labels)
