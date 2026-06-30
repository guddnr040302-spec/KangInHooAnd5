"""
베이스라인 2: 작은 사전학습 인코더 파인튜닝 골격.

- TF-IDF 베이스라인으로 점수가 한계에 부딪힐 때 다음 단계로 시도.
- 모델은 반드시 '오프라인'으로 동작해야 하므로, tools/download_model.py 로
  미리 받아 submit/model/<이름> 에 저장한 뒤 그 로컬 경로에서 로드한다.
- 이 파일은 '골격'이다. 데이터 형식 확정 후 TODO를 채워 완성한다.
  (torch / transformers 가 설치된 action-decision 환경에서 실행)

실행(예정):
    python tools/download_model.py --model <hf_id>     # 1) 오프라인용으로 미리 저장
    python src/baseline_encoder.py                     # 2) 파인튜닝
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)  # `src` 패키지 import 가능하게
sys.path.insert(0, os.path.join(ROOT, "submit"))
import featurelib as fl  # noqa: E402

# 미리 받아둔 로컬 모델 경로 (예시 — 실제 받은 폴더명으로 교체)
LOCAL_MODEL_DIR = os.path.join(ROOT, "submit", "model", "encoder")
NUM_LABELS = len(fl.LABELS)
MAX_LEN = 256
EPOCHS = 3
BATCH_SIZE = 16
LR = 2e-5


def load_train():
    """학습 데이터 -> (texts, labels). TODO: 실제 학습 파일로 교체."""
    import pandas as pd
    df = pd.read_csv(os.path.join(ROOT, "data", "test.csv"))
    ans = pd.read_csv(os.path.join(ROOT, "data", "_answer.csv"))
    df = df.merge(ans, on="id")
    texts = fl.build_text(df).tolist()
    labels = [fl.LABEL2IDX[l] for l in df["label"]]
    return texts, labels


def main():
    # 무거운 import는 함수 안에서 (골격 상태로도 파일을 import 할 수 있게)
    import numpy as np
    import torch
    from torch.utils.data import Dataset
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              Trainer, TrainingArguments)
    from src.metrics import macro_f1

    if not os.path.isdir(LOCAL_MODEL_DIR):
        raise SystemExit(
            f"로컬 모델이 없음: {LOCAL_MODEL_DIR}\n"
            f"→ 먼저 `python tools/download_model.py --model <hf_id>` 로 받아두세요."
        )

    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(
        LOCAL_MODEL_DIR, num_labels=NUM_LABELS)

    texts, labels = load_train()

    class DS(Dataset):
        def __init__(self, texts, labels):
            self.enc = tokenizer(texts, truncation=True, max_length=MAX_LEN, padding=True)
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
            item["labels"] = torch.tensor(self.labels[i])
            return item

    # TODO: train/val 분리 (src.cv.make_folds 활용) 후 검증 점수 측정
    train_ds = DS(texts, labels)

    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        return {"macro_f1": macro_f1(p.label_ids, preds)}

    args = TrainingArguments(
        output_dir=os.path.join(ROOT, "runs", "encoder"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LR,
        fp16=torch.cuda.is_available(),   # T4에서 속도/메모리 절약
        logging_steps=20,
        save_strategy="no",
        report_to=[],
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                      compute_metrics=compute_metrics)
    trainer.train()

    # 추론용으로 저장 (script.py 에서 from_pretrained(MODEL_DIR/encoder)로 로드)
    save_dir = os.path.join(ROOT, "submit", "model", "encoder_finetuned")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"저장 완료: {save_dir}")
    # NOTE: 인코더로 최종 제출하려면 submit/script.py 의 추론부도
    #       해당 인코더 로드/예측으로 교체해야 한다.


if __name__ == "__main__":
    main()
