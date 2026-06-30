"""
사전학습 모델을 '오프라인용'으로 미리 받아 submit/model/ 에 저장한다.

평가 서버는 추론 중 인터넷이 없으므로, 대회 전에 인터넷이 되는 환경에서
이 스크립트를 한 번 실행해 가중치/토크나이저를 로컬에 박아둬야 한다.

실행(인터넷 필요, action-decision 환경):
    python tools/download_model.py --model microsoft/Multilingual-MiniLM-L12-H384
    python tools/download_model.py --model klue/roberta-base --out submit/model/klue

이후 학습/추론 코드에서는 저장된 로컬 폴더 경로로만 from_pretrained 한다.
"""
import argparse
import os


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="허깅페이스 모델 ID")
    ap.add_argument("--out", default=None, help="저장 폴더 (기본: submit/model/<모델명>)")
    args = ap.parse_args()

    if args.out is None:
        safe = args.model.replace("/", "__")
        args.out = os.path.join(root, "submit", "model", safe)
    os.makedirs(args.out, exist_ok=True)

    from transformers import AutoTokenizer, AutoModel
    print(f"다운로드: {args.model}")
    AutoTokenizer.from_pretrained(args.model).save_pretrained(args.out)
    AutoModel.from_pretrained(args.model).save_pretrained(args.out)

    total = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fs in os.walk(args.out) for f in fs
    )
    print(f"저장 완료: {args.out}  ({total / 1024**2:.1f} MB)")
    print("→ 이제 코드에서 from_pretrained(\"" + os.path.relpath(args.out, root) + "\") 로 로드")
    if total > 1024**3:
        print("[경고] 단일 모델이 1GB를 넘음 — fp16 저장 등 용량 축소 검토 필요")


if __name__ == "__main__":
    main()
