"""
로컬 평가 서버 시뮬레이터 — 제출 전 '유효 제출'을 끝까지 검증한다.

평가 서버가 하는 일을 그대로 흉내낸다:
  1) submit/ 내용물(script.py, requirements.txt, model/)을 임시 폴더 최상위로 복사
  2) data/ 를 임시 폴더에 추가 (읽기 입력)
  3) output/ 디렉터리 추가
  4) `python script.py` 실행 (제한시간 10분 측정)
  5) output/submission.csv 생성 여부·형식 검증
  6) (정답 파일이 있으면) Macro-F1 계산

사용법 (저장소 루트에서):
    python tools/run_local_eval.py
    python tools/run_local_eval.py --submit submit --data data --answer data/_answer.csv

이게 통과하면 실제 서버에서도 '제출 오류' 없이 돌 가능성이 매우 높다.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time

TIME_LIMIT_SEC = 600  # 추론 10분


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", default=os.path.join(root, "submit"))
    ap.add_argument("--data", default=os.path.join(root, "data"))
    ap.add_argument("--answer", default=os.path.join(root, "data", "_answer.csv"))
    args = ap.parse_args()

    ok = True

    # 0) 입력 점검
    script_src = os.path.join(args.submit, "script.py")
    if not os.path.isfile(script_src):
        print(f"[실패] script.py 가 없음: {script_src}")
        sys.exit(1)
    if not os.path.isdir(args.data):
        print(f"[실패] data 폴더가 없음: {args.data}  (먼저 make_dummy_data.py 실행)")
        sys.exit(1)

    work = tempfile.mkdtemp(prefix="local_eval_")
    try:
        # 1) submit 내용물을 임시 폴더 최상위로 복사
        for name in os.listdir(args.submit):
            src = os.path.join(args.submit, name)
            dst = os.path.join(work, name)
            if os.path.isdir(src):
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)

        # 2) data/ 복사 (서버 입력 흉내). 정답 파일은 서버엔 없으니 제외.
        data_dst = os.path.join(work, "data")
        os.makedirs(data_dst, exist_ok=True)
        for name in os.listdir(args.data):
            if name.startswith("_"):  # _answer.csv 등 로컬 전용 파일 제외
                continue
            src = os.path.join(args.data, name)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(data_dst, name))

        # 3) output/ 생성
        os.makedirs(os.path.join(work, "output"), exist_ok=True)

        # 4) script.py 실행 (cwd=임시 루트, 제한시간 측정)
        print(f"[실행] python script.py  (작업 폴더: {work})")
        t0 = time.time()
        try:
            proc = subprocess.run(
                [sys.executable, "script.py"],
                cwd=work, timeout=TIME_LIMIT_SEC,
                capture_output=True, text=True,
            )
        except subprocess.TimeoutExpired:
            print(f"[실패] 추론이 {TIME_LIMIT_SEC}s(10분)를 초과함 → 제출 오류")
            sys.exit(1)
        elapsed = time.time() - t0

        if proc.stdout:
            print("--- script.py stdout ---\n" + proc.stdout.strip())
        if proc.returncode != 0:
            print("--- script.py stderr ---\n" + (proc.stderr or "").strip())
            print(f"[실패] script.py 가 에러로 종료 (returncode={proc.returncode}) → 제출 오류")
            sys.exit(1)
        print(f"[OK] 실행 시간 {elapsed:.1f}s / 제한 {TIME_LIMIT_SEC}s")

        # 5) submission.csv 검증
        sub_path = os.path.join(work, "output", "submission.csv")
        if not os.path.isfile(sub_path):
            print("[실패] output/submission.csv 가 생성되지 않음 → 제출 오류")
            sys.exit(1)

        import pandas as pd
        sub = pd.read_csv(sub_path)
        print(f"[OK] submission.csv 생성됨 — {len(sub)}행, 컬럼: {list(sub.columns)}")
        print(sub.head().to_string(index=False))

        test = pd.read_csv(os.path.join(data_dst, "test.csv"))
        if len(sub) != len(test):
            print(f"[경고] 행 수 불일치: submission {len(sub)} vs test {len(test)}")
            ok = False

        # 6) Macro-F1 (정답 파일이 있을 때만)
        if os.path.isfile(args.answer):
            from sklearn.metrics import f1_score
            ans = pd.read_csv(args.answer)
            pred_col = "prediction" if "prediction" in sub.columns else sub.columns[-1]
            n = min(len(ans), len(sub))
            macro_f1 = f1_score(ans["label"][:n], sub[pred_col][:n],
                                average="macro", zero_division=0)
            print(f"[참고] 더미 정답 기준 Macro-F1 = {macro_f1:.4f} "
                  f"(더미라 점수 자체는 의미 없음, 채점이 도는지 확인용)")

    finally:
        shutil.rmtree(work, ignore_errors=True)

    print("\n" + ("=" * 40))
    print("결과: " + ("✅ 파이프라인 검증 통과" if ok else "⚠️ 통과(경고 있음) — 위 메시지 확인"))
    print("=" * 40)


if __name__ == "__main__":
    main()
