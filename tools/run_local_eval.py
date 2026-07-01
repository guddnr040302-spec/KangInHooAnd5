"""
로컬 평가 서버 시뮬레이터 — 제출 전 '유효 제출'을 끝까지 검증한다.

평가 서버가 하는 일을 그대로 흉내낸다:
  1) submit/ 내용물(script.py, requirements.txt, model/)을 임시 폴더 최상위로 복사
  2) data/ 를 임시 폴더에 추가 (읽기 입력)
  3) output/ 디렉터리 추가
  4) `python script.py` 실행 — 시간(10분) + 최대 RAM + 최대 VRAM 측정
  5) output/submission.csv 생성 여부·형식 검증
  6) (정답 파일이 있으면) Macro-F1 계산

서버 제한: 추론 ≤ 10분, RAM ≤ 12GB, VRAM ≤ 16GB (T4).

사용법 (저장소 루트에서):
    python tools/run_local_eval.py

메모리 측정에는 psutil(RAM)과 nvidia-smi(VRAM)가 필요하다.
없으면 해당 항목만 건너뛰고 나머지는 정상 검증한다.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time

TIME_LIMIT_SEC = 600            # 추론 10분
RAM_LIMIT_GB = 12               # 서버 RAM
VRAM_LIMIT_GB = 16              # T4 VRAM
POLL_SEC = 0.2                  # 메모리 표본 간격

try:
    import psutil
    HAVE_PSUTIL = True
except ImportError:
    HAVE_PSUTIL = False


def _sample_ram_bytes(ps):
    """프로세스 + 자식들의 RSS 합계(bytes)."""
    try:
        procs = [ps] + ps.children(recursive=True)
        return sum(p.memory_info().rss for p in procs if p.is_running())
    except Exception:
        return 0


def _pid_tree(ps, root_pid):
    pids = {root_pid}
    try:
        pids |= {c.pid for c in ps.children(recursive=True)}
    except Exception:
        pass
    return pids


def _sample_vram_mib(pids):
    """nvidia-smi로 해당 PID들의 GPU 사용량(MiB) 합계. GPU/nvidia-smi 없으면 None."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode != 0:
            return None
        total = 0
        for line in out.stdout.strip().splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 2 and parts[0].isdigit() and int(parts[0]) in pids:
                try:
                    total += int(parts[1])
                except ValueError:
                    pass
        return total
    except Exception:
        return None


def _run_with_monitor(work):
    """script.py 를 실행하며 시간·RAM·VRAM 피크를 측정. 반환 dict."""
    out_path = os.path.join(work, "_stdout.txt")
    err_path = os.path.join(work, "_stderr.txt")
    out_f = open(out_path, "w", encoding="utf-8")
    err_f = open(err_path, "w", encoding="utf-8")

    t0 = time.time()
    proc = subprocess.Popen([sys.executable, "script.py"], cwd=work,
                            stdout=out_f, stderr=err_f, text=True)
    ps = psutil.Process(proc.pid) if HAVE_PSUTIL else None

    peak_ram = 0
    peak_vram = None
    timed_out = False
    while True:
        ret = proc.poll()
        elapsed = time.time() - t0

        if HAVE_PSUTIL:
            peak_ram = max(peak_ram, _sample_ram_bytes(ps))
            pids = _pid_tree(ps, proc.pid)
        else:
            pids = {proc.pid}
        v = _sample_vram_mib(pids)
        if v is not None:
            peak_vram = max(peak_vram or 0, v)

        if ret is not None:
            break
        if elapsed > TIME_LIMIT_SEC:
            timed_out = True
            if HAVE_PSUTIL:
                for c in ps.children(recursive=True):
                    try:
                        c.kill()
                    except Exception:
                        pass
            proc.kill()
            break
        time.sleep(POLL_SEC)

    out_f.close()
    err_f.close()
    with open(out_path, encoding="utf-8") as f:
        stdout = f.read()
    with open(err_path, encoding="utf-8") as f:
        stderr = f.read()

    return {
        "elapsed": time.time() - t0,
        "returncode": proc.returncode,
        "timed_out": timed_out,
        "peak_ram_gb": peak_ram / 1024 ** 3 if peak_ram else None,
        "peak_vram_gb": peak_vram / 1024 if peak_vram is not None else None,
        "stdout": stdout,
        "stderr": stderr,
    }


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", default=os.path.join(root, "submit"))
    ap.add_argument("--data", default=os.path.join(root, "data"))
    ap.add_argument("--answer", default=os.path.join(root, "data", "_answer.csv"))
    args = ap.parse_args()

    ok = True

    script_src = os.path.join(args.submit, "script.py")
    if not os.path.isfile(script_src):
        print(f"[실패] script.py 가 없음: {script_src}")
        sys.exit(1)
    if not os.path.isdir(args.data):
        print(f"[실패] data 폴더가 없음: {args.data}  (먼저 make_dummy_data.py 실행)")
        sys.exit(1)
    if not HAVE_PSUTIL:
        print("[안내] psutil 미설치 → RAM 측정 생략 (pip install psutil 하면 측정됨)")

    work = tempfile.mkdtemp(prefix="local_eval_")
    try:
        # submit 내용물 → 임시 폴더 최상위
        for name in os.listdir(args.submit):
            src = os.path.join(args.submit, name)
            dst = os.path.join(work, name)
            if os.path.isdir(src):
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)

        # data/ 복사 (로컬 전용 _파일 제외)
        data_dst = os.path.join(work, "data")
        os.makedirs(data_dst, exist_ok=True)
        for name in os.listdir(args.data):
            if name.startswith("_"):
                continue
            src = os.path.join(args.data, name)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(data_dst, name))

        os.makedirs(os.path.join(work, "output"), exist_ok=True)

        # 실행 + 자원 모니터링
        print(f"[실행] python script.py  (작업 폴더: {work})")
        r = _run_with_monitor(work)

        if r["stdout"].strip():
            print("--- script.py stdout ---\n" + r["stdout"].strip())
        if r["timed_out"]:
            print(f"[실패] 추론이 {TIME_LIMIT_SEC}s(10분)를 초과함 → 제출 오류")
            sys.exit(1)
        if r["returncode"] != 0:
            print("--- script.py stderr ---\n" + r["stderr"].strip())
            print(f"[실패] script.py 가 에러로 종료 (returncode={r['returncode']}) → 제출 오류")
            sys.exit(1)

        # 시간
        print(f"[OK] 실행 시간 {r['elapsed']:.1f}s / 제한 {TIME_LIMIT_SEC}s")

        # RAM
        if r["peak_ram_gb"] is not None:
            mark = "OK " if r["peak_ram_gb"] <= RAM_LIMIT_GB else "!! "
            print(f"[{mark}] 최대 RAM {r['peak_ram_gb']:.2f} GB / 제한 {RAM_LIMIT_GB} GB")
            if r["peak_ram_gb"] > RAM_LIMIT_GB:
                print("        └ RAM 초과 위험 → 배치 축소/모델 경량화 필요"); ok = False
            elif r["peak_ram_gb"] > RAM_LIMIT_GB * 0.85:
                print("        └ RAM 여유 부족(85% 초과) — 주의")

        # VRAM
        if r["peak_vram_gb"] is not None:
            mark = "OK " if r["peak_vram_gb"] <= VRAM_LIMIT_GB else "!! "
            print(f"[{mark}] 최대 VRAM {r['peak_vram_gb']:.2f} GB / 제한 {VRAM_LIMIT_GB} GB")
            if r["peak_vram_gb"] > VRAM_LIMIT_GB:
                print("        └ VRAM 초과 → fp16/배치 축소/작은 모델 필요"); ok = False
        else:
            print("[안내] VRAM 측정 생략 (GPU 미사용 또는 nvidia-smi 없음)")

        # submission 검증
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

    print("\n" + ("=" * 44))
    print("결과: " + ("✅ 파이프라인 검증 통과" if ok else "⚠️ 통과(경고 있음) — 위 메시지 확인"))
    print("=" * 44)


if __name__ == "__main__":
    main()
