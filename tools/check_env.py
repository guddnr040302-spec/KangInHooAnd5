"""
내 로컬 환경이 평가 서버와 일치하는지 점검한다.

사용법:
    python tools/check_env.py

각 패키지 버전을 서버 기준과 비교해 OK / 불일치 / 없음 을 표시하고,
GPU(CUDA) 사용 가능 여부도 알려준다.
"""
import sys

try:
    import importlib.metadata as md
except ImportError:  # py<3.8 대비 (사실상 불필요)
    import importlib_metadata as md

# (pip 패키지명, 서버 기준 버전)  — torch는 +cu128 빌드라 앞부분만 비교
EXPECTED = [
    ("torch", "2.7.1"),
    ("numpy", "1.26.4"),
    ("pandas", "2.0.3"),
    ("scipy", "1.15.3"),
    ("scikit-learn", "1.8.0"),
    ("joblib", "1.5.3"),
    ("threadpoolctl", "3.6.0"),
    ("narwhals", "2.21.2"),
    ("transformers", "4.46.3"),
    ("accelerate", "1.9.0"),
    ("sentencepiece", "0.1.99"),
    ("regex", "2023.12.25"),
    ("tqdm", "4.66.4"),
    ("loguru", "0.7.2"),
    ("pyyaml", "6.0.1"),
    ("rich", "13.7.1"),
]


def main():
    # 1) Python 버전
    py = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    py_ok = (sys.version_info.major, sys.version_info.minor) == (3, 11)
    mark = "OK " if py_ok else "!! "
    print(f"[{mark}] Python {py}   (서버: 3.11.15)")
    if not py_ok:
        print("        └ 3.11 계열이 아님 — conda 환경을 활성화했는지 확인 "
              "(conda activate action-decision)")

    # 2) 패키지 버전 비교
    print("-" * 56)
    mismatches = 0
    for pkg, exp in EXPECTED:
        try:
            ver = md.version(pkg)
        except md.PackageNotFoundError:
            print(f"[MISS] {pkg:<16} 설치 안 됨           (서버: {exp})")
            mismatches += 1
            continue
        base = ver.split("+")[0]
        ok = base == exp
        if not ok:
            mismatches += 1
        mark = "OK " if ok else "!! "
        note = "" if ok else f"  <- 서버: {exp}"
        print(f"[{mark}] {pkg:<16} {ver}{note}")

    # 3) GPU / CUDA
    print("-" * 56)
    try:
        import torch
        if torch.cuda.is_available():
            print(f"[OK ] CUDA 사용 가능 — {torch.cuda.get_device_name(0)} "
                  f"(torch CUDA {torch.version.cuda})")
        else:
            print("[!! ] CUDA 사용 불가 — CPU로만 실행됨 "
                  "(NVIDIA GPU/드라이버 또는 CPU용 torch 확인)")
    except Exception as e:
        print(f"[MISS] torch import 실패: {e}")

    # 4) 요약
    print("=" * 56)
    if mismatches == 0 and py_ok:
        print("결과: ✅ 서버와 동일한 환경")
    else:
        print(f"결과: ⚠️ 불일치 {mismatches}건 + "
              f"{'Python OK' if py_ok else 'Python 버전 확인 필요'} "
              "— 위 '!!' 항목을 맞추면 됨")


if __name__ == "__main__":
    main()
