"""
제출 zip 자동 패키징 — 구조를 규칙에 맞게 묶고 흔한 실수를 자동 점검한다.

핵심 규칙:
  - zip 최상위에 script.py / requirements.txt / model/ 이 바로 와야 한다.
    (submit/ 폴더째로 묶으면 '구조 불일치' 설치 오류!) → 이 스크립트가 알아서 맞춰줌
  - 총 용량 ≤ 1GB

사용법 (저장소 루트에서):
    python tools/make_submit.py            # submit.zip 생성
    python tools/make_submit.py --name submit_v2.zip

출력: 저장소 루트에 submit.zip (또는 지정한 이름)
"""
import argparse
import os
import zipfile

SIZE_LIMIT = 1 * 1024 ** 3  # 1GB

# 평가 서버에 이미 설치된 기본 패키지 (requirements.txt에 넣으면 충돌 위험)
BASE_PKGS = {
    "torch", "pandas", "numpy", "scipy", "scikit-learn", "sklearn", "joblib",
    "threadpoolctl", "narwhals", "transformers", "accelerate", "sentencepiece",
    "regex", "tqdm", "loguru", "pyyaml", "rich",
}


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", default=os.path.join(root, "submit"))
    ap.add_argument("--name", default="submit.zip")
    args = ap.parse_args()

    submit_dir = args.submit
    out_path = os.path.join(root, args.name)

    problems = []

    # 필수 파일 점검
    if not os.path.isfile(os.path.join(submit_dir, "script.py")):
        problems.append("script.py 가 submit/ 안에 없음")
    if not os.path.isfile(os.path.join(submit_dir, "requirements.txt")):
        problems.append("requirements.txt 가 submit/ 안에 없음")
    if not os.path.isdir(os.path.join(submit_dir, "model")):
        problems.append("model/ 폴더가 submit/ 안에 없음")

    # requirements.txt 에 기본 패키지가 들어있는지 경고
    req_path = os.path.join(submit_dir, "requirements.txt")
    warns = []
    if os.path.isfile(req_path):
        with open(req_path, encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                name = s.split("==")[0].split(">=")[0].split("<=")[0].split("~=")[0].strip().lower()
                if name in BASE_PKGS:
                    warns.append(f"requirements.txt 에 기본 패키지 '{name}' 포함 → 충돌 위험, 빼는 게 안전")

    if problems:
        print("[실패] 패키징 중단 — 다음을 먼저 해결:")
        for p in problems:
            print("  - " + p)
        raise SystemExit(1)

    # zip 생성: submit/ 의 '내용물'을 최상위로 (submit 폴더 자체는 포함 안 함)
    total = 0
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for dirpath, dirnames, filenames in os.walk(submit_dir):
            # 캐시/숨김 폴더 제외
            dirnames[:] = [d for d in dirnames if d not in ("__pycache__", ".ipynb_checkpoints")]
            for fn in filenames:
                if fn == ".gitkeep":
                    continue
                full = os.path.join(dirpath, fn)
                arc = os.path.relpath(full, submit_dir)  # submit/ 기준 상대경로 = zip 최상위
                zf.write(full, arc)
                total += os.path.getsize(full)

    zip_size = os.path.getsize(out_path)
    print(f"[OK] 생성: {out_path}")
    print(f"     원본 합계 {total/1024**2:.1f} MB / 압축 후 {zip_size/1024**2:.1f} MB")

    # zip 내부 구조 출력 (최상위에 script.py가 보여야 정상)
    with zipfile.ZipFile(out_path) as zf:
        names = zf.namelist()
    print("     zip 내부 구조(상위):")
    for n in sorted(names)[:15]:
        print("       " + n)

    if zip_size > SIZE_LIMIT:
        print(f"[실패] 용량 초과: {zip_size/1024**3:.2f} GB > 1GB 제한")
        raise SystemExit(1)
    if "script.py" not in names:
        print("[실패] zip 최상위에 script.py 가 없음 — 구조 불일치")
        raise SystemExit(1)

    for w in warns:
        print("[경고] " + w)
    print("\n✅ 제출 zip 준비 완료. 대회 사이트 [제출] 탭에 업로드하면 돼.")


if __name__ == "__main__":
    main()
