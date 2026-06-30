"""
더미(가짜) 평가 데이터 생성기 — 데이터 공개 전 파이프라인 검증용.

실제 대회 데이터의 컬럼/형식은 아직 모르므로, 문제 설명을 바탕으로 그럴듯한
구조를 임시로 만든다. 샘플 평가 데이터가 공개되면 이 파일의 컬럼/형식을
실제와 똑같이 맞추면 된다.

사용법 (저장소 루트에서):
    python tools/make_dummy_data.py            # 기본 50건
    python tools/make_dummy_data.py --n 200    # 200건

생성물:
    data/test.csv     : 평가 입력 (라벨 없음 — 서버가 주는 형태를 흉내)
    data/_answer.csv  : 로컬 채점용 정답 (실제 서버엔 없음. 로컬 Macro-F1 확인용)
"""
import argparse
import json
import os
import random

# 14개 행동 클래스 (임시 placeholder — 데이터 공개 후 실제 클래스명으로 교체)
LABELS = [
    "read_file", "search_files", "edit_file", "create_file", "delete_file",
    "run_shell", "run_test", "ask_user", "web_search", "read_docs",
    "git_command", "install_package", "summarize", "finish",
]

PROMPT_SAMPLES = [
    "이 함수에 버그가 있는 것 같아 고쳐줘",
    "테스트를 실행해서 통과하는지 확인해줘",
    "프로젝트에서 login 관련 코드를 찾아줘",
    "README에 설치 방법을 추가해줘",
    "이 에러 메시지가 무슨 뜻이야?",
    "requirements.txt에 numpy를 추가해줘",
    "방금 변경한 내용을 커밋해줘",
    "이 모듈이 어떻게 동작하는지 설명해줘",
]


def make_history(rng):
    """직전까지의 대화·행동 이력을 흉내낸 리스트."""
    n = rng.randint(0, 4)
    turns = []
    for _ in range(n):
        turns.append({
            "role": rng.choice(["user", "assistant", "tool"]),
            "action": rng.choice(LABELS),
            "content": rng.choice(PROMPT_SAMPLES),
        })
    return turns


def make_session_meta(rng):
    """요금제·잔여 토큰 예산·작업공간 상태 등 세션 메타정보를 흉내."""
    return {
        "plan": rng.choice(["free", "pro", "max"]),
        "remaining_tokens": rng.randint(0, 200000),
        "workspace_files": rng.randint(1, 500),
        "has_uncommitted_changes": rng.choice([True, False]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="생성할 샘플 수")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(root, "data")
    os.makedirs(data_dir, exist_ok=True)

    rows, answers = [], []
    for i in range(args.n):
        rows.append({
            "id": i,
            "current_prompt": rng.choice(PROMPT_SAMPLES),
            "history": json.dumps(make_history(rng), ensure_ascii=False),
            "session_meta": json.dumps(make_session_meta(rng), ensure_ascii=False),
        })
        answers.append({"id": i, "label": rng.choice(LABELS)})

    import csv
    test_path = os.path.join(data_dir, "test.csv")
    with open(test_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "current_prompt", "history", "session_meta"])
        w.writeheader()
        w.writerows(rows)

    ans_path = os.path.join(data_dir, "_answer.csv")
    with open(ans_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "label"])
        w.writeheader()
        w.writerows(answers)

    print(f"생성 완료: {test_path} ({args.n}건)")
    print(f"생성 완료: {ans_path} (로컬 채점용 정답)")


if __name__ == "__main__":
    main()
