"""
더미(가짜) 데이터 생성기 — 실제 대회 데이터 스키마를 그대로 흉내낸다.

실제 데이터(train.jsonl 등)가 있으면 이걸 쓸 필요는 없지만, 빠른 파이프라인
테스트/CI 용으로 유용하다. 실제와 동일한 JSON Lines 구조로 만든다.

사용법 (저장소 루트에서):
    python tools/make_dummy_data.py               # train 300 / test 50
    python tools/make_dummy_data.py --n 1000 --n_test 100

생성물:
    data/train.jsonl        : 학습 입력 (id/session_meta/history/current_prompt)
    data/train_labels.csv   : 학습 정답 (id, action)
    data/test.jsonl         : 평가 입력 (정답 없음)
    data/_answer.csv        : 로컬 채점용 정답 (실제 서버엔 없음)
    data/sample_submission.csv : 제출 양식 (id, action)
"""
import argparse
import csv
import json
import os
import random

LABELS = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch",
    "run_bash", "run_tests", "lint_or_typecheck",
    "ask_user", "plan_task", "web_search", "respond_only",
]
USER_TIERS = ["enterprise", "pro", "free"]
LANG_PREFS = ["ko", "en", "mixed"]
CI_STATUS = ["passed", "failed", "none"]
LANGS = ["py", "js", "ts", "sql", "go", "java", "rs"]

PROMPTS = [
    "이 함수에 버그가 있는 것 같아 고쳐줘",
    "테스트를 실행해서 통과하는지 확인해줘",
    "프로젝트에서 login 관련 코드를 찾아줘",
    "README에 설치 방법을 추가해줘",
    "이 에러 메시지가 무슨 뜻이야?",
    "please refactor this module",
    "run the linter and fix warnings",
    "search where config is loaded",
]
RESULTS = ["ok", "3 matches found", "exit code 0", "2 files changed",
           "tests passed", "no results", "patch applied"]


def make_workspace(rng):
    k = rng.randint(1, 3)
    langs = rng.sample(LANGS, k)
    raw = [rng.random() for _ in langs]
    s = sum(raw) or 1.0
    lang_mix = {l: round(v / s, 2) for l, v in zip(langs, raw)}
    return {
        "language_mix": lang_mix,
        "loc": rng.randint(100, 100000),
        "git_dirty": rng.choice([True, False]),
        "open_files": [f"src/mod{i}.py" for i in range(rng.randint(0, 4))],
        "last_ci_status": rng.choice(CI_STATUS),
    }


def make_session_meta(rng):
    return {
        "user_tier": rng.choice(USER_TIERS),
        "language_pref": rng.choice(LANG_PREFS),
        "budget_tokens_remaining": rng.randint(0, 200000),
        "turn_index": rng.randint(0, 12),
        "elapsed_session_sec": rng.randint(0, 3600),
        "workspace": make_workspace(rng),
    }


def make_history(rng):
    n = rng.randint(0, 12)
    turns, is_user = [], True
    for _ in range(n):
        if is_user:
            turns.append({"role": "user", "content": rng.choice(PROMPTS)})
        else:
            turns.append({
                "role": "assistant_action",
                "name": rng.choice(LABELS),
                "args": {},
                "result_summary": rng.choice(RESULTS),
            })
        is_user = not is_user
    return turns


def make_sample(rng, sid):
    return {
        "id": sid,
        "session_meta": make_session_meta(rng),
        "history": make_history(rng),
        "current_prompt": rng.choice(PROMPTS),
    }


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_csv(path, header, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="학습 샘플 수")
    ap.add_argument("--n_test", type=int, default=50, help="평가 샘플 수")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d = os.path.join(root, "data")
    os.makedirs(d, exist_ok=True)

    # 학습
    train = [make_sample(rng, f"sess_dummy-step_{i:05d}") for i in range(args.n)]
    write_jsonl(os.path.join(d, "train.jsonl"), train)
    write_csv(os.path.join(d, "train_labels.csv"), ["id", "action"],
              [[s["id"], rng.choice(LABELS)] for s in train])

    # 평가 (정답 없음) + 로컬 채점용 정답
    test = [make_sample(rng, f"sess_dummy_test-step_{i:05d}") for i in range(args.n_test)]
    write_jsonl(os.path.join(d, "test.jsonl"), test)
    answers = [[s["id"], rng.choice(LABELS)] for s in test]
    write_csv(os.path.join(d, "_answer.csv"), ["id", "action"], answers)

    # 제출 양식
    write_csv(os.path.join(d, "sample_submission.csv"), ["id", "action"],
              [[s["id"], LABELS[0]] for s in test])

    print(f"생성 완료: train.jsonl({args.n}) / test.jsonl({args.n_test}) "
          f"/ train_labels.csv / _answer.csv / sample_submission.csv  @ {d}")


if __name__ == "__main__":
    main()
