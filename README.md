# 강인후와그외5 — AI Agent 행동(Action) 의사결정 예측 챌린지

## 팀
- **팀명:** 강인후와그외5 (5인)

## 대회 개요
AI 코딩 에이전트 세션의 특정 시점 상태 데이터를 바탕으로, 에이전트가 **다음에 수행할 행동(action)을 14개 클래스 중 하나로 예측**한다.

- **입력:** `current_prompt`(현재 발화), `history`(대화·행동 이력), `session_meta`(요금제·잔여 토큰·작업공간 상태 등)
- **평가지표:** Macro-F1 (14 classes) — 클래스 불균형에 민감하므로 소수 클래스도 잘 맞혀야 함
- **방식:** 코드 제출 대회 (`submit.zip` = `model/` + `script.py` + `requirements.txt`)

## 평가 서버 제약 (반드시 준수)
- 환경: **T4 GPU 16GB / 3 vCPU / 12GB RAM**, Ubuntu 22.04, **Python 3.11.15**, CUDA 12.8
- **추론 ≤ 10분**, **패키지 설치 ≤ 10분**, **제출 zip ≤ 1GB**
- **오프라인** — 패키지 설치 외 인터넷 불가. 모델·토크나이저·설정파일을 전부 `model/`에 동봉할 것 (`from_pretrained("hub이름")` ❌ → `from_pretrained("model/...")` ✅)
- `data/`는 읽기전용, 결과는 반드시 `output/submission.csv`로 저장

## 주요 일정
- **예선 제출 마감:** 2026-07-15(수) 10:00 — 1일 최대 10회 제출
- **본선 코드/발표:** 2026-07-20(월) 10:00
- **포스터:** 2026-07-30(목) 10:00

## 폴더 구조
```
KangInHooAnd5/
├── README.md            # 이 문서
├── .gitignore           # 데이터/모델 가중치 등 git 제외 목록
├── data/                # 로컬 샘플 데이터 (git에는 올리지 않음)
├── src/                 # 학습·실험 코드
│   └── train.py         # 학습 코드 (로컬 전용, 제출 zip에는 미포함)
└── submit/              # ★ 제출 패키징 폴더 (이 안의 내용물을 zip으로 압축)
    ├── script.py        # 추론 코드 (평가 서버에서 자동 실행)
    ├── requirements.txt # 추론용 추가 패키지만 명시
    └── model/           # 학습된 가중치/토크나이저 (오프라인용)
```

> **제출 시 주의:** `submit/` 폴더 자체가 아니라 **그 안의 내용물**(`script.py`, `requirements.txt`, `model/`)이 zip 최상위에 오도록 압축해야 함. 최상위에 불필요한 폴더가 한 겹 더 있으면 구조 불일치로 설치 오류 발생.

## 개발 흐름
1. `src/train.py`로 로컬에서 모델 학습 → 가중치를 `submit/model/`에 저장
2. `submit/script.py`로 로컬에서 추론 테스트 (`data/` 샘플로 `output/submission.csv` 생성 확인)
3. `submit/` 내용물을 zip으로 압축 → 대회 사이트 [제출] 탭 업로드

## git 기본 사용법 (팀원 공통)
```bash
git pull                       # 최신 코드 받기 (작업 시작 전 항상)
# ... 코드 수정 ...
git add .                      # 변경사항 담기
git commit -m "무엇을 했는지"   # 저장
git push                       # GitHub에 올리기
```
