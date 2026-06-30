# 팀 협업 가이드 — 강인후와그외5

5명이 충돌 없이 함께 작업하기 위한 규칙. (git 초보 기준으로 작성)

## 1. 매번 지키는 기본 흐름
```bash
git pull origin main          # ① 작업 시작 전 항상 최신화
git checkout -b 이름/작업내용   # ② 내 브랜치 만들기 (예: hyeonguk/eda)
# ... 작업 ...
git add .
git commit -m "무엇을 왜 했는지"
git push -u origin 이름/작업내용  # ③ 내 브랜치 올리기
```
그 뒤 GitHub에서 **Pull Request(PR)** 를 만들어 팀원 1명이 확인 후 `main`에 병합한다.

> **왜 브랜치?** 모두가 `main`에 직접 push하면 서로 코드를 덮어써 충돌이 난다.
> 각자 브랜치에서 작업 → PR로 합치면 안전하다.

## 2. 브랜치 이름 규칙
`이름/작업` 형식. 예: `hyeonguk/baseline-tfidf`, `jongn/encoder`, `team/eda`.

## 3. 커밋 메시지
한 줄로 "무엇을" 적는다. 예: `Add macro-F1 CV for tfidf baseline`, `Fix history parsing in featurelib`.

## 4. 충돌 줄이는 습관
- 같은 파일을 동시에 크게 고치지 않기 (역할 분담대로).
- 작업은 작게 자주 commit/push, 시작 전 항상 `git pull`.
- 큰 데이터·모델 가중치는 git에 올리지 않는다(`.gitignore`가 막아줌).

## 5. 역할 분담 (제안 — 팀 상황에 맞게 조정)
| 담당 | 주요 파일 | 일 |
|---|---|---|
| ① 데이터/EDA·피처 | `submit/featurelib.py`, `notebooks/` | 데이터 분석, 피처 설계, 클래스 분포 파악 |
| ② 고전 ML 베이스라인 | `src/baseline_tfidf.py`, `src/cv.py` | TF-IDF/LightGBM, 교차검증 점수 관리 |
| ③ 인코더 모델링 | `src/baseline_encoder.py`, `tools/download_model.py` | 소형 인코더 파인튜닝, 오프라인 모델 준비 |
| ④ 제출·인프라 | `submit/script.py`, `tools/run_local_eval.py`, `tools/make_submit.py`, `env/` | 파이프라인·패키징·속도/메모리 최적화, 제출 담당 |
| ⑤ 검증·기록·발표 | `src/metrics.py`, `experiments/`, 발표/포스터 | 실험 기록, 결과 검증, 본선 자료 |

## 6. 일일 제출(10회) 운영
- 리더보드 제출은 **로컬 CV로 거른 뒤**에만 한다(횟수 낭비 방지).
- 제출은 ④ 담당이 총괄하고, 누가 무엇을 제출했는지 `experiments/`에 기록.
- 제출 전 체크리스트(아래)를 매번 확인.

## 7. 제출 전 체크리스트
- [ ] `python tools/run_local_eval.py` 통과 (10분 이내)
- [ ] 오프라인 테스트 통과 (`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`)
- [ ] `python tools/make_submit.py` 로 zip 생성, 최상위에 `script.py` 있고 1GB 이하
- [ ] `requirements.txt`에 서버 기본 패키지 미포함
- [ ] 사용한 외부 모델/데이터 출처 기록 (대회 규칙)
