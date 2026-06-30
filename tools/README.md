# tools/ — 제출 파이프라인 검증 도구

데이터가 없어도 "유효 제출" 흐름을 끝까지 검증하기 위한 개발용 스크립트 모음.
(제출 zip에는 포함하지 않는다 — 추론에는 `submit/`만 필요.)

## 1. 더미 데이터 만들기
```bash
python tools/make_dummy_data.py --n 50
```
`data/test.csv`(평가 입력 흉내)와 `data/_answer.csv`(로컬 채점용 정답)를 만든다.
> 실제 샘플 데이터가 공개되면 `make_dummy_data.py`의 컬럼·형식을 실제와 똑같이 맞출 것.

## 2. 로컬에서 평가 서버 흉내내어 검증
```bash
python tools/run_local_eval.py
```
`submit/` 내용물을 임시 폴더 최상위로 복사 → `data/`, `output/` 추가 →
`python script.py` 실행(10분 제한 측정) → `output/submission.csv` 생성·형식 검증 →
(정답이 있으면) Macro-F1 계산. **이게 통과하면 서버에서도 돌 가능성이 높다.**

## 3. 제출 zip 자동 패키징
```bash
python tools/make_submit.py
```
`submit/`의 **내용물**을 zip 최상위로 묶어 `submit.zip` 생성.
구조 불일치·1GB 초과·기본 패키지 중복을 자동 점검한다.

## 전체 흐름 한 번에
```bash
python tools/make_dummy_data.py      # 1) 더미 데이터
python tools/run_local_eval.py       # 2) 로컬 검증
python tools/make_submit.py          # 3) 제출 zip 생성
```
