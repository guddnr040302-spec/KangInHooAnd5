# 모델 후보 조사 (오프라인 · ≤1GB · ≤10분 · T4 16GB)

입력은 **한국어 발화 + 코드/영어가 섞인 텍스트**(`current_prompt`, `history`) + 구조화된
`session_meta`다. 따라서 **다국어(한국어 포함) 소형 인코더**가 핵심 후보다.
아래 용량은 fp32 가중치 대략치이며, **라이선스는 제출 전 각 모델 카드에서 반드시 재확인**할 것.

## 추천 로드맵
1. **TF-IDF + LogisticRegression** (다운로드 없음) — `src/baseline_tfidf.py`로 이미 구현.
   가장 먼저 제출해 리더보드 파이프라인을 확정하고 기준선을 잡는다.
2. **다국어 소형 인코더 파인튜닝** — 텍스트 의미를 더 잘 잡아 점수 상승.
   속도 우선이면 MiniLM, 정확도 우선이면 mDeBERTa-v3-base.
3. **메타피처 결합 / 앙상블** — 인코더 출력 + `session_meta` 피처를 합쳐 마무리.

## 후보 비교

| 모델 | 파라미터/용량(fp32) | 한국어 | 속도(T4) | 라이선스 | 메모 |
|---|---|---|---|---|---|
| TF-IDF + LogReg/LightGBM | 수 MB | 토큰 기반 OK | 매우 빠름 | - | 다운로드 불필요, 첫 제출용 |
| microsoft/Multilingual-MiniLM-L12-H384 | ~118M / ~0.47GB | ○ | 빠름 | MIT | 속도·용량 균형 최고 |
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | ~118M / ~0.47GB | ○ | 빠름 | Apache-2.0 | 임베딩 활용에 좋음 |
| distilbert-base-multilingual-cased | ~134M / ~0.54GB | ○ | 빠름 | Apache-2.0 | 가벼운 대안 |
| microsoft/mdeberta-v3-base | ~278M / ~0.56GB | ○ | 보통 | MIT | 다국어 정확도 강함(추천) |
| klue/roberta-base | ~110M / ~0.44GB | ◎(한국어 특화) | 빠름 | 모델카드 확인 | 입력이 한국어 위주면 유리 |
| xlm-roberta-base | ~270M / ~1.1GB | ○ | 보통 | MIT | fp16로 저장하면 ~0.55GB |
| mmBERT (2025, 다국어 최신) | base/small | ○ | 보통 | 카드 확인 | 최신 다국어 인코더, 검토 가치 |

## 제약별 체크포인트
- **용량 1GB**: 위 후보는 대부분 여유. xlm-roberta-base는 fp16 저장 권장.
- **10분 추론**: T4에서 MiniLM/distil 계열이 안전. 큰 모델은 `max_length` 축소·fp16·배치 키우기로 대응.
- **오프라인**: 반드시 `tools/download_model.py`로 미리 받아 `submit/model/`에 저장 후 로컬 경로로 로드.
- **라이선스**: 상업적/대회 사용 가능 여부를 모델 카드에서 확인하고 `README`/발표자료에 출처 명시(대회 규칙).

## 다음 행동
- 데이터 공개 후 `current_prompt`의 한국어/영어 비중을 보고 MiniLM(다국어) vs klue(한국어 특화) 중 택1해 1차 파인튜닝.
- `python tools/download_model.py --model microsoft/Multilingual-MiniLM-L12-H384` 로 미리 받아두기.

## 출처
- mDeBERTa-v3 (NLI 파인튜닝 예시·라이선스): https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-mnli-xnli
- Multilingual-MiniLM-L12-H384: https://www.aimodels.fyi/models/huggingFace/multilingual-minilm-l12-h384-microsoft
- mmBERT (2025) 논문: https://arxiv.org/abs/2509.06888
- 오픈소스 임베딩 모델 비교: https://supermemory.ai/blog/best-open-source-embedding-models-benchmarked-and-ranked/
