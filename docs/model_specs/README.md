# docs/model_specs — 모델 스펙 자료 모음

티처(BLIP-large) / 참조(BLIP-base) / 학생(DINOv3 ViT-S + MiniLM-L12-H384)의
**구조 · 파라미터 · 사전학습 코퍼스 · 학습 목표 · 토크나이저**를 한곳에 모은 폴더.
모든 수치는 코드를 실제로 실행해 측정한 값이며 재생성 가능하다.

## 파일

| 파일 | 내용 |
|---|---|
| `model_specs.html` | **시각화 (브라우저로 열기)** — 스펙 카드, 파라미터 구성 스택 차트, 사전학습/랜덤 초기화 분해, 코퍼스 표, `itm_head` 기하 차트. light/dark 자동 대응, 외부 리소스 없음 |
| `count_specs.py` | 파라미터 실측 스크립트. `python docs/model_specs/count_specs.py` (repo root에서, GPU 불필요) |
| `specs.json` | 위 스크립트 출력 (HTML/md의 숫자 출처) |

## 원본 문서 (텍스트 기준본)

| 문서 | 내용 |
|---|---|
| `configs/model_size.md` | 파라미터 회계, 모듈별 분해, 사전학습 로드 vs 랜덤 초기화, `itm_head` 유효 자유도와 기하 |
| `configs/corpus_and_pretraining.md` | 코퍼스·학습 목표·토크나이저 검증본. MiniLM 판본 확정 근거, vocab 바이트 동일성 증명, DINOv3/BLIP 보충 |

## 핵심 수치 (요약)

| | BLIP-large (티처) | BLIP-base | 학생 |
|---|---|---|---|
| online 고유 파라미터 | 474,729,022 | 252,441,918 | **69,387,198** (14.6%) |
| vision | ViT-L/16 · 303.3M | ViT-B/16 · 85.8M | DINOv3 ViT-S/16 · 21.6M (7.1%) |
| text | bert-base · 768 | bert-base · 768 | MiniLM · 384 |
| 랜덤 초기화 (cross-attention) | 33.1M | 28.4M | **7.1M** (fused encoder의 17.6%) |
| `itm_head` | `Linear(768,2)` = 1,538 | 1,538 | `Linear(384,2)` = **770** (유효 자유도 385) |

세 줄 결론:

1. **토크나이저는 bert-base와 MiniLM이 바이트 단위 동일**(`vocab.txt` git blob sha1 일치) → LM logit 증류의 vocab 정렬이 구조적으로 보장된다.
2. **코퍼스는 다르다.** 우리가 쓰는 MiniLM 릴리즈본은 UniLMv2 티처로부터 **160GB**에서 증류된 판본(논문 §5.1)이며, BERT의 16GB가 아니다.
3. **`itm_head`는 명목 770개 중 385개만 손실이 구속**한다. 나머지(합 성분 wₛ)는 weight decay만 깎으며, 티처는 수렴해 wₛ≈0(비율 0.038)이지만 학생은 0.55~0.86으로 남아 있다. `separation` 원값은 `‖w_d‖`에 비례하므로 모델 간 비교가 성립하지 않고, **올바른 스케일 무관 지표는 within-query `d′`** 다 (티처 2.195 > baseline 1.492 > schemeA 1.083). → `critical_bugfix/2026-07-30_itm_logit_offset_and_head_geometry/`

## 관련 조사 문서
- `critical_bugfix/2026-07-24_itm_distill_sharpness/` — arm C 실패 + sharpness 프로브 + 스킴 A/B 설계
- `critical_bugfix/2026-07-27_itm_eval_ranking_and_structure/` — scheme A 실패, z1 vs gap 재정렬 실험, per-pair 이진 정답률
