# 사전학습 코퍼스 · 학습 목표 · 토크나이저 (출처 검증본)

**작성:** 2026-07-30
**검증 방법:** HF Hub API의 git blob sha1 대조, HF 캐시 실측, 논문 원문(ar5iv) 대조, 체크포인트 텐서 실측.

이 문서의 핵심 결론 2줄:
1. **토크나이저는 BERT-base와 MiniLM이 완전히 동일하다** (`vocab.txt`가 바이트 단위 같은 파일).
2. **코퍼스는 같지 않다.** BERT는 16GB(Wiki+Books), 우리가 쓰는 MiniLM 릴리즈본은 **160GB**(그 16GB를 포함하는 상위집합)에서 증류됐다.

---

## 1. 요약 표

| | **bert-base-uncased** | **MiniLM-L12-H384-uncased** (student text) | **DINOv3 ViT-S/16** (student vision) | **BLIP-large** (teacher) |
|---|---|---|---|---|
| 사전학습 데이터 | English Wikipedia + BookCorpus (**≈16GB**) | Wikipedia + BookCorpus + OpenWebText + CC-News + Stories (**160GB**) | **LVD-1689M** (16.89억 이미지) | 129M 이미지-텍스트 쌍 (COCO, VG, SBU, CC3M, CC12M, LAION) |
| 학습 방식 | 처음부터 사전학습 | **증류** (task-agnostic KD) | **증류** (self-distillation) | 처음부터 사전학습 |
| 티처 | — | **in-house UniLM v2 (BERT-base 크기)** | **DINOv3 ViT-7B** | — |
| 학습 목표 | MLM + NSP | $\mathcal{L}_{AT} + \mathcal{L}_{VR}$ (마지막 층 self-attention KL) | DINO/iBOT 계열 self-supervised + Gram anchoring | ITC + ITM + LM (CapFilt 부트스트랩 캡션) |
| 토크나이저 / 전처리 | WordPiece 30,522 uncased | **동일 파일** | ImageNet mean/std, 256×256 native | bert-base-uncased WordPiece |
| 파라미터 | 109M | 33.4M | 21.6M | 474.7M (online) |

**우리 학생의 학습 데이터** (참고): `used_dataset: coco_vg`
= COCO Karpathy train **566,747 캡션 / 113,287 이미지** + VG **5,408,689 캡션 / 108,077 이미지**
→ 티처(129M 이미지)의 약 **0.17%** 규모. 이 격차가 증류를 하는 이유이자, 학생이 티처의 마진을 재현하지 못하는 1차 원인이다.

---

## 2. MiniLM: 어느 판본인가 (가장 오해하기 쉬운 지점)

MiniLM 논문(Wang et al., NeurIPS 2020, arXiv:2002.10957)에는 **같은 이름의 L12-H384 모델이 두 개** 있다.

| | **논문 §4.1 (main experiments)** | **논문 §5.1 "Better Teacher Better Student"** |
|---|---|---|
| 티처 | BERT-base-uncased (109M) | in-house UniLM v2 계열, BERT-base 크기 |
| 코퍼스 | English Wikipedia(enwiki-20181101) + BookCorpus ≈ 16GB | RoBERTa-base급 **160GB**: Wikipedia + BookCorpus + OpenWebText + CC-News + Stories |
| 학생 증류 데이터 | 위 16GB | 논문 원문: *"We distill the teacher model into 12-layer and 6-layer models with 384 hidden size **using the same corpora**"* → 160GB |
| SQuAD2 / MNLI-m | 논문 Table 3 계열 | **81.7 / 85.7** (논문 Table 8) |

**우리가 쓰는 `microsoft/MiniLM-L12-H384-uncased` = §5.1 판본이다.** 근거 2개:

1. HF 모델 카드 원문: *"We release the uncased 12-layer model with 384 hidden size **distilled from an in-house pre-trained UniLM v2 model in BERT-Base size**."*
   (microsoft/unilm repo의 `minilm/README.md`도 동일: *"...distilled from an in-house pre-trained UniLM v2 model in BERT-Base size. The models use the same WordPiece vocabulary as BERT."*)
2. 카드에 실린 성능표 `SQuAD2 81.7 / MNLI-m 85.7 / SST-2 93.0 / QNLI 91.5 / CoLA 58.5 / RTE 73.3 / MRPC 89.5 / QQP 91.3` 가
   **논문 Table 8(=§5.1)과 숫자 하나까지 일치**한다. §4.1 표와는 일치하지 않는다.

→ 따라서 **"MiniLM은 BERT를 증류한 축소판"이라는 서술은 부정확하다.** 정확한 서술:

> MiniLM-L12-H384-uncased는 BERT-base와 **동일한 WordPiece 어휘(30,522)** 를 공유하지만,
> **UniLMv2 계열 BERT-base 크기 티처**로부터 **160GB 코퍼스**에서 **마지막 층 self-attention 증류**로 학습된 12층 384차원 모델이다.
> MLM·NSP 목표는 사용되지 않았다.

---

## 3. MiniLM의 학습 목표 — MLM/NSP가 "없다"는 것의 의미

논문 Table 1의 "Distilled Knowledge" 열: MiniLM = **Self-Attention distributions + Self-Attention value relation**.
(DistilBERT는 soft target probabilities + embedding outputs, TinyBERT/MobileBERT는 hidden states까지 사용)

$$\mathcal{L}_{AT}=\frac{1}{A_h|x|}\sum_{a=1}^{A_h}\sum_{t=1}^{|x|} D_{KL}\!\left(\mathbf{A}^T_{L,a,t}\,\|\,\mathbf{A}^S_{M,a,t}\right)$$
$$\mathbf{VR}_{L,a}=\mathrm{softmax}\!\left(\frac{\mathbf{V}_{L,a}\mathbf{V}_{L,a}^\top}{\sqrt{d_k}}\right),\qquad \mathcal{L}_{VR}=D_{KL}(\mathbf{VR}^T \,\|\, \mathbf{VR}^S)$$

전체 손실 $\mathcal{L} = \mathcal{L}_{AT} + \mathcal{L}_{VR}$ — **티처의 마지막 층 self-attention 기하만 모사**한다.
학생은 단어를 예측하도록(MLM) 학습된 적이 없고, 문장쌍 관계를 판별하도록(NSP) 학습된 적도 없다.

**체크포인트로 확인되는 결과:**

| | bert-base-uncased | MiniLM-L12-H384-uncased |
|---|---|---|
| 텐서 수 / 파라미터 | 206 / 109M | **199 / 33,360,000** |
| `cls.predictions.*` (MLM 헤드) | **있음** | **없음** |
| `cls.seq_relationship.*` (NSP 헤드) | **있음** | **없음** |
| `pooler.*` | 있음 | 있음 (BLIP은 `add_pooling_layer=False`로 미사용) |

MiniLM repo의 `log.txt`(변환 당시 로그)는 **45,260,348** 파라미터를 기록한다. 분해하면
`33,360,000` (현재 배포본) `+ 11,899,578` (untied MLM head: 30522×384 + transform dense/LN + bias) `+ 770` (NSP head: 384×2+2)
`= 45,260,348` — **자릿수까지 정확히 일치**. 즉 원래는 `BertForPreTraining` 래퍼였고, 배포판에서 두 헤드를 떼어냈다.

**우리 프로젝트에 미치는 실제 영향** (→ `configs/model_size.md` §3):
`BertLMHeadModel.from_pretrained('microsoft/MiniLM-...')` 시
`cls.predictions.transform.{dense,LayerNorm}` + `cls.predictions.bias`가 **랜덤 초기화**된다
(출력 행렬 자체는 word_embeddings와 tie돼 있어 로드됨). bert-base 학생은 이 부분을 사전학습값으로 받는다.

---

## 4. 토크나이저 — 바이트 단위 동일 (증명)

| 항목 | bert-base-uncased | MiniLM-L12-H384-uncased |
|---|---|---|
| `vocab.txt` git blob sha1 | `fb140275c155a9c7c5a3b3e0e77a9e839594a938` | **동일** |
| `vocab.txt` 크기 | 231,508 B | **동일** |
| `vocab_size` | 30,522 | 30,522 |
| `tokenizer_config.json` | 48 B (`{"do_lower_case": true}`) | **2 B = `{}`** ⚠️ |

git blob sha1은 `blob <size>\0<content>`의 SHA-1이므로 **동일 sha1 + 동일 size ⇒ 내용 완전 동일**이다.
로컬 HF 캐시(`~/.cache/huggingface/hub/models--bert-base-uncased/blobs/fb140275...`)의 파일명이 바로 그 sha1이라,
현재 런이 쓰는 vocab이 문자 그대로 그 파일이다.

**`models/blip.py:195-200` `init_tokenizer()` 실측:**
```
BertTokenizer, vocab_size=30522, do_lower_case=True
[CLS]=101  [SEP]=102  [PAD]=0
[DEC]=30522 (bos_token)   [ENC]=30523 (additional_special_tokens[0])
len(tokenizer) = 30524
tokenize("a photo of forecasted dogs playing") -> ['a','photo','of','forecast','##ed','dogs','playing']
```

**주의사항 (현재는 문제 없음)**
- MiniLM repo의 `tokenizer_config.json`이 `{}` 라 `do_lower_case`가 명시돼 있지 않다.
  `AutoTokenizer.from_pretrained("microsoft/MiniLM-...")`로 직접 로드하면 `BertTokenizer` 기본값(True)에 의존한다.
  우리는 `bert-base-uncased`에서 로드하므로 **현재 방식이 더 견고하다.** 바꾸지 말 것.
- unilm repo에는 NLG 파인튜닝용 별도 vocab(`minilm-l12-h384-uncased-vocab-nlg.txt`)이 있으나 우리는 사용하지 않는다.
- MiniLMv2 계열(`MiniLM-L12-H384-distilled-from-RoBERTa-Large` 등)은 **RoBERTa BPE**를 쓴다.
  이걸 골랐다면 티처(bert-base tokenizer)와 토큰 id가 어긋나 **LM logit distillation이 원리적으로 불가능**했다.

**왜 이게 중요한가:** LM logit 증류는 티처와 학생이 동일 vocab·동일 id 배치를 가져야 성립한다.
`models/blip_pretrain.py:519`에 방어선이 있다:
```python
assert torch.equal(teacher_lm_input_ids, decoder_input_ids), \
    "teacher/student decoder input mismatch (tokenizer drift?)"
```
vocab이 바이트 동일이므로 이 assert는 구조적으로 항상 참이다.

---

## 5. `vocab_size` 30522 / 30524 두 config가 공존하는 이유 (검증됨: 버그 아님)

| 파일 | vocab_size | 쓰이는 경로 |
|---|---|---|
| `configs/bert_minilm_config.json` | **30522** | `blip_pretrain.py:45` 기본값 → pretrain에서 `from_pretrained` 직후 `resize_token_embeddings(30524)` |
| `configs/med_minilm_config.json` | **30524** | `models/blip.py:30` `DECODER_CONFIGS` → 학생 체크포인트 로드(이미 30524, resize 불필요) |

`resize_token_embeddings`가 **공유 config 객체를 in-place로 30524로 갱신**하고(`m.config is cfg == True` 실측),
momentum encoder는 `blip_pretrain.py:242`에서 **그 이후에** 생성되므로 30524로 만들어진다 → `copy_params()` shape mismatch 없음.
(transformers 4.33.3에서 실측 확인)

---

## 6. DINOv3 ViT-S/16 (student vision) 보충

- timm id: `vit_small_patch16_dinov3.lvd1689m` — 21,586,944 파라미터, 12층, 384차원, **6 heads**, MLP 1536, patch 16, register token 4개
- HF 카드 원문: *"A DINOv3 ViT model image feature encoder. **Distilled on LVD-1689M from the DINOv3 ViT-7B model.**"*
- **native 입력 256×256**(`fixed_input_size: True`)인데 우리는 `img_size=224`로 생성 → position 처리 재구성. 학습/평가 모두 224로 일관.
- 위치 인코딩은 **RoPE** (학습 파라미터 없음 — `named_parameters`에 pos/rope 항목 0개)
- timm은 원본의 전부-0인 QKV bias를 로드하지 않고 `qkv_bias=False`로 만든다 (모델 카드 명시)
- 정규화: ImageNet mean/std `(0.485,0.456,0.406)/(0.229,0.224,0.225)`
- 출력 토큰: raw 201 = 1 CLS + 4 reg + 196 patch → `DINOv3_Wrapper`가 reg 4개를 제거해 **197** (BLIP의 ViT-B/L과 동일한 197)
- ⚠️ `vit: 'small'`(reg 없는 경로)은 timm 1.0.27에서 체크포인트의 `reg_token` 키 때문에 strict load 실패한다.
  현재 실험은 모두 `small_reg`를 쓰므로 영향 없음.

---

## 7. 티처 BLIP-large 보충

- 체크포인트: `output/official_pretrain_checkpoint/model_large.pth` (공식 Salesforce BLIP-large **pretrain**, 129M 이미지 설정)
  실측: 1,560 텐서 / 1,087,042,940 파라미터(momentum + queue 포함), `visual_encoder.pos_embed` = (1,197,1024), 24 ViT 블록, 12 텍스트 층,
  `itm_head.weight` = (2,768), `text_encoder.embeddings.word_embeddings.weight` = (30524,768), `temp` 스칼라 보유
- 비전 백본 초기화: `vit_large_patch16_224_in21k` (ImageNet-21k). BLIP-base는 `deit_base_patch16_224`.
- 학습 목표: ITC + ITM + LM 3항. 웹 캡션은 BLIP의 CapFilt(captioner+filter)로 부트스트랩된 것을 사용(BLIP 논문 기준).
- 텍스트 측은 bert-base-uncased → **학생과 토크나이저 동일**(§4).

---

## 관련 문서
- `configs/model_size.md` — 파라미터 회계, 사전학습/랜덤 초기화 분해, `itm_head` 기하
- `docs/model_specs/` — 위 두 문서의 시각화(HTML) + 재생성 스크립트
- `critical_bugfix/2026-07-24_itm_distill_sharpness/`, `critical_bugfix/2026-07-27_itm_eval_ranking_and_structure/` — ITM 증류 조사
