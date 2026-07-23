# RandomAugment AutoContrast uint8 오버플로 수정 — 설계

관련 코드: `transform/randaugment.py` (`autocontrast_func`, 30–31줄)
성격: 상류 BLIP 상속 코드의 잠재 버그. cc12m 데이터로더 작업과 독립 (별개 이슈로 분리).

## 1. 배경 / 증상

`autocontrast_func`(기본 `cutoff=0`)는 채널을 `[low, high] → [0, 255]`로 선형 스트레치해야 한다. 변환표를 `새값 = v·scale + offset`, `offset = -low·scale`로 만든다.

`low = ch.min()`은 **np.uint8** 스칼라다. uint8은 음수를 못 담아 `-low`가 언더플로로 래핑된다(예: `low=5 → -low=251`). 결과:

```
scale  = 255/(high-low)
offset = -low*scale        # 의도: 음수(-6.5) / 실제: 큰 양수(+328)
table  = arange*scale + offset   # 전 구간이 255를 초과
table[table>255] = 255           # → 전부 255
```

수학적으로 `high ≤ 255`인 한 **`low>0`이면 offset은 항상 255를 넘겨 채널 전체가 255(백색)로 포화**된다. `low==0`(채널에 순검정 픽셀 존재)일 때만 래핑이 없어 정상. 이 경고(`RuntimeWarning: overflow encountered in scalar negative`)는 `models/blip.py:9`의 `warnings.filterwarnings("ignore")`에 가려져 최초 BLIP 커밋 이래 **조용히** 발생해 왔다.

## 2. 영향 정량 (실측)

실제 coco 학습 이미지 500장, 파이프라인 동일 crop(`RandomResizedCrop(224, 0.2~1.0)`+flip) 후 측정:

**발동 조건 빈도**
- 채널 하나 `min>0`: 28.1%
- **3채널 모두 `min>0` → 이미지 전체 백색: 15.6%**
- ≥1채널 `min>0`: 41.2%

**발동 시 심각도 (버그 vs 의도된 `PIL.ImageOps.autocontrast`)**
- 평균 픽셀 차이 87.1/255, 출력 픽셀의 68.6%가 255로 포화

**실제 학습 유효 손상률**
`RandomAugment(N=2, augs 10종)`: `np.random.choice(augs,2)`(복원추출) 후 각 op를 prob 0.5로 적용
→ P(AutoContrast 실제 적용) = 1−(1−0.1·0.5)² = **9.75%** / 이미지·에폭
- → 샘플의 **1.52%/에폭이 완전 백색** 이미지로(진짜 캡션과 짝지어짐)
- → 샘플의 **4.02%/에폭이 ≥1채널 백색화**
- 매 에폭 crop이 새로 뽑히므로 특정 이미지 고정 손상이 아닌 **일시적 노이즈**

**판정**: 실재 버그, 심각도 중하. 학습은 정상 수렴해 파괴급은 아니나, 의도된 증강과 87/255 어긋나며 매 에폭 ~4% 샘플을 훼손하는 저강도 데이터 품질 누수. coco/vg/cc12m 공통(transform 공유). 리스크 없는 한 줄 수정 대상.

## 3. 설계 (수정)

`transform/randaugment.py`의 `tune_channel` 30–31줄, 음수화·뺄셈 전에 signed int 캐스팅:

```python
        else:
            scale = (n_bins - 1) / (int(high) - int(low))
            offset = -int(low) * scale
```

`int(high)-int(low)`도 함께 캐스팅해 uint8 뺄셈 언더플로 여지를 없앤다(현재는 `if high<=low` 가드로 항상 양수라 실질 무해하나 방어적). 그 외 로직·클리핑은 불변.

## 4. 테스트 계획

- **정합성**: 실제 coco 이미지 표본에 수정판 `autocontrast_func` vs `PIL.ImageOps.autocontrast` 비교 → 평균/최대 픽셀 차이 ≈ 0. (검증 완료: n=300에서 mean/max diff = 0.000.)
- **회귀**: 수정 후 `l1_smoke.py` 재실행 시 `randaugment.py:31` overflow 경고가 사라지는지 확인.
- **불변**: `low==0` 케이스(원래 정상 경로) 출력이 수정 전후 동일한지 스팟 확인.

## 5. 범위 밖 (YAGNI)

- `equalize_func` 등 다른 aug 함수의 유사 uint8 취약성 전수 감사 — 이번 이슈는 실측으로 확인된 AutoContrast에 한정.
- `warnings.filterwarnings("ignore")`(`models/blip.py:9`) 자체 제거/완화 — 별개 결정. 이 수정으로 해당 경고 원인은 제거되나 필터 정책은 건드리지 않음.
