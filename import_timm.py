import timm
import torch

# BLIP 코드를 거치지 않고 timm에서 직접 로드 테스트
model_name = 'vit_small_patch16_dinov3'

try:
    # pretrained=True로 로드 시도
    model = timm.create_model(model_name, pretrained=True)
    print("성공적으로 로드되었습니다. reg_token 유무:", hasattr(model, 'reg_token'))
except Exception as e:
    print("timm 단독 로드 실패:", e)