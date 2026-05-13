# vit base를 다운해서 돌린 결과
# Load Result: _IncompatibleKeys(missing_keys=[], unexpected_keys=['head.weight', 'head.bias'])
# Base model output shape: torch.Size([1, 197, 768])
# Vision Width: 768

# 최신 모델의 경우 Output shape: torch.Size([1, 201, 384])
# 스몰은 384로 고정이 된다. 즉, 224/16 = 14고 14*14 = 196이다. 토큰이 그럼 5개인듯?
# 대신 출력 차원이 384로 고정이 되어있다는 것.

# vit_output_shape.py
import torch
import timm

# BLIP 전체를 안 불러오고 timm 모델만 먼저 확인
model = timm.create_model(
    model_name='vit_small_patch16_dinov3.lvd1689m',
    # model_name='vit_small_plus_patch16_dinov3.lvd1689m',
    pretrained=True,
    img_size=224,
    num_classes=0,
    global_pool=''
)
def test_dinov3_model():
    model = timm.create_model(
        model_name='vit_small_patch16_dinov3.lvd1689m',
        # model_name='vit_small_plus_patch16_dinov3.lvd1689m',
        pretrained=True,
        img_size=224,
        num_classes=0,
        global_pool=''
    )
    test_input = torch.randn(1, 3, 224, 224)
    test_output = model(test_input)
    print(f"Output shape: {test_output.shape}")

def test_dinov3_wrapper_model():
    from models.blip_pretrain import DINOv3_Wrapper
    raw_model = timm.create_model(
        model_name='vit_small_patch16_dinov3.lvd1689m',
        # model_name='vit_small_plus_patch16_dinov3.lvd1689m',
        pretrained=True,
        img_size=224,
        num_classes=0,
        global_pool=''
    )
    wrapper_model = DINOv3_Wrapper(base_model=raw_model, model_register_tokens=4)
    test_input = torch.randn(1, 3, 224, 224)
    test_output = wrapper_model(test_input)
    print(f"Output shape: {test_output.shape}")


# Output shape: torch.Size([1, 384]) -> vit small 경우 w/o gloabl pool option
# Output shape: torch.Size([1, 201, 384]) -> vit small w global pool ''
# 결과가 [1, 384]인지, [1, 197, 384]인지 바로 확인 가능!

# 스몰은 384로 고정이 된다. 즉, 224/16 = 14고 14*14 = 196이다. 토큰이 그럼 5개인듯?
# 대신 출력 차원이 384로 고정이 되어있다는 것.


#### 베이스 생성 코드
# vit == 'base' 케이스 안에서 실행할 테스트 코드
from models.blip import create_vit

def test_base_model():
    print("--- Testing DeiT-Base Model ---")
    
    # 1. 모델 구조 생성 (self 없이 로컬 변수로)
    # vit='base', image_size=224 등 기존 인자값 그대로 사용
    base_model, vision_width = create_vit(vit='base', image_size=224)
    
    # 2. 가중치 다운로드 및 로드
    url = "https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth"
    checkpoint = torch.hub.load_state_dict_from_url(
        url=url,
        map_location="cpu", 
        check_hash=True
    )
    
    state_dict = checkpoint["model"]
    
    # strict=False로 로드하고 결과(msg) 확인
    msg = base_model.load_state_dict(state_dict, strict=False)
    print(f"Load Result: {msg}")

    # 3. 테스트 인풋 먹이기
    test_input = torch.randn(1, 3, 224, 224)
    
    base_model.eval() # 추론 모드로 변경
    with torch.no_grad():
        test_output = base_model(test_input)
    
    print(f"Base model output shape: {test_output.shape}")
    print(f"Vision Width: {vision_width}")

if __name__ == "__main__":
    # test_base_model()
    # test_dinov3_model()
    test_dinov3_wrapper_model()

