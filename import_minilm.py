import torch
from transformers import AutoModel, AutoConfig

def print_model_structure(model_name):
    print(f"\n{'='*20} 구조 출력: {model_name} {'='*20}")
    # MiniLM 자체를 불러옴
    model = AutoModel.from_pretrained(model_name)
    
    # 0번 레이어의 구조를 상세히 출력
    # 여기를 보면 LayerNorm의 위치나 어텐션 블록의 구성을 볼 수 있어
    print(model)

# 순수 MiniLM 구조 확인
print_model_structure("microsoft/MiniLM-L12-H384-uncased")