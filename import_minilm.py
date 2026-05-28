import torch
from transformers import AutoModel, BertModel, BertConfig

def show_me_the_boundary():
    hub_id = "microsoft/MiniLM-L12-H384-uncased"

    print("모델 로드 및 세팅 중...\n")
    pure_model = AutoModel.from_pretrained(hub_id)
    
    config = BertConfig.from_pretrained(hub_id)
    config.add_cross_attention = True
    config.is_decoder = True
    mod_model = BertModel.from_pretrained(hub_id, config=config, add_pooling_layer=False)
    
    orig_vocab = pure_model.config.vocab_size # 30522
    
    # 토큰 2개 추가! (새로운 크기: 30524)
    mod_model.resize_token_embeddings(orig_vocab + 2)

    pure_emb = pure_model.embeddings.word_embeddings.weight.detach()
    mod_emb = mod_model.embeddings.word_embeddings.weight.detach()

    print("="*65)
    print(f"🎯 임베딩 경계선 집중 확인 (기존: {orig_vocab} / 변경: {orig_vocab+2}) 🎯")
    print("="*65)

    # 1. 기존 단어장의 마지막 2개 토큰 (인덱스 30520, 30521)
    # 이 부분은 순정 모델과 개조된 모델의 숫자가 완벽히 같아야 함!
    print("\n[1] 기존 마지막 토큰 2개 (숫자가 일치해야 정상)")
    for i in [orig_vocab - 2, orig_vocab - 1]:
        print(f"--- 인덱스 {i} ---")
        print(f"순정: {[round(x, 6) for x in pure_emb[i, :5].tolist()]}")
        print(f"개조: {[round(x, 6) for x in mod_emb[i, :5].tolist()]}")

    # 2. 새로 추가된 2개 토큰 (인덱스 30522, 30523)
    # 순정 모델에는 이 인덱스가 아예 없고, 개조된 모델은 무작위(Random) 값으로 채워져 있어야 함!
    print("\n[2] 새로 추가된 토큰 2개 (개조 모델에만 존재, 랜덤 값)")
    for i in [orig_vocab, orig_vocab + 1]:
        print(f"--- 인덱스 {i} ---")
        print(f"순정: (원본 모델은 이 인덱스가 없습니다)")
        print(f"개조: {[round(x, 6) for x in mod_emb[i, :5].tolist()]}")

    print("\n" + "="*65)
    print("💡 인덱스 30521까지는 숫자가 똑같고, 30522부터 새로운 숫자가 찍히는지 직접 확인해 봐!")
    print("="*65)

import torch
from transformers import AutoModel, BertModel, BertConfig

def show_me_the_numbers():
    hub_id = "microsoft/MiniLM-L12-H384-uncased"

    print("모델 로드 및 세팅 중 (잠시만 기다려주세요)...\n")
    pure_model = AutoModel.from_pretrained(hub_id)
    
    config = BertConfig.from_pretrained(hub_id)
    config.add_cross_attention = True
    config.is_decoder = True
    mod_model = BertModel.from_pretrained(hub_id, config=config, add_pooling_layer=False)
    
    orig_vocab_size = pure_model.config.vocab_size
    mod_model.resize_token_embeddings(orig_vocab_size + 2)

    print("="*60)
    print("👀 두 눈으로 직접 확인하는 가중치 리얼 비교 👀")
    print("="*60)

    # 1. 임베딩 첫 단어(0번 인덱스)의 앞 5개 숫자 비교
    pure_emb = pure_model.embeddings.word_embeddings.weight.detach()
    mod_emb = mod_model.embeddings.word_embeddings.weight.detach()
    
    print("\n[1] 임베딩 레이어 (0번 단어 가중치 첫 5개 값)")
    # 보기 편하게 리스트로 변환하고 소수점 6자리까지만 출력
    print(f"순정 MiniLM: {[round(x, 6) for x in pure_emb[0, :5].tolist()]}")
    print(f"개조된 모델: {[round(x, 6) for x in mod_emb[0, :5].tolist()]}")

    # 2. 0번 레이어 (첫 번째 층) Query 가중치 비교
    pure_l0 = pure_model.encoder.layer[0].attention.self.query.weight.detach()
    mod_l0 = mod_model.encoder.layer[0].attention.self.query.weight.detach()
    
    print("\n[2] 0번 레이어(첫 층) Query 가중치 (첫 5개 값)")
    print(f"순정 MiniLM: {[round(x, 6) for x in pure_l0[0, :5].tolist()]}")
    print(f"개조된 모델: {[round(x, 6) for x in mod_l0[0, :5].tolist()]}")

    # 3. 11번 레이어 (마지막 층) Output Dense 가중치 비교
    pure_l11 = pure_model.encoder.layer[11].output.dense.weight.detach()
    mod_l11 = mod_model.encoder.layer[11].output.dense.weight.detach()
    
    print("\n[3] 11번 레이어(마지막 층) Output 가중치 (첫 5개 값)")
    print(f"순정 MiniLM: {[round(x, 6) for x in pure_l11[0, :5].tolist()]}")
    print(f"개조된 모델: {[round(x, 6) for x in mod_l11[0, :5].tolist()]}")

    print("\n" + "="*60)
    print("💡 위아래 숫자가 음수 부호, 소수점 끝자리까지 100% 똑같은지 확인해 보세요!")
    print("="*60)

if __name__ == "__main__":
    show_me_the_numbers()

# if __name__ == "__main__":
#     show_me_the_boundary()