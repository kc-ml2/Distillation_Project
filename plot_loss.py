import re
import numpy as np
import matplotlib.pyplot as plt

# 1. 파일 이름 설정
log_file = 'current_log.txt'
# log_file = 'output/Pretrain/log.txt'

# 2. 로스 값을 담을 빈 리스트 준비
loss_ita_list = []
loss_itm_list = []
loss_lm_list = []

# 3. 정규식(Regex) 패턴: 터미널 로그 라인에서 세 가지 로스 숫자만 정확히 낚아챕니다.
pattern = re.compile(r"loss_ita:\s+([\d\.]+)\s+loss_itm:\s+([\d\.]+)\s+loss_lm:\s+([\d\.]+)")
# pattern = re.compile(
#     r'"train_loss_ita":\s*"?([\d\.]+)"?,\s*'
#     r'"train_loss_itm":\s*"?([\d\.]+)"?,\s*'
#     r'"train_loss_lm":\s*"?([\d\.]+)"?'
# )

print("🔍 로그 파일을 분석하는 중...")
with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        match = pattern.search(line)
        if match:
            loss_ita_list.append(float(match.group(1)))
            loss_itm_list.append(float(match.group(2)))
            loss_lm_list.append(float(match.group(3)))

print(f"✅ 총 {len(loss_ita_list)}개의 스텝 데이터를 찾았습니다. 그래프를 그립니다!")

itm_scale = 10 # 잘 안보여서 스케일 키움
# loss_itm_list = np.array(loss_itm_list)[1200:1400] * itm_scale
# loss_ita_list = np.array(loss_ita_list)[1200:1400]
# loss_lm_list = np.array(loss_lm_list)[1200:1400]
loss_itm_list = np.array(loss_itm_list)
loss_ita_list = np.array(loss_ita_list)
loss_lm_list = np.array(loss_lm_list)
loss_total = loss_ita_list + loss_itm_list + loss_lm_list

# 4. 캔버스 설정 및 그래프 그리기
plt.figure(figsize=(12, 6))

# 데이터가 너무 많아 그래프가 지저분해지는 걸 막기 위해 투명도(alpha)를 살짝 줍니다.
plt.plot(loss_ita_list, label='Loss ITA (Contrastive)', color='blue', alpha=0.7, linewidth=0.5)
plt.plot(loss_itm_list, label='Loss ITM (Matching)', color='green', alpha=0.9, linewidth=0.5)
plt.plot(loss_lm_list, label='Loss LM (Language Model)', color='orange', alpha=0.7, linewidth=0.5)
plt.plot(loss_total, label='Loss ToT (sum of all losses)', color='red', alpha=0.7, linewidth=0.5)

# 5. 디자인 (제목, 축, 격자무늬)
plt.title('Training Loss Trend (Over 13 Hours)', fontsize=16, fontweight='bold')
plt.xlabel('Training Steps(x50)', fontsize=12)
plt.ylabel('Loss Value', fontsize=12)
plt.legend(loc='upper right', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)

# 6. 이미지 파일로 저장 (서버 환경 필수)
output_filename = 'loss_curve.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"🎉 성공! '{output_filename}' 파일이 생성되었습니다.")