# custom file - vg dataset 전처리 구조
import json
import os
from pathlib import Path
from tqdm import tqdm

# 문자열로 루트 폴더 위치를 받음
# img_root_vg = "/home/minwoo/Distillation_Project/datasets/vision/vg/images/"
# annotation_root_vg = "/home/minwoo/Distillation_Project/datasets/vision/vg/annotations/"

def mod_vg_json(annotation_root_vg, annotation_output_vg):
    root_path = Path(annotation_root_vg)
    ann_file = root_path / "region_descriptions.json" # 기본파일명임
    out_path = Path(annotation_output_vg)
    output_file = out_path / "vg_train.json"

    print(f"🍳 VG 데이터 굽기 시작: {ann_file}")
    with open(ann_file, 'r') as f:
        raw_data = json.load(f)

    formatted_data = []
    caption_id = 0 # 캡션 아이디는 우리 마음대로 조정해버리자고 근데 형식을 보니깐 coco_번호 이렇게가네 아니다

    for item in tqdm(raw_data, desc="Flattening VG Regions"): # 디스크립션은 뭘까
        # 추론 결과 로우데이터는 하나의 이미지에 엄청나게 많은 수의 캡션을 닮
        image_id = item['id'] # 이미지 번호따오기
        img_name = f"{image_id}.jpg" # 그대로긴함
        
        for region in item['regions']: # 이건 리스트가 하나 나옴. 그래서 하나의 리전당 캡션, 리전 아이디 위치, 이미지 아이디, phrase(caption) 가 나옴
            # 딕셔너리 형태로 저장하기
            formatted_data.append({
                "image_id": image_id,
                "id": caption_id,
                "image": img_name,
                "caption": region['phrase'].strip(),
                # VG는 캡션이 이미지 전체가 아니라 region 하나를 가리키므로 bbox를 같이 들고가서 로더에서 crop 한다
                "x": region['x'],
                "y": region['y'],
                "width": region['width'],
                "height": region['height'],
                # "dataset_source": "vg" # 데이터로더 라우팅을 위한 꼬리표 이건나중에 한번에 하는게?
            })
            caption_id += 1

    with open(output_file, 'w') as f:
        json.dump(formatted_data, f)
    print(f"✨ VG 완료! 총 {len(formatted_data)}개의 캡션 쌍이 {output_file}에 저장되었습니다.\n")

# 저 라벨 다는건 json에다가 적으면 제일 좋을 것 같긴 한데? 아니다 그대로 두고, 기존 자원을 최대한 활용을 하자
# 데이터셋 단계에서 로딩이 끝나면 그때 추가적으로 데이터셋 소스를 달아버리자고

if __name__ == "__main__":
    # 1. 파일 경로 설정 (실제 환경에 맞게 수정해주세요)
    VG_RAW_FILE = "/home/minwoo/Distillation_Project/datasets/vision/vg/annotations/"
    VG_OUT_FILE = "/home/minwoo/Distillation_Project/datasets/vision/vg/annotations/"
    # VG 파일이 있다면 실행
    if os.path.exists(VG_RAW_FILE):
        mod_vg_json(VG_RAW_FILE, VG_OUT_FILE)
    else:
        print(f"⚠️ {VG_RAW_FILE} 파일을 찾을 수 없습니다.")