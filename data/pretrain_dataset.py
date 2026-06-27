import json
import os
import random

from torch.utils.data import Dataset

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from data.utils import pre_caption
import os,glob

class pretrain_dataset(Dataset):
    def __init__(self, ann_file, laion_path, img_root_coco, img_root_vg, transform, teacher_cache=None):

        self.ann_pretrain = []
        dataset_len = []
        for f in ann_file: # 리스트 안에 두개의 캡션json 경로가 들어가있음.
            print('loading '+f) # 그 경로 출력
            ann = json.load(open(f,'r')) # 이건 리스트인디 결국 json안에다 데이터셋 타입 추가하는수밖에 없나
            # 여기서 미리 딕셔너리 태그를 싹 달아버리자

            # 파일 안에 데이터셋 소스 태깅이 누락되어 있다면 (첫 번째 아이템으로 검사)
            if len(ann) > 0 and 'dataset_source' not in ann[0]:
                print(f"🕵️ '{f}'에서 'dataset_source' 태그를 찾지 못했습니다. 실시간 인메모리 태깅 중...")
                
                # 파일명 규칙에 따라 소스 태그 결정 (예: coco, vg 등)
                source_tag = 'coco' if 'coco' in f.lower() else 'vg' # 경로니깐 coco혹은 vg가 적혀있을것
                for item in ann:
                    item['dataset_source'] = source_tag

            self.ann_pretrain += ann # 옆에다 붙여서 넣는 방법이네 오케이
            dataset_len += [len(ann)]
        # 테스트코드
        print(f"0번 소스타입: {self.ann_pretrain[0]['dataset_source']}") # 아마 coco일거임
        # print(f"1번 소스타입: {self.ann_pretrain[dataset_len[0]]['dataset_source']}") # 아마 vg일거임
        print(f"coco dataset size: {dataset_len[0]}")
        # print(f"vg dataset size: {dataset_len[1]}")
        
        self.laion_path = laion_path
        if self.laion_path:
            self.laion_files = glob.glob(os.path.join(laion_path,'*.json'))

            print('loading '+self.laion_files[0])
            with open(self.laion_files[0],'r') as f:
                self.ann_laion = json.load(f)  

            self.annotation = self.ann_pretrain + self.ann_laion
        else:
            self.annotation = self.ann_pretrain
            
        self.transform = transform
        self.teacher_cache = teacher_cache
        self.img_root_coco = img_root_coco
        self.img_root_vg = img_root_vg

        # 이건 여기서 선언 가능
        self.dataset_root_dict = {
            "laion": laion_path,
            "coco": img_root_coco,
            "vg": img_root_vg
        }

    def reload_laion(self, epoch):
        n = epoch%len(self.laion_files)
        print('loading '+self.laion_files[n])
        with open(self.laion_files[n],'r') as f:
            self.ann_laion = json.load(f)      
        
        self.annotation = self.ann_pretrain + self.ann_laion    
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        ann = self.annotation[index] # 해당 번째의 아노테이션을 당겨온다. 이는 리스트로 안에 이미지 캡션 쌍이 주르르
        img_root = self.dataset_root_dict[ann['dataset_source']] # coco / vg / laion이 들어올 것 -> 맞는 경로 루트를 준다

        image_path = os.path.join(img_root, ann['image']) # 진짜 이미지 파일이 있는 경로 만들기
        # ann['image']: 'val2014/COCO_val2014_000000522418.jpg' 코코예시
        # ann['image']: '1.jpg', 'caption' vg예시 (x,y,width,height: caption이 가리키는 region)

        # 그러면 해당 루트 경로를 받아서
        image = Image.open(image_path).convert('RGB')

        if ann['dataset_source'] == 'vg':
            # VG는 캡션이 이미지 전체가 아니라 특정 region을 설명하므로 전체 이미지 대신 해당 region만 crop
            img_w, img_h = image.size
            x0 = min(max(ann['x'], 0), img_w)
            y0 = min(max(ann['y'], 0), img_h)
            x1 = min(max(ann['x'] + ann['width'], 0), img_w)
            y1 = min(max(ann['y'] + ann['height'], 0), img_h)
            if x1 > x0 and y1 > y0: # 일부 region이 좌표가 음수거나 범위를 벗어나는 경우가 있어 방어
                image = image.crop((x0, y0, x1, y1))

        image = self.transform(image)
        caption = pre_caption(ann['caption'],30)

        if self.teacher_cache is not None:
            img_feat_t, txt_feat_t = self.teacher_cache.get(index)
            return image, caption, img_feat_t, txt_feat_t
        return image, caption # 반출