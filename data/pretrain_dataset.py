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
    def __init__(self, ann_file, laion_path, img_root_coco, img_root_vg, transform): 

        self.ann_pretrain = []
        for f in ann_file: # 리스트 안에 두개의 캡션json 경로가 들어가있음.
            print('loading '+f) # 그 경로 출력
            ann = json.load(open(f,'r')) # 이건 리스트인디 결국 json안에다 데이터셋 타입 추가하는수밖에 없나
            # 그게 제일 낫겠다.
            self.ann_pretrain += ann
        
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
        ann = self.annotation[index] # 해당 번째의 아노테이션을 고려한다
        img_root = self.dataset_root_dict[ann['dataset_type']] # coco / vg / laion이 들어올 것 -> 맞는 경로 루트를 준다

        image_path = os.path.join(img_root, ann['image']) # 진짜 이미지 파일이 있는 경로 만들기
        # ann['image']: 'val2014/COCO_val2014_000000522418.jpg' 코코예시
        # ann['image']: '1.jpg', 'caption' vg예시

        # 그러면 해당 루트 경로를 받아서
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)
        caption = pre_caption(ann['caption'],30)
        
        return image, caption # 반출