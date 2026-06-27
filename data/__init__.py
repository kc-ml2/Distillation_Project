import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from data.coco_karpathy_dataset import coco_karpathy_train, coco_karpathy_caption_eval, coco_karpathy_retrieval_eval
from data.nocaps_dataset import nocaps_eval
from data.flickr30k_dataset import flickr30k_train, flickr30k_retrieval_eval
from data.vqa_dataset import vqa_dataset
from data.nlvr_dataset import nlvr_dataset
from data.pretrain_dataset import pretrain_dataset
from distillation.teacher_cache import TeacherCache
from data.pretrain_cc12m_webdataset import cc12m_webdataset # 이것도 웹데이터셋용으로 추가
import glob

from transform.randaugment import RandomAugment

def create_dataset(dataset, config, min_scale=0.5):
    
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    transform_train = transforms.Compose([                        
            transforms.RandomResizedCrop(config['image_size'],scale=(min_scale, 1.0),interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2,5,isPIL=True,augs=['Identity','AutoContrast','Brightness','Sharpness','Equalize',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
            transforms.ToTensor(),
            normalize,
        ])        
    transform_test = transforms.Compose([
        transforms.Resize((config['image_size'],config['image_size']),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
        ])  
        
    if dataset=='pretrain':
        # pretrain_train_aug=false면 학습 입력을 transform_test(결정적)로 고정 (캐싱 distillation용 control)
        use_train_aug = config.get('pretrain_train_aug', True)
        pretrain_transform = transform_train if use_train_aug else transform_test

        teacher_cache = None
        distill_itc = config.get('distill', {}).get('itc', {})
        if distill_itc.get('enabled', False):
            teacher_cache = TeacherCache(distill_itc['cache_dir'])

        dataset = pretrain_dataset(ann_file=config['train_file'],
                                   laion_path=config['laion_path'],
                                   img_root_coco=config['image_root_coco'],
                                   img_root_vg=config['image_root_vg'],
                                   transform=pretrain_transform,
                                   teacher_cache=teacher_cache)
        if teacher_cache is not None:
            teacher_cache.validate_against(len(dataset), config['train_file'])
        return dataset
    
    elif dataset=='pretrain_cc12m_webdataset':
        dataset = cc12m_webdataset(
            tar_root=config['cc12m_tar_path'],
            transform=transform_train,
            batch_size=config['batch_size'] # 이 코드는 여기서 받아와야함.
        )
        return dataset
    
    elif dataset=='caption_coco':   
        train_dataset = coco_karpathy_train(transform_train, config['image_root'], config['ann_root'], prompt=config['prompt'])
        val_dataset = coco_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'], 'val')
        test_dataset = coco_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'], 'test')   
        return train_dataset, val_dataset, test_dataset
    
    elif dataset=='nocaps':   
        val_dataset = nocaps_eval(transform_test, config['image_root'], config['ann_root'], 'val')
        test_dataset = nocaps_eval(transform_test, config['image_root'], config['ann_root'], 'test')   
        return val_dataset, test_dataset   
    
    elif dataset=='retrieval_coco':          
        train_dataset = coco_karpathy_train(transform_train, config['image_root'], config['ann_root'])
        val_dataset = coco_karpathy_retrieval_eval(transform_test, config['image_root'], config['ann_root'], 'val') 
        test_dataset = coco_karpathy_retrieval_eval(transform_test, config['image_root'], config['ann_root'], 'test')          
        return train_dataset, val_dataset, test_dataset    
    
    elif dataset=='retrieval_flickr':          
        train_dataset = flickr30k_train(transform_train, config['image_root'], config['ann_root'])
        val_dataset = flickr30k_retrieval_eval(transform_test, config['image_root'], config['ann_root'], 'val') 
        test_dataset = flickr30k_retrieval_eval(transform_test, config['image_root'], config['ann_root'], 'test')          
        return train_dataset, val_dataset, test_dataset     
    
    elif dataset=='vqa': 
        train_dataset = vqa_dataset(transform_train, config['ann_root'], config['vqa_root'], config['vg_root'], 
                                    train_files = config['train_files'], split='train') 
        test_dataset = vqa_dataset(transform_test, config['ann_root'], config['vqa_root'], config['vg_root'], split='test')
        return train_dataset, test_dataset
    
    elif dataset=='nlvr': 
        train_dataset = nlvr_dataset(transform_train, config['image_root'], config['ann_root'],'train')
        val_dataset = nlvr_dataset(transform_test, config['image_root'], config['ann_root'],'val')
        test_dataset = nlvr_dataset(transform_test, config['image_root'], config['ann_root'],'test')     
        return train_dataset, val_dataset, test_dataset   
    
    
def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset,shuffle in zip(datasets,shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers     


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns, config=None, transform=None):
    loaders = []
    for dataset,sampler,bs,n_worker,is_train,collate_fn in zip(datasets,samplers,batch_size,num_workers,is_trains,collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False

        loader = DataLoader(
            dataset=dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
        loaders.append(loader)

        # # train 상태이고, cc12m경로가 있으면 컴바인드 로더로 업그레이드
        # elif isinstance(dataset, cc12m_webdataset):
        #     cc12m_loader = wds.WebLoader(
        #         dataset,
        #         batch_size=None, # 흠 I/O speed면에선 이게 맞다고함. 이미 데이터셋이 배치사이즈로 묶어서 주는것
        #         num_workers=n_worker,
        #         pin_memory=True
        #     )
        #     import torch.distributed as dist
        #     if dist.is_initialized():
        #         world_size = dist.get_world_size()
        #         num_tars = len(dataset.tar_files)
        #         cc12m_loader = cc12m_loader.ddp_equalize(num_tars // world_size)

        #     loaders.append(cc12m_loader)
        
        # else:
        #     raise ValueError("잘못된 데이터셋")

    return loaders    


