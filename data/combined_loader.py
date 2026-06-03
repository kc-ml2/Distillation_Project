# cc12m webdataset and coco dataset combined dataloader
class CombinedLoader:
    def __init__(self, loader_map, loader_iterable, ratio=5):
        self.loader_map = loader_map
        self.loader_iterable = loader_iterable
        self.total_steps = len(loader_map) * (1+ratio)  # 에포크 길이를 기존 로더의 2배로 정의
        self.cc12m_batch_count = 0
        self.ratio = ratio
    
    @property
    def sampler(self):
        return self.loader_map.sampler

    def __iter__(self):
        iter_map = iter(self.loader_map)
        iter_iterable = iter(self.loader_iterable)
        
        for batch_map in iter_map:
            yield batch_map  # 기존 데이터 (COCO/LAION 등) 1배치 출고
            
            for _ in range(self.ratio): # 5번 반복해서 cc12m출고
                try:
                    yield next(iter_iterable)  # CC12M WebDataset 1배치 출고
                    self.cc12m_batch_count += 1
                except StopIteration:
                    # CC12M을 다 썼으면 다시 처음부터 이터레이터를 열어서 스트리밍 재개
                    iter_iterable = iter(self.loader_iterable)
                    self.cc12m_batch_count += 1
                    yield next(iter_iterable)

    def __len__(self):
        return self.total_steps # 기존 길이의 2배로 정의
    
    def get_cc12m_status(self):
        return self.cc12m_batch_count
