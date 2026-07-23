# additional cc12m dataset pretraining
import os
import glob
import webdataset as wds
from torch.utils.data import IterableDataset
from data.utils import pre_caption


class cc12m_webdataset(IterableDataset):
    def __init__(self, tar_root, transform, batch_size, shardshuffle_size=100):
        super().__init__()
        self.tar_root = tar_root
        self.transform = transform
        self.batch_size = batch_size
        
        # 1. tar 파일 목록 확보 및 정렬
        tar_pattern = os.path.join(tar_root, "*.tar")
        self.tar_files = sorted(glob.glob(tar_pattern))
        
        if not self.tar_files:
            raise FileNotFoundError(f"지정된 경로에 tar 파일이 없습니다: {tar_root}")
            
        print(f"✅ [Class] CC12M WebDataset 로드 완료: 총 {len(self.tar_files)}개 샤드")

        # 2. 내부 WebDataset 스트리밍 파이프라인 구성
        self.pipeline = (
            wds.WebDataset(
                self.tar_files,
                shardshuffle=shardshuffle_size,
                nodesplitter=wds.split_by_node,  # 멀티 GPU 분할 가속
                handler=wds.warn_and_continue    # 에러 방어선
            )
            .shuffle(5000)
            .decode("pil")
            .to_tuple("jpg", "txt")
            .map_tuple(self.transform, self.process_text)  # 내장 텍스트 전처리 메서드 연결
            .batched(self.batch_size, partial=False)       # 배치 묶기 - 묶어서 전달하는게 확실히 빠르다고함.
        )

    def process_text(self, raw_text):
        """바이트 데이터를 문자열로 세타 가공하는 내부 메서드"""
        if isinstance(raw_text, bytes):
            text = raw_text.decode('utf-8')
        else:
            text = str(raw_text)
        return pre_caption(text, max_words=30)

    def __iter__(self):
        """
        🌟 핵심: DataLoader가 데이터를 요청할 때 
        내부 WebDataset 파이프라인의 이터레이터를 다이렉트로 넘겨줍니다.
        """
        return iter(self.pipeline)