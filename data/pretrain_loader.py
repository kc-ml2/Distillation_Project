# cc12m 통합 사전학습 데이터로더 팩토리 (일원화)
import webdataset as wds

import utils
from data import create_dataset, create_sampler, create_loader
from data.combined_loader import CombinedLoader


def build_pretrain_dataloader(config, min_scale=0.2):
    """coco+vg base 로더를 만들고, cc12m_tar_path가 있으면 CombinedLoader로 결합해 반환.

    반환 계약(pretrain.py가 의존): 반환 객체는 .sampler(set_epoch 가능)와 len()을
    제공한다. cc12m OFF면 base DataLoader, ON이면 CombinedLoader.

    주의(문서 note): itc_target_mix variant=queue 등 concat_all_gather 경로는
    torch.distributed 초기화가 필요하다 → 반드시 torchrun으로 실행. (assert는 두지 않음)

    주의(에폭 길이 커플링): cc12m ON이면 len(data_loader)가 (1+ratio)배가 된다.
    warmup_steps 등 '절대 step' 하이퍼파라미터는 에폭 길이 변화에 맞춰 재튜닝 필요.
    """
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    base_dataset = create_dataset('pretrain', config, min_scale=min_scale)
    print('number of training samples: %d' % len(base_dataset))
    sampler = create_sampler([base_dataset], [True], num_tasks, global_rank)[0]
    base_loader = create_loader(
        [base_dataset], [sampler],
        batch_size=[config['batch_size']], num_workers=[4],
        is_trains=[True], collate_fns=[None])[0]

    if not config['cc12m_tar_path']:
        return base_loader

    print("Creating cc12m dataset")
    ratio = config['cc12m_ratio']
    cc12m_dataset = create_dataset('pretrain_cc12m_webdataset', config, min_scale=min_scale)
    cc12m_loader = wds.WebLoader(
        cc12m_dataset, batch_size=None, num_workers=4, pin_memory=True)
    # #6 with_epoch 일관 적용(단일 GPU·DDP 공통): rank당 배치수를 len(base)*ratio로 고정.
    #    DDP에서 split_by_node로 rank별 샤드 수가 달라도 collective를 동기화한다.
    cc12m_loader = cc12m_loader.with_epoch(int(len(base_loader) * ratio))

    return CombinedLoader(
        loader_map=base_loader, loader_iterable=cc12m_loader, ratio=ratio)
