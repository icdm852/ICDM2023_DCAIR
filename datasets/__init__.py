from .beauty import BeautyDataset
from .music import MusicDataset

DATASETS = {
    BeautyDataset.code(): BeautyDataset,
    MusicDataset.code(): MusicDataset,
}

def dataset_factory(args):
    dataset = DATASETS[args.dataset_code]
    return dataset(args)
