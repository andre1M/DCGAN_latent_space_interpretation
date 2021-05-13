from modules.global_vars import DATA_PATH, ITER_METHODS

from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from omegaconf import DictConfig


def get_dataloader(cfg: DictConfig) -> DataLoader:
    """
    Download and pre-process the dataset.

    :param cfg:
    :return:
    """

    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Resize(32)
        # transforms.Normalize(mean=[0.1307], std=[0.3081]),
        # transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    if cfg.model.name in ITER_METHODS:
        batch_size = cfg.model.num_samples
    else:
        batch_size = cfg.batch_size

    dataset = datasets.MNIST(
        root=DATA_PATH,
        train=True,
        transform=transform,
        download=True
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    return dataloader
