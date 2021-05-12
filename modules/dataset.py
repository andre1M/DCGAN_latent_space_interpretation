from modules.global_vars import DATA_PATH

from torch.utils.data import DataLoader
from torchvision import transforms, datasets


def get_dataloader(batch_size: int) -> DataLoader:
    """
    Download and pre-process the dataset.

    :param batch_size:
    :return:
    """

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(32)
        # transforms.Normalize(mean=[0.1307], std=[0.3081]),
        # transforms.Normalize(mean=[0.5], std=[0.5])
    ])
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
