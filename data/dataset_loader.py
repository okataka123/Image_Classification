import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, MNIST
from torch.utils.data import DataLoader


def _get_dataset(dataset_name, transform, train=True):
    '''
    Args:
        train (bool): 学習時True, 推論時False
    '''
    if dataset_name == 'CIFAR10':
        dataset = CIFAR10(root='~/datasets', train=train, download=True, transform=transform)
    elif dataset_name == 'MNIST':
        dataset = MNIST(root='~/datasets', train=train, download=True, transform=transform)
    else:
        raise ValueError(f'Unknown dataset {dataset_name}')
    return dataset


def get_data_loader(dataset_name, batch_size, train):
    '''
    Args:
        train (bool): 学習時True, 推論時False
    '''
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = _get_dataset(dataset_name, transform, train=train)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
