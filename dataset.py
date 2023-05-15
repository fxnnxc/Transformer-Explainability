import torch 
import torchvision 
import torchvision.transforms as T
import os


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

CIFAR100_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR100_STD  = [0.2023, 0.1994, 0.2010] 

CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD  = [0.2023, 0.1994, 0.2010]

MNIST_MEAN = [0.1307]
MNIST_STD  = [0.3081] 

CUB_MEAN = [0.4862, 0.5005, 0.4334]
CUB_STD = [0.2321, 0.2277, 0.2665]
    
def get_img_datasets(name, data_path, transform=None, use_default_transform=True,):
    # ---- Define the wrapper if required -----
    if transform is None and use_default_transform:
        if name in ['cifar10', 'cifar100', 'mnist', 'fashion_mnist']:
            mean, std = {
                "cifar10": [CIFAR10_MEAN, CIFAR10_STD],
                "cifar100": [CIFAR100_MEAN, CIFAR100_STD],
                "mnist": [MNIST_MEAN, MNIST_STD],
                "fashion_mnist": [MNIST_MEAN, MNIST_STD], # incorrect!
                "mnist": [MNIST_MEAN, MNIST_STD],
            }[name]
            transform = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        elif 'imagenet' in name:
            transform = T.Compose([
                            T.Resize(256),
                            T.CenterCrop(224),
                            T.ToTensor(), 
                            T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
                        ])
        elif 'cub' in name:
            transform = T.Compose([
                            T.Resize(224),
                            T.CenterCrop(224),
                            T.ToTensor(), 
                            T.Normalize(CUB_MEAN, CUB_STD)
                        ])
        

    # ------ CIFAR ---------
    if name =="cifar10": # sadsad
        train_dataset  = torchvision.datasets.CIFAR10(root=data_path,  
                                                    train=True,  
                                                    download=True,
                                                    transform=transform) 
        valid_dataset  = torchvision.datasets.CIFAR10(root = data_path,  
                                                    train=False,  
                                                    download=True,
                                                    transform =transform) 
        test_dataset = None


    elif name =="cifar100":
        train_dataset  = torchvision.datasets.CIFAR100(root=data_path,  
                                                    train=True,  
                                                    download=True,
                                                    transform=transform) 
        valid_dataset  = torchvision.datasets.CIFAR100(root = data_path,  
                                                    train=False,  
                                                    download=True,
                                                    transform =transform) 
        test_dataset = None
    elif name =="cub": # sadsad
        from .cub import CUB2011
        train_dataset  = CUB2011(root=data_path,  
                                train=True,  
                                download=True,
                                transform=transform) 
        valid_dataset  = CUB2011(root = data_path,  
                                                    train=False,  
                                                    download=True,
                                                    transform =transform) 
        test_dataset = None
    # ------ ImageNet ---------
    elif name =="imagenet1k":
        # train_dataset = torchvision.datasets.ImageNet(root=data_path, split="train", transform=transform)
        train_dataset = torchvision.datasets.ImageNet(root=data_path, split="train", transform=transform)
        valid_dataset = torchvision.datasets.ImageNet(root=data_path, split="val", transform=transform)
        test_dataset = None
        # test_dataset = torchvision.datasets.ImageNet(root=data_path, split="test", transform=transform)
    
    elif name == "mnist":
        train_dataset = torchvision.datasets.MNIST(root=data_path, train=True, transform=transform, download=True)
        valid_dataset = torchvision.datasets.MNIST(root=data_path, train=False, transform=transform, download=True) 
        test_dataset = None
        
    elif name == "fashion_mnist":
        train_dataset = torchvision.datasets.FashionMNIST(root=data_path, train=True, transform=transform, download=True)
        valid_dataset = torchvision.datasets.FashionMNIST(root=data_path, train=False, transform=transform, download=True) 
        test_dataset = None
    else:
        raise ValueError(f"{name} is not implemented data")
    return train_dataset, valid_dataset, test_dataset, transform


def get_imagenet_image_boundary():
    
    a = torch.zeros(3,224,224)
    b = torch.zeros(3,224,224).fill_(1.0)
    
    min_img = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)(a)
    max_img = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)(b)
    
    return min_img, max_img



def denormalize(tensor):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    denormalized = tensor.clone()

    for channel, mean, std in zip(denormalized, means, stds):
        channel.mul_(std).add_(mean)

    return denormalized
