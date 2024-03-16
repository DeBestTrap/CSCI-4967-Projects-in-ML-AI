import torch
import random
import numpy as np
import torchvision.utils as vutils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.use_deterministic_algorithms(True) # Needed for reproducible results
    # if (torch.backends.cudnn.version() != None and manualSeed != None):
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = False

def load_data(dataroot:str, image_size:int, augment:bool = False):
    dataset = dset.ImageFolder(root=dataroot,
                            transform=transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))

    if augment:
        print("Augmenting dataset")
        augmented_dataset = dset.ImageFolder(root=dataroot,
                                transform=transforms.Compose([
                                    transforms.Resize(image_size),
                                    transforms.CenterCrop(image_size),
                                    transforms.RandomHorizontalFlip(p=0.1),
                                    transforms.RandomRotation(degrees=10),
                                    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                                    transforms.GaussianBlur(kernel_size=3),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ]))
        classes = dataset.classes
        # classes = dataset.classes + augmented_dataset.classes
        targets = dataset.targets + augmented_dataset.targets
        dataset = torch.utils.data.ConcatDataset([dataset, augmented_dataset])
        dataset.classes = classes
        dataset.targets = targets
        
    return dataset

def plot_grid(images, title:str = "Grid", nrow:int=None):
    '''
    Plot a grid of images of shape (batch_size, channels, height, width)
    '''
    if not nrow:
        nrow = int(np.sqrt(images.shape[0]))
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title(title)
    plt.imshow(np.transpose(vutils.make_grid(images, padding=2, normalize=True, nrow=nrow).cpu(),(1,2,0)))
    plt.show()

def get_sample_images(dataloader, n_images:int = 25):
    # Sample some images
    real_batch = next(iter(dataloader))
    images = real_batch[0][:n_images]
    targets = real_batch[1][:n_images]
    labels = [dataloader.dataset.classes[target] for target in targets]
    return images, targets, labels
