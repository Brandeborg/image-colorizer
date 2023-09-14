from os import sep
import torch

from torchvision import datasets, transforms
import matplotlib.pyplot as plt

def load_image_data():
    def crop_left(image):
        return transforms.functional.crop(image, 0, 0, 1600, 1040)
    
    transform = transforms.Compose([transforms.Lambda(crop_left),
                                    transforms.Resize(1040),
                                    transforms.ToTensor()])
    dataset = datasets.ImageFolder(f"dataset", transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=57, shuffle=True)

    imgs, labs = next(iter(dataloader))
    img = imgs[2]
    img = img.permute(1,2,0)
    plt.imshow(img)
    plt.show()


    
load_image_data()