from matplotlib import pyplot as plt
import cv2
import numpy as np
import glob
from nets.mobilenet_v2 import MobileNetV2
import torch
from torch.autograd import Variable

from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader



model = MobileNetV2(num_classes=6, input_size=224).to("cuda")
model_path = './checkpoints/checkpoint.pth.tar'
model.load_state_dict(torch.load(model_path)['state_dict'])
model.eval()


transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]
    )

    # write some data loader
dataset = ImageFolder('./data/Cdata/', transform=transform)
print(dataset.class_to_idx)
num_classes = len(dataset.classes)
dataloader = DataLoader(dataset=dataset, batch_size=24, shuffle=True, num_workers=0)

Ofull= []
Tfull=[]
for data, target in dataloader:
    data, target = data.to("cuda"), target.to("cuda")
    data_var = Variable(data)

    output = model(data_var)
    target=target.cpu().tolist()
    Opart = [np.argmax(i) for i in output.detach().cpu().numpy()]
    Ofull.extend(Opart)

    Tfull.extend(target)
    #a = np.array([np.argmax(i) for i in output.detach().cpu().numpy()])
    #b = target.cpu().numpy()
    #print(b)
    #print(a)
    #c = [i for i in a - b if i == 0]
    #print('accuracy: {}%\n'.format((len(c) / len(a)) * 100))

Tfull = np.array(Tfull)
Ofull = np.array(Ofull)

c = [i for i in Ofull - Tfull if i == 0]
print('accuracy: {}%\n'.format((len(c) / len(Ofull)) * 100))
