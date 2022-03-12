import numpy as np
import torch
from torch.autograd import Variable
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import torch.nn as nn
import time
model = models.resnet50().cpu()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 7).cpu().eval()
input_size = 224
model_path = 'ModelMondayWeek3.pth'
model.load_state_dict(torch.load('ModelMondayWeek3.pth'))
model.eval()

transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    )

dataset = ImageFolder(""'./data/test/'"", transform=transform)
print(dataset.class_to_idx)
num_classes = len(dataset.classes)
dataloader = DataLoader(dataset=dataset, batch_size=10, shuffle=True, num_workers=0)

example_image, example_label = next(iter(dataloader))
traced_script_module = torch.jit.trace(model, example_image.cpu())

outputtt = traced_script_module(torch.ones(1,3,224,224).cpu())
print(outputtt)
traced_script_module.save("traced_Classify.pt")

Ofull= []
Tfull=[]
for data, target in dataloader:
    data, target = data.to("cpu"), target.to("cpu")
    data_var = Variable(data)
    model.eval()
    
    t1=time.time() * 1000

    output = model(data_var).cpu()
    t2=time.time()* 1000
    print(t2-t1)
    target=target.cpu().tolist()
    Opart = [np.argmax(i) for i in output.detach().cpu().numpy()]
    Ofull.extend(Opart)

    Tfull.extend(target)


Tfull = np.array(Tfull)
Ofull = np.array(Ofull)

c = [i for i in Ofull - Tfull if i == 0]
print('Validation Accuracy: {}%\n'.format((len(c) / len(Ofull)) * 100))