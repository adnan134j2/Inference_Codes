

import torch
from torch.autograd import Variable

from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.nn.functional as F

device = "cuda"


model = models.resnet50().cuda()

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2).cuda().eval()
input_size = 224




model.load_state_dict(torch.load('Lagu-Model-12-31\\model.pth'))
model.eval()



transform = transforms.Compose(
        [
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    )

dataset = ImageFolder("D:\\ALL-EVERYTHING-AOUT-LAGU-FROM-11-29\\Training-Data-With-Special-Class\\train", transform=transform)
print(dataset.class_to_idx)
num_classes = len(dataset.classes)
dataloaders = DataLoader(dataset=dataset, batch_size=12, shuffle=True, num_workers=0)

nb_classes = 2
confusion_matrix = torch.zeros(nb_classes, nb_classes)
with torch.no_grad():
    for i, (inputs, classes) in enumerate(dataloaders):
        inputs = inputs.to(device)
        classes = classes.to(device)
        outputs = model(inputs)
        #print(outputs)
        _, preds = torch.max(outputs, 1)
        for t, p in zip(classes.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

print(confusion_matrix)
print(confusion_matrix.diag()/confusion_matrix.sum(1))