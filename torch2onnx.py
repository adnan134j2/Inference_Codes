import torch
from torch.autograd import Variable

from torchvision import datasets, models, transforms
import torch.nn as nn
import onnx
from onnxsim import simplify

model = models.resnet50().cuda()

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 3).cuda().eval()
input_size = 224
pytorch_pth = 'C:\\pytorch_image_classifier\\Lagu-Model-1-8\\model.pth'
onnx_out = 'C:\\pytorch_image_classifier\\Lagu-Model-1-8\\model.onnx'
model.load_state_dict(torch.load(pytorch_pth))

dummy_input = Variable(torch.randn(1, 3, 224, 224))
torch.onnx.export(model, dummy_input.cuda(), onnx_out,
                    input_names=["input"],
                    output_names=["output"])
onnx_model = onnx.load(onnx_out)
model_simp, check = simplify(onnx_model)
assert check, "Simplified ONNX model could not be validated"
onnx.save(model_simp, onnx_out)
print('finished exporting onnx')
print("ending    ")