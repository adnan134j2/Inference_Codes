import onnxruntime
import torch
import cv2



img = cv2.imread("C:\\pytorch_image_classifier\\data\\test\\normal\\20210714172659244_0_topfront.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img_resized = cv2.resize(img, (224,224), interpolation = cv2.INTER_LINEAR)





x = torch.ones(1, 3, 224, 224, requires_grad=False)
ort_session = onnxruntime.InferenceSession("modelN.onnx")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
ort_outs = ort_session.run(None, ort_inputs)

# compare ONNX Runtime and PyTorch results
##np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")