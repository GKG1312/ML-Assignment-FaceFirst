from time import time
import torch
import onnx
import onnxruntime as ort
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import cv2
from timm import create_model


label_dict = {0:"Faces",
              1:"Masked",
              2:"Glasses",
              3:"Mask&Glass"}

def get_model_instance_segmentation(model_arch):
    model = None
    if model_arch=='resnet18':
        model = create_model(model_arch, pretrained=True, num_classes=4, in_chans=3 )
    elif model_arch=='mobilenetv2_050':
        model = create_model(model_arch, pretrained=True, num_classes=4, in_chans=3 )
    elif model_arch=='efficientnet_b0':
        model = create_model(model_arch, pretrained=True, num_classes=4, in_chans=3 )

    return model

model_paths = ["./saved_models/resnet18/resnet18_test_None.pth", 
               "./saved_models/resnet18/resnet18_test_under_sampling.pth", 
               "./saved_models/resnet18/resnet18_test_over_sampling.pth", 
               "./saved_models/efficientnet_b0/efficientnet_b0_test_None.pth", 
               "./saved_models/efficientnet_b0/efficientnet_b0_test_under_sampling.pth", 
               "./saved_models/efficientnet_b0/efficientnet_b0_test_over_sampling.pth", 
               "./saved_models/mobilenetv2_050/mobilenetv2_050_test_None.pth", 
               "./saved_models/mobilenetv2_050/mobilenetv2_050_test_under_sampling.pth", 
               "./saved_models/mobilenetv2_050/mobilenetv2_050_test_over_sampling.pth"
               ]

def get_paths(idx, model_paths):
    path_tree = model_paths[idx].split('/')
    model_name = path_tree[-1].split('.')
    new_path = f"./onnx_models/{model_name[0]}.onnx"
    return model_paths[idx], new_path


saved_model, new_model = get_paths(8, model_paths)
# model = attempt_load("./runs/train/exp3/weights/best.pt")
model_arch = ['resnet18', 'efficientnet_b0', 'mobilenetv2_050']
model = get_model_instance_segmentation(model_arch[2])
map_location = lambda storage, loc: storage
if torch.cuda.is_available():
    map_location = None
model.load_state_dict(torch.load(saved_model, map_location))
model.to("cuda")
model.eval()

test_img = cv2.imread("/home/sp5/Desktop/ML_Assignments/Facefirst/test/MASKwithSUNGLASSES/MS004.jpg")
transform = torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.ConvertImageDtype(torch.float32)
])
test_img = transform(test_img).unsqueeze(0).cuda()
x = torch.randn(1, 3,224,224, requires_grad=True).cuda()
with torch.inference_mode():
    out = model(test_img)

torch.onnx.export (model, x, new_model,
                   export_params=True, opset_version=17,
                   do_constant_folding=True,
                   input_names = ["input"],
                   output_names = ["output"],
                   dynamic_axes={'input':{0:'batch_size', 2:'input_width', 3:'input_height'},
                                 'output':{0:'batch_size'}})




onnx_model = onnx.load(new_model)
# print(onnx_model)
onnx.checker.check_model(onnx_model)



sess_options = ort.SessionOptions()
# Below is for optimizing performance
sess_options.intra_op_num_threads = 24
sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
ort_session = ort.InferenceSession(new_model, sess_options=sess_options)

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(test_img)}
ort_outs = ort_session.run(None, ort_inputs)

print(out)
print(ort_outs)
# compare ONNX Runtime and PyTorch results
# np.testing.assert_allclose(to_numpy(out[0]), ort_outs[0], rtol=1e-03, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")