from time import time
import torch
import onnx
import onnxruntime as ort
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import cv2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


label_dict = {0:"Faces",
              1:"Masked",
              2:"Glasses",
              3:"Mask&Glass"}

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='FasterRCNN_ResNet50_FPN_Weights.COCO_V1')
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
# model = attempt_load("./runs/train/exp3/weights/best.pt")
model = get_model_instance_segmentation(4)
map_location = lambda storage, loc: storage
if torch.cuda.is_available():
    map_location = None
model.load_state_dict(torch.load("./frcnn_model_03.pth", map_location))
model.to("cuda")
model.eval()

test_img1 = cv2.imread("/home/sp5/Desktop/ML_Assignments/Facefirst/test/MASKwithSUNGLASSES/MS004.jpg")
test_img2 = cv2.imread("/home/sp5/Desktop/ML_Assignments/Facefirst/test/MASKwithSUNGLASSES/MS001.jpg")
test_img = np.stack([test_img1/255, test_img2/255], axis=0)
# print(test_img.shape, type(test_img))
test_img = torch.tensor(test_img.transpose(0,3,1,2), dtype=torch.float)
test_img = test_img.cuda()
x = torch.randn(32, 3,112,112, requires_grad=True).cuda()
with torch.inference_mode():
    out = model(x)

torch.onnx.export (model, x, "frcnn_ONNX_BS32.onnx",
                   export_params=True, opset_version=17,
                   do_constant_folding=True,
                   input_names = ["input"],
                   output_names = ["output"],
                   )




onnx_model = onnx.load("frcnn_ONNX_BS32.onnx")
# print(onnx_model)
onnx.checker.check_model(onnx_model)



sess_options = ort.SessionOptions()
# Below is for optimizing performance
sess_options.intra_op_num_threads = 24
sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
ort_session = ort.InferenceSession("frcnn_ONNX_BS32.onnx", sess_options=sess_options)

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
ort_outs = ort_session.run(None, ort_inputs)


print(out)
print(ort_outs)
# compare ONNX Runtime and PyTorch results
# np.testing.assert_allclose(to_numpy(out[0]), ort_outs[0], rtol=1e-03, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")