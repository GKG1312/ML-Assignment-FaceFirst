from time import time
import torch
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
model.load_state_dict(torch.load("./frcnn_model_03.pth"))
model.to("cuda")
model.eval()

cap = cv2.VideoCapture('/home/sp5/Desktop/ML_Assignments/Facefirst/vid02.mp4')
while cap.isOpened():
    start = time()
    ret, frame = cap.read()
    height, width, _ = frame.shape
    frame = cv2.resize(frame, (width//4, height//4))
    original = frame
    frame = frame/255
    frame = np.expand_dims(frame, axis=0).transpose(0,3,1,2)
    # frame = cv2.resize(frame, ())
    # transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    with torch.inference_mode():
        result = model(torch.tensor(frame, dtype=torch.float).cuda())
    
    if len(result[0]['scores']) != 0:
        # print(result[0]['scores'])
        bbox = result[0]['boxes'][torch.argmax(result[0]['scores'])]
        label = result[0]['labels'][torch.argmax(result[0]['scores'])]
        conf = result[0]['scores'][torch.argmax(result[0]['scores'])]
        # print(result[0]['boxes'][torch.argmax(result[0]['scores'])])
        # for (box, label, conf)in zip(bbox, label, conf):
        # print(box, label, conf)
        xmin, ymin, xmax, ymax = list(bbox.cpu().numpy())
        # print(int(xmin), int(ymin), int(xmax), int(ymax))
        cv2.rectangle(original, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color=(255,0,255), thickness=1)
        text = f"{label_dict[label.item()]} : {round(conf.item(), 2)}"
        cv2.putText(original, text, (int(xmin), int(ymin)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,0,255), 1, cv2.LINE_AA)

        cv2.imshow('Screen', original)
        if cv2.waitKey(5) & 0xff==ord('x'):
            break
        if cv2.getWindowProperty("Screen", cv2.WND_PROP_VISIBLE)<1:
            break

        end = time()
        fps = 1/(end-start)
        print(fps)

cap.release()
cv2.destroyAllWindows()

# img = cv2.imread("/home/sp5/Desktop/ML_Assignments/Facefirst/test/MASKED/M002.jpg")
# img = (img/255)
# img = np.expand_dims(img, axis=0).transpose(0,3,1,2)
# print(img.shape)
# result = model(torch.tensor(img, dtype=torch.float))
# # cv2.imshow('Screen', np.squeeze(result.render()))
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
# print(result)