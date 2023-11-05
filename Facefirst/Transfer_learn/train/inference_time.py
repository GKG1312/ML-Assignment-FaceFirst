import torch 
import time
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from timm import create_model
from dataloader import ImageDataset, loadDataset, splitArray

# model_arch = "resnet18" 

# if model_arch=='resnet18':
#     model = create_model(model_arch, pretrained=True, num_classes=4, in_chans=3 )
# elif model_arch=='mobilenetv2_050':
#     model = create_model(model_arch, pretrained=True, num_classes=4, in_chans=3 )
# elif model_arch=='efficientnet_b0':
#     model = create_model(model_arch, pretrained=True, num_classes=4, in_chans=3 )
# # torchinfo.summary(model, (1,3,224,224))

# folders = ['neutral', 'sunglasses'] #'masked'

labeledData, I2F, F2I = loadDataset("./Train", )
trainData, otherData = splitArray(labeledData, 0.7, shuffle=True)
valData, testData = splitArray(otherData, split=0.5, shuffle=True)

trainTransforms = torchvision.transforms.Compose([
    # torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.ConvertImageDtype(torch.float32),
    # torchvision.transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomPerspective(),
])

valTransforms = torchvision.transforms.Compose([
    # torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.ConvertImageDtype(torch.float32),
    # torchvision.transforms.Normalize([0.449], [0.226]),
])

trainDataset = ImageDataset(trainData, trainTransforms)
valDataset = ImageDataset(valData, valTransforms)
testDataset = ImageDataset(testData, valTransforms)

train_loader = DataLoader(trainDataset, batch_size=32, shuffle=True)
val_loader = DataLoader(valDataset, batch_size=32, shuffle=True)
test_loader = DataLoader(testDataset, batch_size=1, shuffle=False)
dataloaders = {"train": train_loader, "val": val_loader, "test":test_loader}

for inputs, labels in dataloaders['test']:
    inputs = inputs.to("cpu")
    labels = labels.to("cpu")
    break

# model_paths = ["./saved_models/resnet18/resnet18_test_None.pth", "./saved_models/resnet18/resnet18_test_quant_None.pth",
#                "./saved_models/resnet18/resnet18_test_under_sampling.pth", "./saved_models/resnet18/resnet18_test_quant_under_sampling.pth",
#                "./saved_models/resnet18/resnet18_test_over_sampling.pth", "./saved_models/resnet18/resnet18_test_quant_over_sampling.pth",
#                "./saved_models/efficientnet_b0/efficientnet_b0_test_None.pth", "./saved_models/efficientnet_b0/efficientnet_b0_test_quant_None.pth",
#                "./saved_models/efficientnet_b0/efficientnet_b0_test_under_sampling.pth", "./saved_models/efficientnet_b0/efficientnet_b0_test_quant_under_sampling.pth",
#                "./saved_models/efficientnet_b0/efficientnet_b0_test_over_sampling.pth", "./saved_models/efficientnet_b0/efficientnet_b0_test_quant_over_sampling.pth",
#                "./saved_models/mobilenetv2_050/mobilenetv2_050_test_None.pth", "./saved_models/mobilenetv2_050/mobilenetv2_050_test_quant_None.pth",
#                "./saved_models/mobilenetv2_050/mobilenetv2_050_test_under_sampling.pth", "./saved_models/mobilenetv2_050/mobilenetv2_050_test_quant_under_sampling.pth",
#                "./saved_models/mobilenetv2_050/mobilenetv2_050_test_over_sampling.pth", "./saved_models/mobilenetv2_050/mobilenetv2_050_test_quant_over_sampling.pth"]
# time_taken = {}

model_paths = ["./saved_models/resnet18/resnet18_test_None.pth", "./saved_models/resnet18/resnet18_test_FXGquant_None.pth"]
def plot_result(inputs, predictions, targets, model_name, ttime):
    # print("=="*10)
    # print(model_name)
    # print("=="*10)

    fig, axes = plt.subplots(1,1, figsize=(15,8))
    
    for i, (img, pred, lbl) in enumerate(zip(inputs, predictions, targets)):
        pred = pred.cpu().numpy()
        lbl = lbl.cpu().numpy()
        img = img.cpu().numpy()

        # col = i
        ax = axes

        ax.imshow(np.transpose(img, (1,2,0)))
        ax.axis('off')
        ax.set_title(f"{lbl} : {pred}")
    fig.supxlabel(f"Inference Time : {ttime}")
    fig.suptitle(model_name)
    plt.tight_layout()
    plt.show()


# models = np.vstack([resnet_model_paths, efficientnet_model_paths, mobilenet_model_paths])
# print(models[0,0])
for i, path in enumerate(model_paths):
    dir_tree = path.split("/")
    model_name = dir_tree[2]
    # print(model_name)
    if i%2==0:
        model = create_model(model_name, pretrained=True, num_classes=4, in_chans=3 )
        model.load_state_dict(torch.load(path))
        model.to("cpu")
        model.eval()
        start = time.time()
        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
        end = time.time()
        ttime = round(((end-start))*1000,2)
        plot_result(inputs, preds, labels, dir_tree[3], ttime)
        # time_taken[dir_tree[3]] = round(((end-start)/len(inputs))*1000,2)
    else:
        model_q = create_model(model_name, pretrained=True, num_classes=4, in_chans=3 )
        model_q = torch.jit.load(path)
        model_q.to("cpu")
        model_q.eval()
        start_q = time.time()
        with torch.no_grad():
            outputs = model_q(inputs)
            _, preds = torch.max(outputs, 1)
        end_q = time.time()
        ttime = round(((end_q-start_q))*1000,2)
        plot_result(inputs, preds, labels, dir_tree[3], ttime)
        # time_taken[dir_tree[3]] = round(((end_q-start_q)/len(inputs))*1000,2)

# for key, value in time_taken.items():
#     print("#####################################")
#     print(f"{key} : {value}")


