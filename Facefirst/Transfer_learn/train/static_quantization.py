# import torch
# import torchvision
# from timm import create_model
# from dataloader import *
# from models import *



# class QuantModel(torch.nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         self.model = model
#         self.quant = torch.quantization.QuantStub()
#         self.dequant = torch.quantization.DeQuantStub()

#     def forward(self, x):
#         x = self.quant(x)
#         # print(x.size())
#         x = self.model(x)
#         # x = self.dequant(x)
#         return x

    

# def fuse_resnet_modules(model):
#     # Fuse Conv1, BN1, and ReLU in the backbone
#     torch.quantization.fuse_modules(model, [["conv1", "bn1", "act1"]], inplace=True)

#     # Fuse Conv2 and BN2 in the ResNet layers
#     for layer_name, layer in model.named_children():
#         if "layer" in layer_name:
#             for block in layer:
#                 torch.quantization.fuse_modules(block, [["conv1", "bn1", "act1"], ["conv2", "bn2", "act2"]], inplace=True)

#     return model

# def calibrate_model(model, loader, device=torch.device("cpu:0")):

#     model.to(device)
#     model.eval()

#     for imgs, _ in loader:
#         imgs = imgs.to('cpu')
#         # print(type(imgs))
#         _ = model(imgs)

# def get_model_instance(model_arch):
#     if model_arch=='custom_resnet':     
#         model = ResNet(ResidualBlock, [1, 2, 2, 1])
#     elif model_arch=='custom_fcn':
#         model = CNNmodel()
#     elif model_arch=='resnet18':
#         model = create_model(model_arch, pretrained=True, num_classes=4, in_chans=3 )
#     elif model_arch=='mobilenetv2_050':
#         model = create_model(model_arch, pretrained=True, num_classes=4, in_chans=3 )
#     elif model_arch=='efficientnet_b0':
#         model = create_model(model_arch, pretrained=True, num_classes=4, in_chans=3 )

#     return model    
# model_names = ['resnet18', 'efficientnet_b0', 'mobilenetv2_050']

# model = get_model_instance(model_names[0])
# model.load_state_dict(torch.load("./saved_models/resnet18/resnet18_test_None.pth"))
# # print(model.named_children)
# # model.to("cpu")
# model.eval()
# # torch.quantization.fuse_modules(model, ['conv1', 'bn1', 'act1'], inplace=True)
# # torch.quantization.fuse_modules(model, ['conv2', 'bn2', 'act1'], inplace=True)
# model = fuse_resnet_modules(model)
# quant_model = QuantModel(model)
# # quant_model = nn.Sequential(torch.quantization.QuantStub(),
# #                             *model,
# #                             torch.quantization.DeQuantStub())

# labeledData, I2F, F2I = loadDataset("./Train", balancing=None)
# trainData, otherData = splitArray(labeledData, 0.7, shuffle=True)
# valData, testData = splitArray(otherData, split=0.5, shuffle=True)

# trainTransforms = torchvision.transforms.Compose([
#     # torchvision.transforms.Resize((224, 224)),
#     torchvision.transforms.ToTensor(),
#     torchvision.transforms.ConvertImageDtype(torch.float32),
#     # torchvision.transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
#     torchvision.transforms.RandomHorizontalFlip(),
#     torchvision.transforms.RandomPerspective(),
# ])

# valTransforms = torchvision.transforms.Compose([
#     # torchvision.transforms.Resize((224, 224)),
#     torchvision.transforms.ToTensor(),
#     torchvision.transforms.ConvertImageDtype(torch.float32),
#     # torchvision.transforms.Normalize([0.449], [0.226]),
# ])

# trainDataset = ImageDataset(trainData, trainTransforms)
# valDataset = ImageDataset(valData, valTransforms)
# testDataset = ImageDataset(testData, valTransforms)

# train_loader = torch.utils.data.DataLoader(trainDataset, batch_size=32, shuffle=True)
# val_loader = torch.utils.data.DataLoader(valDataset, batch_size=32, shuffle=True)
# test_loader = torch.utils.data.DataLoader(testDataset, batch_size=8, shuffle=False)
# dataloaders = {"train": train_loader, "val": val_loader, "test":test_loader}

# model.qconfig = torch.quantization.get_default_qconfig("x86")
# torch.backends.quantized.engine = "x86"
# model_staticQuantize = torch.quantization.prepare(model, inplace=True)
# calibrate_model(model_staticQuantize, test_loader)
# model_staticQuantize = torch.quantization.convert(model_staticQuantize, inplace=True)
# torch.jit.save(torch.jit.script(model_staticQuantize), "./saved_models/resnet18/resnet18_none_StaticQuant_01.pth")

# # torch.save(model_staticQuantize.state_dict(), './saved_models/resnet18/resnet18_none_StaticQuant_02.pth')



import torch
import copy
import argparse
import torchvision
from torch.utils.data import DataLoader
from dataloader import *
from models import *
from pytorchUtils import save_model_q, evaluate_qmodel, visualize_predictions
from timm import create_model
from torch.ao.quantization import get_default_qconfig
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization import QConfigMapping

def test(qmodel, test_loader):
    correct = 0
    total = 0
    
    qmodel.eval()

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = qmodel(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy of the quantized model on the test dataset: {accuracy:.2f}%")


def quantized_model (model_arch, label, model_path, data_path, sampling_type=None):
    if model_arch=='custom_resnet':     
        model = ResNet(ResidualBlock, [1, 2, 2, 1])
    elif model_arch=='custom_fcn':
        model = CNNmodel()
    elif model_arch=='resnet18':
        model = create_model(model_arch, pretrained=True, num_classes=4, in_chans=3 )
    elif model_arch=='mobilenetv2_050':
        model = create_model(model_arch, pretrained=True, num_classes=4, in_chans=3 )
    elif model_arch=='efficientnet_b0':
        model = create_model(model_arch, pretrained=True, num_classes=4, in_chans=3 )
        
    model.load_state_dict(torch.load(model_path))
    new_model = copy.deepcopy(model)

    labeledData, I2F, F2I = loadDataset(data_path)
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
    test_loader = DataLoader(testDataset, batch_size=32, shuffle=False)
    dataloaders = {"train": train_loader, "val": val_loader, "test":test_loader}


    new_model.eval()

    qconfig = get_default_qconfig("x86")
    qconfig_mapping = QConfigMapping().set_global(qconfig)

    example_inputs = torch.randn(4,3,224,224)

    prepared_model = prepare_fx(new_model, qconfig_mapping, example_inputs)
    print(prepared_model.graph)

    def calibrate(model, data_loader):
        with torch.inference_mode():
            for image, target in data_loader:
                model(image)
    calibrate(prepared_model, dataloaders['val'])  # run calibration on sample data

    quantized_model = convert_fx(prepared_model)
    print(quantized_model)
    # model = torch.ao.quantization.fuse_modules(model, [['conv1', 'bn1', 'relu1']])

    # model.qconfig = torch.ao.quantization.get_default_qconfig('x86')
    # model = torch.ao.quantization.fuse_modules(model, [['conv1', 'bn1', 'ReLU1']])
    # qmodel = torch.ao.quantization.prepare(model, inplace=False)

    # # input_dummy = torch.randn(4,3,224,224)
    # # qmodel(input_dummy)
    # qmodel.eval()
    # for batch, target in val_loader:
    #     model(batch.to('cpu'))
    
    # torch.quantization.convert(qmodel, inplace=True)

    # test(qmodel, val_loader)

    mname = f"{model_arch}_{label}_FXGquant"
    save_model_q(quantized_model, model_arch, mname, sampling_type)
    acc = evaluate_qmodel(quantized_model, dataloaders['test'])
    visualize_predictions(quantized_model, dataloaders['test'],model_arch, sampling_type, 20, quantized=True)
    print("#### Successfully saved quantized model ####")

    
    # return quantized_model


    # if arch == 'resnet18':
    #     model = create_model(arch, pretrained=True, num_classes=4, in_chans=3 )
    # elif arch =='resnet':
    #     model = ResNet(ResidualBlock, [1, 2, 2, 1])
    # else:
    #     model = CNNmodel()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", "-a", type=str, default="resnet18", help="Model architecture")
    parser.add_argument("--label", "-l", type=str, default="test", help="Model label")
    parser.add_argument("--saved_model", "-m", type=str, required=True, help="Saved Model path")
    parser.add_argument("--dataset", "-d", type=str, required=True, help="Dataset path")
    parser.add_argument("--sampling_type", "-sm", type=str, default=None, help="Sampling Type")
    args = parser.parse_args()
    
    quantized_model(args.arch, args.label, args.saved_model, args.dataset, args.sampling_type)


