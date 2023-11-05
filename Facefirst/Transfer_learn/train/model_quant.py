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


def quantized_model (model_path, model_arch, label, dataloaders, sampling_type):
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
    # new_m= copy.deepcopy(model)

    model.eval()

    qconfig = get_default_qconfig("x86")
    qconfig_mapping = QConfigMapping().set_global(qconfig)

    example_inputs = torch.randn(4,3,224,224)

    prepared_model = prepare_fx(model, qconfig_mapping, example_inputs)
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


