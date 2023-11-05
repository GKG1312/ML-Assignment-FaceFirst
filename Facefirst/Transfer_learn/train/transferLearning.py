import os
import torch
from torch.utils.data import DataLoader
import torchvision
import torchinfo
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from timm import create_model
from train.extra_utils import imshow_cus
from train.models import *
from train.miscUtils import plot_graphs
from train.model_quant import quantized_model
from train.dataloader import ImageDataset, loadDataset, splitArray
from train.pytorchUtils import EarlyStopping, save_model, train_model
from train.pytorchUtils import evaluate_model, visualize_predictions

def transferLearn(model_arch, model_label, sampling_type, *paths):
    model_name = f"{model_arch}_{model_label}"

    # data_dir = "./RealWorldOccludedFaces/images"
    # Get a list of all folders in the current directory
    class_names = os.listdir(*paths)

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
    torchinfo.summary(model, (1,3,224,224))

    # folders = ['neutral', 'sunglasses'] #'masked'

    labeledData, I2F, F2I = loadDataset(*paths, balancing=sampling_type)
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

    # # Get a batch of training data
    # inputs, classes = next(iter(dataloaders['train']))

    # # Make a grid from batch
    # out = torchvision.utils.make_grid(inputs, nrow=4)
    # # print((classes.numpy().astype('int8')))

    # fig, ax = plt.subplots(1, figsize=(10, 10))
    # imshow_cus(out, title=[class_names[x] for x in classes.numpy().astype('int8')], ax=ax)

    loss_func = torch.nn.CrossEntropyLoss()

    # freeze(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=3, verbose=True)
    earlyStopping = EarlyStopping(5)
    model, history = train_model(model, dataloaders, loss_func, optimizer, lr_scheduler, num_epochs=100, earlyStopping=earlyStopping)

    # unfreeze(model)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=3, verbose=True)
    # earlyStopping = EarlyStopping(5)
    # model, history = train_model(model, dataloaders, loss_func, optimizer, lr_scheduler, num_epochs=100, earlyStopping=earlyStopping, history=history)

    save_model(model, model_arch, model_name, sampling_type)

    plot_graphs(history, model_arch, model_name, sampling_type)

    accuracy = evaluate_model(model, dataloaders['test'])
    visualize_predictions(model, dataloaders['test'], model_arch, sampling_type, 20)

    model_path = f"./saved_models/{model_arch}/{model_name}_{sampling_type}.pth"

    quantized_model(model_path, model_arch, model_label, dataloaders, sampling_type)
    # accuracy = evaluate_model(qmodel, dataloaders['test'])
    # visualize_predictions(qmodel, dataloaders['test'])

    # qmodel_name = f"{model_name}_quant"

    # save_model(qmodel, model_arch, qmodel_name)


