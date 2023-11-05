import torch
import torch.nn as nn
from torchvision import models
from torch.ao.quantization import QuantStub, DeQuantStub

class CNNmodel(nn.Module):
    def __init__(self):
        super(CNNmodel, self).__init__()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        self.conv1 = nn.Conv2d(3,32,(3,3))
        self.max1 = nn.MaxPool2d(2,2)

        self.conv2 = nn.Conv2d(32,64,(3,3))
        self.max2 = nn.MaxPool2d(2,2)

        self.conv3 = nn.Conv2d(64,128,(3,3))
        self.max3 = nn.MaxPool2d(2,2)

        self.conv4 = nn.Conv2d(128,256,(3,3))
        self.max4 = nn.MaxPool2d(2,2)

        self.conv5 = nn.Conv2d(256,512,(3,3))
        self.max5 = nn.MaxPool2d(2,2)

        self.conv6 = nn.Conv2d(512,512,(3,3))
        self.max6 = nn.MaxPool2d(2,2)

        # self.conv7 = nn.Conv2d(512,512,(3,3))

        self.fc = nn.Linear(512,4)

        # self.batch_size = batch_size

    def forward(self, images):
        conv_out1 = self.conv1(images)
        max_out1 = self.max1(conv_out1)
        conv_out2 = self.conv2(max_out1)
        max_out2 = self.max2(conv_out2)
        conv_out3 = self.conv3(max_out2)
        max_out3 = self.max3(conv_out3)
        conv_out4 = self.conv4(max_out3)
        max_out4 = self.max4(conv_out4)
        conv_out5 = self.conv5(max_out4)
        max_out5 = self.max5(conv_out5)
        conv_out6 = self.conv6(max_out5)
        max_out6 = self.max6(conv_out6)
        # conv_out7 = self.conv7(max_out6)


        out = max_out6.view(max_out6.size(0), -1)
        linear_out = self.fc(out)
        # final_out = self.softmax(linear_out)

        return linear_out

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
    
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes = 4):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3),
                        nn.BatchNorm2d(64),
                        nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride = 2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride = 2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride = 2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    
class PreTrainedModels:
    def __init__(self, model_name, num_classes):
        self.model_name = model_name
        self.num_classes = num_classes

    def create_model(self):
        if self.model_name == 'resnet18':
            model = models.resnet18(pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, self.num_classes)
        elif self.model_name == 'mobilenet':
            model = models.mobilenet_v2(pretrained=True)
            model.classifier[1] = nn.Linear(model.last_channel, self.num_classes)
        elif self.model_name == 'efficientnet':
            model = models.efficientnet_b0(pretrained=True)
            model.classifier[1] = nn.Linear(in_features=1280, out_features=self.num_classes)
        else:
            raise ValueError("Invalid model name")

        return model