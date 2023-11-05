import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import numpy as np
import copy
import os
import random
import torch.quantization
import matplotlib.pyplot as plt

class ClassificationTrainer:
    def __init__(self, model_name, data_dir, num_classes, dataloader, num_epochs):
        self.model_name = model_name
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.dataloader = dataloader
        self.losses = {'train': [], 'val': []}
        self.accuracy = {'train': [], 'val': []}
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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

    def train_model(self):
        model = self.create_model()
        model = model.to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(self.num_epochs):
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in self.dataloader[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # Store the loss for plotting
                self.losses[phase].append(epoch_loss)
                self.accuracy[phase].append(epoch_acc)

                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

        model.load_state_dict(best_model_wts)
        return model

    def plot_loss_curves(self):
        plt.figure()
        plt.plot(range(self.num_epochs), self.losses['train'], label='Training Loss')
        plt.plot(range(self.num_epochs), self.losses['val'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss Curves')
        plt.show()

    def evaluate_model(self, model, test_loader):
        model.eval()
        
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        print(f"Test Accuracy: {accuracy:.4f}")

        return accuracy

    def save_weights(self, model, filename):
        torch.save(model.state_dict(), filename)

    def static_quantization(self, model):
        model.qconfig = torch.quantization.default_qconfig
        model = torch.quantization.quantize(model, inplace=True)
        return model
    
    def visualize_predictions(self, model, test_loader, num_images=15):
        model.eval()
        
        random.seed(42)  # For reproducibility, you can change the seed

        # Get a random subset of test data
        images, labels = zip(*random.sample(test_loader.dataset, num_images))

        images = torch.stack(images).to(self.device)

        with torch.no_grad():
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

        for i in range(num_images):
            image = images[i].cpu().numpy()
            label = labels[i]
            pred = preds[i]

            # Visualize the image with true label and predicted label
            plt.figure()
            plt.imshow(np.transpose(image, (1, 2, 0)))
            plt.title(f"True Label: {label}, Predicted Label: {pred}")
            plt.show()

    def run(self):
        model = self.train_model()
        test_accuracy = self.evaluate_model(model, self.dataloader['test'])
        self.save_weights(model, f'./saved_models/{self.model_name}_trained_weights.pth')
        
        quantized_model = self.static_quantization(model)
        quantized_accuracy = self.evaluate_model(quantized_model, self.dataloader['test'])
        self.save_weights(quantized_model, f'{self.model_name}_quantized_weights.pth')

        # Plot the training and validation loss curves
        self.plot_loss_curves()

        # Visualize predictions
        self.visualize_predictions(model, self.dataloader['test'])

# Example usage:
if __name__ == "__main__":
    trainer = ClassificationTrainer('resnet18', 'path_to_train_data', 4, batch_size=32, num_epochs=10)
    trainer.run()
