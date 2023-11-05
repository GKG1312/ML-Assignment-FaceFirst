import copy
import json
import random
import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25, earlyStopping=None, history=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val']}

    best_model_wts = copy.deepcopy(model.state_dict())

    if history is None:
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'best_epoch': 0,'val_loss_best': 1e10, 'val_acc_best': 0, 'prev_epochs': 0}

    best_loss = history['val_loss_best']
    prev_epochs = history['prev_epochs']

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            running_size = 0

            # Iterate over data.
            with tqdm(dataloaders[phase], unit="batch") as t:
                t.set_description("Training" if (phase == 'train') else "Validation")

                # Loop over batches
                for inputs, labels in t:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        logits = model(inputs)
                        # print(f"####### {labels} ####")
                        # preds = torch.max(torch.sigmoid(logits), dim=1)
                        # print(f"####### {preds.size()} ####")
                        # loss = criterion(preds, labels)
                        # print(f"######## {logits[0]} ########")
                        preds = torch.argmax(logits, dim=1)
                        loss = criterion(logits, labels.long().squeeze())

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels)
                    running_size += inputs.size(0)

                    epoch_loss = running_loss / running_size
                    epoch_acc = running_corrects.double() / running_size

                    t.set_postfix(loss=f"{epoch_loss:.4f}", accuracy=f"{epoch_acc*100:.2f}%")

            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())

            if phase == 'val':
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())

        
        scheduler.step(history['val_loss'][-1])

        # After train & val deep copy the best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            history['best_epoch'] = epoch + prev_epochs + 1
            history['val_loss_best'] = epoch_loss
            history['val_acc_best'] = epoch_acc.item()
            best_model_wts = copy.deepcopy(model.state_dict())

        # Check early stopping
        if earlyStopping is not None:
            if earlyStopping.step(history):
                print("Early stopping")
                break

        print()

    print(f'Best Validation Accuracy: {best_loss:4f}')
    history['prev_epochs'] = epoch + prev_epochs + 1

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, history

def save_model(model, model_arch, path, sampling_type):
    torch.save(model.state_dict(), f"./saved_models/{model_arch}/{path}_{sampling_type}.pth")
    # if history is not None:
    #     json.dump(history, open(f"./saved_model/{model_arch}/{path}_hist.json", 'w'))

def save_model_q(model, model_arch, path, sampling_type):
    # save with script
    torch.jit.save(torch.jit.script(model), f"./saved_models/{model_arch}/{path}_{sampling_type}.pth")
    # loaded_quantized_model = torch.jit.load(fx_graph_mode_model_file_path)
    # torch.save(model.state_dict(), f"./saved_models/{model_arch}/{path}.pt")
    # if history is not None:
    #     json.dump(history, open(f"./saved_model/{model_arch}/{path}_hist.json", 'w'))

def freeze(model):
    for param in model.parameters():
        param.requires_grad = False
    
    model.fc.weight.requires_grad = True
    model.fc.bias.requires_grad = True

def unfreeze(model):
    for param in model.parameters():
        param.requires_grad = True

def evaluate_model(model, test_loader):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.eval()
        
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        print(f"Test Accuracy: {accuracy:.4f}")

        return accuracy

def evaluate_qmodel(model, test_loader):
        device = "cpu"
        model.eval()
        
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        print(f"Test Accuracy: {accuracy:.4f}")

        return accuracy

class EarlyStopping:
    def __init__(self, patience, min_delta=0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0

    def step(self, history):
        if (history['val_loss'][-1] - history['train_loss'][-1]) > self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.counter = 0
        return False
    
def visualize_predictions(model, test_loader, model_arch, sampling_type, num_images=15, quantized=False):
        if quantized:
            device = "cpu"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model.eval()
        
        count=0
        all_preds = []
        all_labels = []
        all_images = []

        fig, axes = plt.subplots(4,5, figsize=(15,8))

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_images.extend(inputs.cpu().numpy())
                count+=1
                if count==num_images:
                    break

        for i in range(num_images):
            image = all_images[i]
            label = all_labels[i]
            pred = all_preds[i]

            row = i//5
            col = i%5
            ax = axes[row, col]

            # Visualize the image with true label and predicted label
            # plt.figure()
            # plt.subplot(4,5, i+1)
            
            ax.imshow(np.transpose(image, (1, 2, 0)))
            ax.axis('off')
            ax.set_title(f"{label} : {pred}")
            # plt.xlabel(f"Pred: {pred}")

        fig.suptitle('Labels = [0:SUNGLASSES, 1:MASKED, 2:FACES, 3:MASKwithSUNGLASSES]', fontsize=13)
        plt.tight_layout()
        plt.savefig(f"./saved_models/{model_arch}/{model_arch}_{sampling_type}_prediction_images.png")
        plt.show()