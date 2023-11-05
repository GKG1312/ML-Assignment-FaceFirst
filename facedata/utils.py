import torch
from tqdm import tqdm
import copy
import numpy as np
import pandas as pd

def split_data(csv_path):
    data = pd.read_csv(csv_path, header=None)
    # print(data)
    data = data.sample(frac=1, ignore_index=True)
    data = data.drop(columns=0, )
    # print(data)
    train_size = int(len(data)*0.7)
    val_split =int((len(data)-train_size)*0.5)

    train_set = data[:train_size].reset_index(drop=True)
    train_set = train_set.set_axis(range(train_set.shape[1]), axis=1)
    val_set = data[train_size:train_size+val_split].reset_index(drop=True)
    val_set = val_set.set_axis(range(val_set.shape[1]), axis=1)
    test_set = data[train_size+val_split:].reset_index(drop=True)
    test_set = test_set.set_axis(range(test_set.shape[1]), axis=1)
#     print(train_set, val_set, test_set)

# split_data("./files.csv")
    
    return train_set, val_set, test_set

def train_model(model, dataloaders, optimizer, scheduler, num_epochs=25, earlyStopping=None, history=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
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
            # running_corrects = 0
            running_size = 0

            # Iterate over data.
            with tqdm(dataloaders[phase], unit="batch") as t:
                t.set_description("Training" if (phase == 'train') else "Validation")

                val_losses = []
                # Loop over batches
                # print(t)
                for inputs, targets in t: 
                    # print(inputs.shape)
                    inputs = list(image.to(device) for image in inputs)
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    # print(targets)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    # print(inputs[0].size(),'####\n', targets[0])
                    with torch.set_grad_enabled(phase == 'train'):
                        loss_dict = model([inputs[0]], [targets[0]])
                        # print("### loss dict ###", loss_dict)
                        if phase == "train":
                            losses = sum(loss for loss in loss_dict.values())
                        else:
                            val_losses.append(loss_dict)
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            losses.backward()
                            optimizer.step()

                    # statistics
                    running_loss += losses.item() * len(inputs)
                    # running_corrects += torch.sum(preds == labels)
                    running_size += len(inputs)

                    epoch_loss = running_loss / running_size
                    # epoch_acc = running_corrects.double() / running_size

                    t.set_postfix(loss=f"{epoch_loss:.4f}") 

            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                # history['train_acc'].append(epoch_acc.item())

            if phase == 'val':
                # print("### Val losses ###\n", val_losses)
                val_losses = [sum(loss for loss in loss_dict.values()) for loss_dict in val_losses]
                epoch_val_loss = sum(val_loss.item() for val_loss in val_losses) / len(val_losses)
                history['val_loss'].append(epoch_val_loss)
                # history['val_acc'].append(epoch_acc.item())

        
        scheduler.step(history['val_loss'][-1])

        # After train & val deep copy the best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            history['best_epoch'] = epoch + prev_epochs + 1
            history['val_loss_best'] = epoch_loss
            # history['val_acc_best'] = epoch_acc.item()
            best_model_wts = copy.deepcopy(model.state_dict())

        # Check early stopping
        if earlyStopping is not None:
            if earlyStopping.step(history):
                print("Early stopping")
                break

    print(f'Best Validation loss: {best_loss:4f}')
    history['prev_epochs'] = epoch + prev_epochs + 1

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, history

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
    

def generate_box(line, img_shape):
    _, x, y, width, height = list(line)
    xmin = x-width/2
    ymin = y-height/2
    xmax = x+width/2
    ymax = y+height/2

    width, height, _ = list(img_shape)

    return [int(xmin*width), int(ymin*height), int(xmax*width), int(ymax*height)]

def generate_label(line):
    label = line[0]
    return label

def generate_target(index, file, img_shape): 
    lines = []
    with open(file) as f:
        for label in f.readlines():
            class_label, x, y, width, height = [
                float(x) if float(x) != int(float(x)) else int(float(x))
                for x in label.replace("\n", "").split()
            ]
            lines.append([class_label, x, y, width, height])

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        labels = []
        for i in lines:
            boxes.append(generate_box(i, img_shape))
            labels.append(generate_label(i))
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # Labels (In my case, I only one class: target class or background)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        # Tensorise img_id
        img_id = torch.tensor([index])
        # Annotation is in dictionary format
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = img_id
        
        return target
    

class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0
        
    def send(self, value):
        self.current_total += value
        self.iterations += 1
    
    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations
    
    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0



def train_model_2(model, dataloaders, optimizer, scheduler, num_epochs=25, earlyStopping=None, history=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val']}

    best_model_wts = copy.deepcopy(model.state_dict())

    if history is None:
        history = {'train_loss': [],  'val_loss': [], 'best_epoch': 0,'val_loss_best': 1e10,  'prev_epochs': 0}

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

            # Iterate over data.
            with tqdm(dataloaders[phase], unit="batch") as t:
                t.set_description("Training" if (phase == 'train') else "Validation")
                # Loop over batches
                for inputs, targets in t:
                    inputs = list(image.to(device) for image in inputs)
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    
                    if phase=='train':
                        # zero the parameter gradients
                        optimizer.zero_grad()
                        loss_dict = model([inputs[0]], [targets[0]])
                        losses = sum(loss for loss in loss_dict.values())
                        loss_value = losses.item()
                        losses.backward()
                        optimizer.step()
                    else:
                        with torch.no_grad():
                            loss_dict = model([inputs[0]], [targets[0]])
                        losses = sum(loss for loss in loss_dict.values())
                        loss_value = losses.item()

                    epoch_loss = loss_value

                    t.set_postfix(loss=f"{epoch_loss:.4f}")

            if phase == 'train':
                history['train_loss'].append(epoch_loss)

            if phase == 'val':
                history['val_loss'].append(epoch_loss)

        
        scheduler.step(history['val_loss'][-1])

        # After train & val deep copy the best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            history['best_epoch'] = epoch + prev_epochs + 1
            history['val_loss_best'] = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())

        # Check early stopping
        if earlyStopping is not None:
            if earlyStopping.step(history):
                print("Early stopping")
                break

        print()

    print(f'Best Validation loss: {best_loss:4f}')
    history['prev_epochs'] = epoch + prev_epochs + 1

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, history