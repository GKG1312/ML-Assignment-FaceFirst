from utils import Averager
from tqdm.auto import tqdm
import torch
import numpy as np
import torchvision
from dataloader import MaskDataset
from utils import split_data
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time

plt.style.use('ggplot')

def accuracy(preds, annotations):
    acc_count = 0
    calc_number = 0
    total = len(annotations)
    for i in range(total):
        prob = preds[i]['scores'].cpu().detach().numpy()
        pred_label = preds[i]['labels'].cpu().detach().numpy()
        true_label = annotations[i]['labels'].cpu().detach().numpy()
        if len(prob)!=0:
            calc_number+=1
            idx = np.argmax(prob)
            pred = pred_label[idx]
            lbl = true_label
            if pred==lbl:
                acc_count+=1
        
    print(f"Model Accuracy: {acc_count/calc_number} where {total-calc_number} images have no prediction.")

# function for running training iterations
def train(train_data_loader, model):
    print('Training')
    global train_itr
    global train_loss_list
    
     # initialize tqdm progress bar
    prog_bar = tqdm(train_data_loader, total=len(train_data_loader))
    
    for i, data in enumerate(prog_bar):
        optimizer.zero_grad()
        images, targets = data
        
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        train_loss_list.append(loss_value)

        train_loss_hist.send(loss_value)

        losses.backward()
        optimizer.step()

        train_itr += 1
    
        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    return train_loss_list

# function for running validation iterations
def validate(valid_data_loader, model):
    print('Validating')
    global val_itr
    global val_loss_list
    
    # initialize tqdm progress bar
    prog_bar = tqdm(valid_data_loader, total=len(valid_data_loader))
    
    for i, data in enumerate(prog_bar):
        images, targets = data
        
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        with torch.no_grad():
            loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        val_loss_list.append(loss_value)

        val_loss_hist.send(loss_value)

        val_itr += 1

        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    return val_loss_list

if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    train_list, val_list, test_list = split_data("files.csv")
    # print(train_list.iloc[:,0])

    data_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(), 
        ])

    train_dataset = MaskDataset(train_list, "./dataset_detect/images",
                                "./dataset_detect/labels")
    val_dataset = MaskDataset(val_list, "./dataset_detect/images",
                                "./dataset_detect/labels")
    test_dataset = MaskDataset(test_list, "./dataset_detect/images",
                                "./dataset_detect/labels")

    def collate_fn(batch):
        return tuple(zip(*batch))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
    dataloaders = {"train": train_loader, "val": val_loader, "test":test_loader}
    # initialize the model and move to the computation device
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = 5 # 4 classes + background
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    model = model.to(device)
    # get the model parameters
    params = [p for p in model.parameters() if p.requires_grad]
    # define the optimizer
    optimizer = torch.optim.SGD(params, lr=0.0001 )

    # initialize the Averager class
    train_loss_hist = Averager()
    val_loss_hist = Averager()
    train_itr = 1
    val_itr = 1
    # train and validation loss lists to store loss values of all...
    # ... iterations till ena and plot graphs for all iterations
    train_loss_list = []
    val_loss_list = []

    NUM_EPOCHS = 20
    # start the training epochs
    for epoch in range(NUM_EPOCHS):
        print(f"\nEPOCH {epoch+1} of {NUM_EPOCHS}")

        # reset the training and validation loss histories for the current epoch
        train_loss_hist.reset()
        val_loss_hist.reset()

        # create two subplots, one for each, training and validation
        figure_1, train_ax = plt.subplots()
        figure_2, valid_ax = plt.subplots()

        # start timer and carry out training and validation
        start = time.time()
        train_loss = train(dataloaders['train'], model)
        val_loss = validate(dataloaders['val'], model)
        print(f"Epoch #{epoch} train loss: {train_loss_hist.value:.3f}")   
        print(f"Epoch #{epoch} validation loss: {val_loss_hist.value:.3f}")   
        end = time.time()
        print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch}")

        # if (epoch+1) % SAVE_MODEL_EPOCH == 0: # save model after every n epochs
        #     torch.save(model.state_dict(), f"{OUT_DIR}/model{epoch+1}.pth")
        #     print('SAVING MODEL COMPLETE...\n')
        
        # if (epoch+1) % SAVE_PLOTS_EPOCH == 0: # save loss plots after n epochs
        #     train_ax.plot(train_loss, color='blue')
        #     train_ax.set_xlabel('iterations')
        #     train_ax.set_ylabel('train loss')
        #     valid_ax.plot(val_loss, color='red')
        #     valid_ax.set_xlabel('iterations')
        #     valid_ax.set_ylabel('validation loss')
        #     figure_1.savefig(f"{OUT_DIR}/train_loss_{epoch+1}.png")
        #     figure_2.savefig(f"{OUT_DIR}/valid_loss_{epoch+1}.png")
        #     print('SAVING PLOTS COMPLETE...')
        
        if (epoch+1) == NUM_EPOCHS: # save loss plots and model once at the end
            train_ax.plot(train_loss, color='blue')
            train_ax.set_xlabel('iterations')
            train_ax.set_ylabel('train loss')
            valid_ax.plot(val_loss, color='red')
            valid_ax.set_xlabel('iterations')
            # valid_ax.set_ylabel('validation loss')
            # figure_1.savefig(f"{OUT_DIR}/train_loss_{epoch+1}.png")
            # figure_2.savefig(f"{OUT_DIR}/valid_loss_{epoch+1}.png")

            torch.save(model.state_dict(), f"frcnn_trnval_model_{epoch+1}.pth")
        
        plt.close('all')
        # sleep for 5 seconds after each epoch
        # time.sleep(5)
    
    for imgs, annotations in val_loader:
        imgs = list(img.to(device) for img in imgs)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
        break

    model.eval()
    preds = model(imgs)
    accuracy(preds, annotations)