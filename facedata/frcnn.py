import torch
import torchvision
from torch.utils.data import DataLoader

from utils import split_data, train_model_2, EarlyStopping
from dataloader import MaskDataset

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

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)
dataloaders = {"train": train_loader, "val": val_loader, "test":test_loader}




device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 5 # 4 classes + background
in_features = model.roi_heads.box_predictor.cls_score.in_features

model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
model = model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.0001)

# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=3, verbose=True)
earlyStopping = EarlyStopping(5)
model, history = train_model_2(model, dataloaders, optimizer, lr_scheduler, num_epochs=100, earlyStopping=earlyStopping)
