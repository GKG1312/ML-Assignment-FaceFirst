import sys
import torch
import torchvision
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader
from train.models import ResidualBlock, ResNet, CNNmodel
from train.dataloader import loadDataset,splitArray
from train.dataloader import ImageDataset
from timm import create_model
from tqdm import tqdm
from torchmetrics.functional import accuracy

def testModel(arch, model_path, dataset_path):
	device = "cuda" if torch.cuda.is_available() else "cpu"
	if arch == 'resnet18':
		model = create_model(arch, pretrained=True, num_classes=4, in_chans=3 )
	elif arch == 'custom_resnet':
		model = ResNet(ResidualBlock, [1,2,2,1])
	elif arch == 'custom_resnet':
		model = CNNmodel()
	model.load_state_dict(torch.load(model_path, map_location=device))

	model.eval()
	torch.no_grad()

	# labeledPaths = getLabeledPaths(dataset_path="FLIR_ADAS_v2/images_thermal_val/")
	# labeledPaths = loadLabeledDataset(dataset_path)
	labeledPaths = loadDataset(dataset_path)
	_, valpaths = splitArray(labeledPaths, 0.8, shuffle=True)

	imgTransforms = torchvision.transforms.Compose([
		torchvision.transforms.Resize((224, 224)),
		torchvision.transforms.ConvertImageDtype(torch.float32),
		torchvision.transforms.Normalize([0.449], [0.226]),
	])

	dataset = ImageDataset(valpaths, imgTransforms)
	loader = DataLoader(dataset, batch_size=64, shuffle=True)

	preds = torch.tensor([])
	targets = torch.tensor([])

	for inputs, labels in tqdm(loader):
		logits = (model(inputs)).detach()
		pred = torch.softmax(logits, dim=1)
		pred = torch.argmax(pred, dim=1)
		# print("####",logits)
		# print("####",labels)

        # loss = criterion(logits, labels.long().squeeze())
		preds = torch.cat((preds, pred), dim=0)
		targets = torch.cat((targets, labels.detach()), dim=-1)

	print("######",preds.size(), targets.size(),'######')
	loss = cross_entropy(preds, targets)

	targets = targets.int()

	acc = accuracy(preds, targets)
	# auroc = AUROC()(preds, targets)
	# precision, recall = precision_recall(preds, targets)

	return acc, loss

if __name__ == "__main__":
	archs = ["resnet18", "custom_resnet", "custom_fcn"]
	results = []
	distances = [""]

	for arch in archs:
		results.append(testModel(arch, sys.argv[1], sys.argv[2]))
	
	for arch, acc, loss in results:
		print(f"| {arch} | {acc:.4f} | {loss:.4f} |")
