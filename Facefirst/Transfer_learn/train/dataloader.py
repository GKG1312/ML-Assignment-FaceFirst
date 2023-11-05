from glob import glob
import random
import os
from PIL import Image
import numpy as np
import torch
from augment import do_over_sampling, do_under_sampling
from torch.utils.data import Dataset
from data_details import get_count


def loadDataset(data_dir, folders=None, balancing=None):
	labeled_dataset = []
	# class_count = {}
	if folders is None:
		folders = os.listdir(data_dir)
	# print(folders)
	F2I = {}
	I2F = {}
	for i, f in enumerate(folders):
		I2F[i] = f
		F2I[f] = i
	# print(f"##### {I2F} ####")
	for folder in folders:
		# count = 0
		# Check if the current folder is a directory
		if os.path.isdir(os.path.join(data_dir,folder)):
			images = os.listdir(os.path.join(data_dir, folder))
			# print(images)

			for image in images:
				# Check if the current file is an image file
				if os.path.isfile(os.path.join(data_dir, folder, image)):
					# labeled_dataset.append((os.path.join(data_dir, folder, image),F2I[folder]))
					img = Image.open(os.path.join(data_dir, folder, image))
					img = img.resize((224,224))
					labeled_dataset.append((np.array(img), F2I[folder]))
					# print(f"### {type(np.array(img))} ### {type(F2I[folder])} ###")
                    # Close the image file
					img.close()
					# labels.append(F2I[folder])
		# class_count[folder] = count
	# print(f"Number of samples per class are {class_count}")
	data = [img[0] for img in labeled_dataset]
	label = [img[1] for img in labeled_dataset]
	print(f"Image shape: {np.shape(data)}, Labels shape: {np.shape(label)}")
	get_count(label, I2F)
	# new_dataset = zip(np.array(data), np.array(label))
	if balancing is not None:
		if balancing=='under_sampling':
			img_res, lbl_res = do_under_sampling(data, label, I2F)
			print("Successfully performed Under Sampling")
			print(f"Image shape: {np.shape(img_res)}, Labels shape: {np.shape(lbl_res)}")
		elif balancing=='over_sampling':
			img_res, lbl_res = do_over_sampling(data, label, I2F)
			print("Successfully performed Under Sampling")
			print(f"Image shape: {np.shape(img_res)}, Labels shape: {np.shape(lbl_res)}")
		
		final_dataset = zip(img_res, lbl_res)
		return final_dataset, I2F, F2I
	else:
		final_dataset = zip(data, label)
		return final_dataset, I2F, F2I

def splitArray(array, split, shuffle=True, seed=None):
	array = list(array)
	if shuffle:
		if seed is not None:
			np.random.seed(seed)
		np.random.shuffle(array)

	print(f"Length of shuffled array: {len(array)}")
	
	left_size = int(len(array) * split)
	print(f"Left size: {left_size}")

	left_dataset = array[:left_size]
	right_dataset = array[left_size:]

	return left_dataset, right_dataset

class ImageDataset(Dataset):
	def __init__(self, dataset, transform=None):
		self.dataset = dataset
		self.transform = transform

		if not all(isinstance(item, tuple) and len(item) == 2 for item in dataset):
			raise ValueError("The dataset should be a list of zipped tuples (image, label).")


	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, idx):
		image, label = self.dataset[idx]

		# image = read_image(img_path, ImageReadMode.RGB)
		if self.transform is not None:
			image = self.transform(image)

		label = torch.tensor(label, dtype=torch.float32)

		return image, label

def get_meanstd(data):

    # list of lists of mean values for each image
    meanRGB = [np.mean(x.numpy(),axis=(1,2)) for x,_ in data]
    stdRGB = [np.std(x.numpy(),axis=(1,2)) for x,_ in data]
    print('Mean & std values for sample:')
    print(meanRGB[0]); print(stdRGB[0])

    # global dataset mean of those means
    meanR = np.mean([m[0] for m in meanRGB])
    meanG = np.mean([m[1] for m in meanRGB])
    meanB = np.mean([m[2] for m in meanRGB])

    # global dataset standard deviation mean
    stdR = np.mean([s[0] for s in stdRGB])
    stdG = np.mean([s[1] for s in stdRGB])
    stdB = np.mean([s[2] for s in stdRGB])
    
    print('\nMean value for dataset:')
    print(f'Mean Values: {meanR} {meanG} {meanB}')
    print(f'STD Values: {stdR} {stdG} {stdB}')

    return [meanR,meanG,meanB],[stdR,stdG,stdB]