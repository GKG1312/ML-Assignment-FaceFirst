import os
import cv2
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from data_details import get_count

# data_dir = './Train'
# folders = os.listdir(data_dir)
# F2I={}
# I2F={}
# for i, f in enumerate(folders):
#     F2I[f] = i
#     I2F[i] = f
# # Load the training images
# print(I2F)
# # X_train, y_train = get_images_labels(data_dir, folders, F2I)

# # images = [img[0] for img in X_train]
# # labels = [img[1] for img in X_train]
# print(X_train.shape)
def do_under_sampling(X_train, y_train, I2F):
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    minority = np.where(y_train==3)[0]
    faces_idx = np.where(y_train==2)[0]
    masked_idx = np.where(y_train==1)[0]
    sunglasses_idx = np.where(y_train==0)[0]

    # print(faces_idx.shape, masked_idx, sunglasses_idx)


    under_sampled_sunglasses = np.random.choice(sunglasses_idx, len(minority), replace=False)
    under_sampled_masked = np.random.choice(masked_idx, len(minority), replace=False)
    under_sampled_faces = np.random.choice(faces_idx, len(minority), replace=False)
    
    faces_undersampled = X_train[under_sampled_faces]
    masked_undersampled = X_train[under_sampled_masked]
    sunglasses_undersampled = X_train[under_sampled_sunglasses]
    faces_undersampled_y = y_train[under_sampled_faces]
    masked_undersampled_y = y_train[under_sampled_masked]
    sunglasses_undersampled_y = y_train[under_sampled_sunglasses]
    minority_sample = X_train[minority]
    minority_sample_y = y_train[minority]


    image_resampled = np.concatenate([faces_undersampled, masked_undersampled, sunglasses_undersampled, minority_sample], axis=0)
    labels_resampled = np.concatenate([faces_undersampled_y, masked_undersampled_y, sunglasses_undersampled_y, minority_sample_y], axis=0)
    get_count(labels_resampled, I2F)
    return image_resampled, labels_resampled
    

def do_over_sampling(X_train, y_train, I2F):
    # get_count(y_train, I2F)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    minority = np.where(y_train==3)[0]
    faces_idx = np.where(y_train==2)[0]
    masked_idx = np.where(y_train==1)[0]
    sunglasses_idx = np.where(y_train==0)[0]

    under_sampled_faces = np.random.choice(faces_idx, 2200, replace=False)
    
    faces_undersampled = X_train[under_sampled_faces]
    masked_undersampled = X_train[masked_idx]
    sunglasses_undersampled = X_train[sunglasses_idx]
    faces_undersampled_y = y_train[under_sampled_faces]
    masked_undersampled_y = y_train[masked_idx]
    sunglasses_undersampled_y = y_train[sunglasses_idx]

    minority_sample = X_train[minority]
    minority_labels = y_train[minority]


    # Determine the number of times to replicate each minority sample
    num_replications = len(sunglasses_idx) // len(minority)
    # print(num_replications)

    # Create a list to store augmented images
    augmented_images = []

    for idx in minority:
        image = X_train[idx]  # Get the image
        for _ in range(num_replications-1):

            # Apply random image augmentation techniques here using OpenCV
            # E.g., rotation, scaling, flipping, etc.
            augmented_image = Augmented_image(image)  # Apply augmentation
            augmented_images.append(augmented_image)
            
    # print(f"Augmented {len(augmented_images)} images")
    # Combine the augmented images with the original dataset
    minority_over_sampled = np.vstack([minority_sample, augmented_images])
    minority_over_sampled_y = np.hstack([minority_labels, 3*np.ones(len(augmented_images))])

    image_resampled = np.concatenate([faces_undersampled, masked_undersampled, sunglasses_undersampled, minority_over_sampled], axis=0)
    labels_resampled = np.concatenate([faces_undersampled_y, masked_undersampled_y, sunglasses_undersampled_y, minority_over_sampled_y], axis=0)
    get_count(labels_resampled, I2F)
    return image_resampled, labels_resampled
    

def Augmented_image(image):
    # Sample image (replace with your own image data)
    
    image = Image.fromarray(image)
    # Randomly flip the image
    if random.random() > 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # Randomly rotate the image
    image = image.rotate(random.randint(-180, 180))
    image = apply_filter(image, np.random.randint(0,4))
    img = np.array(image)
    # image.show()
    # image.close()
    return img

def apply_filter(image, num):
    filters = ['blur', 'detail', 'sharpened', 'contrasted']
    if filters[num]=='blur':
        image = image.filter(ImageFilter.BLUR)
    elif filters[num]=='detail':
        image = image.filter(ImageFilter.DETAIL)
    elif filters[num]=='sharpened':
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(5)
    elif filters[num]=='contrasted':
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2)
    return image


# img_res, lbl_res = do_over_sampling(X_train, y_train)

# get_count(lbl_res, I2F, plot=True)