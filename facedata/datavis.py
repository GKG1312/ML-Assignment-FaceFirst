import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches

data_path = []

for img in os.listdir("./dataset_detect/images"):
    img_path = os.path.join("./dataset_detect/images",img)
    lbl = img.replace(".jpg", ".txt")
    label_path = os.path.join("./dataset_detect/labels", lbl)
    # print(img, lbl)
    data_path.append([img_path, label_path])

# data = pd.DataFrame(data_path)
# print(data)
# data.to_csv("files.csv", sep=',', header=False)
label_dict = {0:"Faces",
              2:"Glasses",
              1:"Masked",
              3:"Mask with Glasses"}

for idx, val in enumerate(data_path):
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    image = plt.imread(val[0])
    ax.imshow(image)

    boxes = []
    # print(f"File {val[1]}")
    with open(val[1]) as f:
        for lines in f.readlines():
            class_label, x, y, w, h = [
                    float(x) if float(x) != int(float(x)) else int(float(x))
                    for x in lines.replace("\n", "").split()
                ]
            boxes.append([class_label, x, y, w, h])
            f.close()

    height, width, _ = image.shape
    for box in boxes:
        class_label,x, y, w, h = box[0],box[1], box[2], box[3], box[4]
        top = x-w/2
        left = y-h/2

        box_xmin = max(0, top*width)
        box_ymin = max(0, left*height)
        box_width = min(width, w*width)
        box_height = min(height, h*height)

        rect = patches.Rectangle((box_xmin, box_ymin), box_width, box_height, edgecolor='red', facecolor="none")
        ax.add_patch(rect)
        ax.annotate(label_dict[class_label], (width//2, 5), color='magenta', weight='bold', fontsize=15, ha='center', va='center')
        # print(f"label {class_label} with size ({height, width})")
        # print(f"Box dimension {box_xmin, box_ymin, box_height, box_width}")
    plt.tight_layout()
    plt.axis("off")
    plt.show()
    # plt.pause(0.2)
    # plt.close()
    # time.sleep(2)
    


# print(img_path, label_path)
                    