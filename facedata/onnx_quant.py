import argparse
import numpy as np
import onnxruntime
import time
import torch
import torchvision
from tqdm import tqdm
from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
from dataloader import *
from sklearn.metrics import classification_report



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
    
def split_data(csv_path):
    data = pd.read_csv(csv_path, header=None)
    # print(data)
    data = data.sample(frac=1, ignore_index=True)
    data = data.drop(columns=0)
    # print(data)
    train_size = int(len(data)*0.7)
    val_split =int((len(data)-train_size)*0.8)

    train_set = data[:train_size].reset_index(drop=True)
    train_set = train_set.set_axis(range(train_set.shape[1]), axis=1)
    val_set = data[train_size:train_size+val_split].reset_index(drop=True)
    val_set = val_set.set_axis(range(val_set.shape[1]), axis=1)
    test_set = data[train_size+val_split:].reset_index(drop=True)
    test_set = test_set.set_axis(range(test_set.shape[1]), axis=1)
    # print(train_set.shape, val_set.shape, test_set.shape)
    
    return train_set, val_set, test_set

class MaskDataset(object):
    def __init__(self, csv_file, img_dir, label_dir, transforms=None):
        self.transforms = transforms
        self.list = csv_file
        self.img_dir = img_dir
        self.label_dir = label_dir

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.list[0][idx])
        label_path = os.path.join(self.label_dir, self.list[1][idx])
        img = np.asarray(Image.open(img_path).convert("RGB"))
        # img = (img//255).astype(float)
        img_shape = img.shape
        #Generate Label
        target = generate_target(idx, label_path, img_shape)
        
        if self.transforms is not None:
            img = self.transforms(img)

        return img.type(torch.FloatTensor), target

    def __len__(self):
        return len(self.list)

class QuntizationDataReader(onnxruntime.quantization.CalibrationDataReader):
    def __init__(self, torch_ds, batch_size, input_name):

        self.torch_dl = torch.utils.data.DataLoader(torch_ds, batch_size=batch_size, shuffle=False)

        self.input_name = input_name
        self.datasize = len(self.torch_dl)

        self.enum_data = iter(self.torch_dl)

    def to_numpy(self, pt_tensor):
        return pt_tensor.detach().cpu().numpy() if pt_tensor.requires_grad else pt_tensor.cpu().numpy()

    def get_next(self):
        batch = next(self.enum_data, None)
        if batch is not None:
          return {self.input_name: self.to_numpy(batch[0])}
        else:
          return None

    def rewind(self):
        self.enum_data = iter(self.torch_dl)


def benchmark(model_path):
    session = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    total = 0.0
    runs = 10
    input_data = np.zeros((1, 3, 224, 224), np.float32)
    # Warming up
    _ = session.run([], {input_name: input_data})
    for i in range(runs):
        start = time.perf_counter()
        _ = session.run([], {input_name: input_data})
        end = (time.perf_counter() - start) * 1000
        total += end
        print(f"{end:.2f}ms")
    total /= runs
    print(f"Avg: {total:.2f}ms")

def performance(input_model_path, output_model_path, val_ds):
    correct_int8 = 0
    correct_onnx = 0
    tot_abs_error = 0
    ort_sess = onnxruntime.InferenceSession(input_model_path, providers=["CPUExecutionProvider"])
    ort_int8_sess = onnxruntime.InferenceSession(output_model_path, providers=["CPUExecutionProvider"])
    dl = torch.utils.data.DataLoader(val_ds, batch_size=32, collate_fn=collate_fn)
    def to_numpy(pt_tensor):
        return pt_tensor.detach().cpu().numpy() if pt_tensor.requires_grad else pt_tensor.cpu().numpy()

    
    for img_batch, label_batch in tqdm(dl, ascii=True, unit="batches"):

        ort_inputs = {ort_sess.get_inputs()[0].name: to_numpy(img_batch)}
        ort_outs = ort_sess.run(None, ort_inputs)[0]

        ort_preds = np.argmax(ort_outs, axis=1)
        correct_onnx += np.sum(np.equal(ort_preds[1], to_numpy(label_batch)[1]))


        ort_int8_outs = ort_int8_sess.run(None, ort_inputs)[0]

        ort_int8_preds = np.argmax(ort_int8_outs, axis=1)
        correct_int8 += np.sum(np.equal(ort_int8_preds[1], to_numpy(label_batch)[1]))

        tot_abs_error += np.sum(np.abs(ort_int8_outs - ort_outs))


    print("\n")
    print("="*15)
    print(classification_report(ort_preds[1], to_numpy(label_batch)[1]))
    print("="*15)
    print(classification_report(ort_int8_preds[1], to_numpy(label_batch)[1]))

    print(f"onnx top-1 acc = {100.0 * correct_onnx/len(val_ds)} with {correct_onnx} correct samples")
    print(f"onnx int8 top-1 acc = {100.0 * correct_int8/len(val_ds)} with {correct_int8} correct samples")

    mae = tot_abs_error/(1000*len(val_ds))
    print(f"mean abs error = {mae} with total abs error {tot_abs_error}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model", required=True, help="input model")
    parser.add_argument("--output_model", required=True, help="output model")
    parser.add_argument(
        "--calibrate_dataset", default="./test_images", help="calibration data set"
    )
    parser.add_argument(
        "--quant_format",
        default=QuantFormat.QDQ,
        type=QuantFormat.from_string,
        choices=list(QuantFormat),
    )
    parser.add_argument("--per_channel", default=False, type=bool)
    args = parser.parse_args()
    return args

def get_data():
    data_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(), 
    ])
    trainList, valList, testList = split_data("./files.csv")
    labels = "./dataset_detect/labels"
    imgs = "./dataset_detect/images"
    testdataset = MaskDataset(testList, imgs, labels, data_transform)
    return testdataset

def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    args = get_args()
    input_model_path = args.input_model
    output_model_path = args.output_model
    ort_sess = onnxruntime.InferenceSession(input_model_path, providers=["CPUExecutionProvider"])
    calib_ds = get_data()
    dr = QuntizationDataReader(calib_ds, batch_size=32,
                               input_name=ort_sess.get_inputs()[0].name)

    
    # Calibrate and quantize model
    # Turn off model optimization during quantization
    quantize_static(
        input_model_path,
        output_model_path,
        dr,
        quant_format=args.quant_format,
        per_channel=args.per_channel,
        weight_type=QuantType.QInt8,
    )
    print("Calibrated and quantized model saved.")

    print("benchmarking fp32 model...")
    benchmark(input_model_path)

    print("benchmarking int8 model...")
    benchmark(output_model_path)

    performance(input_model_path, output_model_path, calib_ds)

if __name__ == "__main__":
    main()