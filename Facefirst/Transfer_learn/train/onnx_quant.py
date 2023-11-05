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
    dl = dl = torch.utils.data.DataLoader(val_ds, batch_size=32, shuffle=False)
    def to_numpy(pt_tensor):
        return pt_tensor.detach().cpu().numpy() if pt_tensor.requires_grad else pt_tensor.cpu().numpy()


    for img_batch, label_batch in tqdm(dl, ascii=True, unit="batches"):

        ort_inputs = {ort_sess.get_inputs()[0].name: to_numpy(img_batch)}
        ort_outs = ort_sess.run(None, ort_inputs)[0]

        ort_preds = np.argmax(ort_outs, axis=1)
        correct_onnx += np.sum(np.equal(ort_preds, to_numpy(label_batch)))


        ort_int8_outs = ort_int8_sess.run(None, ort_inputs)[0]

        ort_int8_preds = np.argmax(ort_int8_outs, axis=1)
        correct_int8 += np.sum(np.equal(ort_int8_preds, to_numpy(label_batch)))

        tot_abs_error += np.sum(np.abs(ort_int8_outs - ort_outs))


    print("\n")
    print("="*15)
    print(classification_report(ort_preds, to_numpy(label_batch)))
    print("="*15)
    print(classification_report(ort_int8_preds, to_numpy(label_batch)))

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

def get_data(paths, sampling_type):
    labeledData, I2F, F2I = loadDataset(paths, balancing=sampling_type)
    trainData, otherData = splitArray(labeledData, 0.7, shuffle=True)
    valData, testData = splitArray(otherData, split=0.5, shuffle=True)

    trainTransforms = torchvision.transforms.Compose([
        # torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.ConvertImageDtype(torch.float32),
        # torchvision.transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomPerspective(),
    ])

    valTransforms = torchvision.transforms.Compose([
        # torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.ConvertImageDtype(torch.float32),
        # torchvision.transforms.Normalize([0.449], [0.226]),
    ])

    # trainDataset = ImageDataset(trainData, trainTransforms)
    # valDataset = ImageDataset(valData, valTransforms)
    testDataset = ImageDataset(testData, valTransforms)
    return testDataset

def main():
    args = get_args()
    input_model_path = args.input_model
    output_model_path = args.output_model
    calibration_dataset_path = args.calibrate_dataset
    ort_sess = onnxruntime.InferenceSession(input_model_path, providers=["CPUExecutionProvider"])
    calib_ds = get_data(calibration_dataset_path, sampling_type=None)
    dr = QuntizationDataReader(calib_ds, batch_size=64,
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