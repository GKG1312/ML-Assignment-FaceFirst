import sys
import time
import torch
# from torch2trt import TRTModule
from timm import create_model

model_arch = sys.argv[1]
model_label = "mix-v1.1"
isTRT = False
optimal_batch_size = 1

def calculateFPS(model_arch, model_label, isTRT, optimal_batch_size):
	model_path = f"models/{model_arch}_{model_label}.pth"

	# Load model
	print("[Info]: Loading model")
	device = "cuda" if torch.cuda.is_available() else "cpu"
	if isTRT:
		model = TRTModule()
		model.load_state_dict(torch.load(model_path))
	else:
		model = create_model(model_arch, pretrained=True, num_classes=1, in_chans=1 )
		model.load_state_dict(torch.load(model_path, map_location=device))

	print("[Info]: Moving to GPU")
	model.to(device)
	dummy_input = torch.randn(optimal_batch_size, 1, 224, 224, dtype=torch.float).to(device)

	repetitions = 30 * 60 * 1
	total_time = 0
	times = []
	with torch.no_grad():
		print("[Info]: Warming up")
		for rep in range(10):
			_ = model(dummy_input)
		print("[Info]: Calculating FPS")
		for rep in range(repetitions):
			if device == "cuda":
				starter, ender = torch.cuda.Event(enable_timing=True),   torch.cuda.Event(enable_timing=True)
				starter.record()
				_ = model(dummy_input)
				ender.record()
				torch.cuda.synchronize()
				curr_time = starter.elapsed_time(ender)/repetitions
				times.append(curr_time)
				total_time += curr_time
			else:
				starter = time.time()
				_ = model(dummy_input)
				ender = time.time()
				curr_time = (ender - starter)
				times.append(curr_time)
				total_time += curr_time
	Throughput =   (repetitions*optimal_batch_size)/total_time
	print('Final Throughput:',Throughput)

if __name__ == "__main__":
	calculateFPS(model_arch, model_label, isTRT, optimal_batch_size)