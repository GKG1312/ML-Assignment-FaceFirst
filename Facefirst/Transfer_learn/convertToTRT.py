import sys
import torch
from torch2trt import torch2trt
from timm import create_model

device = torch.device('cuda')

model_arch = sys.argv[1] if len(sys.argv) >= 2 else "resnet18"
model_name = f"{model_arch}_{sys.argv[2]}" if len(sys.argv) == 3 else f"{model_arch}_unknown"

# Load model
print("[Info]: Loading model")
model = create_model(model_arch, pretrained=True, num_classes=1, in_chans=1 )
model.load_state_dict(torch.load(f"models/{model_name}.pth"))
model.to(device)

# Create example data
x = torch.ones((1,1,224,224)).cuda()

# Convert to trt
print("[Info]: Converting to trt")
model_trt = torch2trt(model, [x])

# Save model_trt
print("[Info]: Saving model_trt")
torch.save(model_trt.state_dict(), f"models/{model_name}_trt2.pth")
