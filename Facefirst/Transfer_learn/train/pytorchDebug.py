import torch
import tqdm

def print_named_modules(model):
    for name, module in model.named_children():
        print(f"{name}:\n\t{module}")
        for param in module.parameters():
            print(f"\t{param.requires_grad} {param.shape}")

def check_all_pred(model, dataloader, value):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    size = len(dataloader.dataset)
    model.eval()
    correct = 0
    with torch.no_grad():
        with tqdm(dataloader, unit="batch") as t:
            for X, y in t:
                t.set_description("Testing")
                X, y = X.to(device), y.to(device)
                pred = model(X)
                values = torch.tile(torch.tensor([value]), pred.shape)
                correct += (pred == values).type(torch.float).sum().item()
            if correct == size:
                print("All predictions match the value!")