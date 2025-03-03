import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

# --------------------------
# Configuration and Cache Paths
# --------------------------
config = {
    "enable_ipex": False,         # Enable IPEX optimization for CPU
    "enable_openvino": True,      # Enable conversion for OpenVINO inference (on CPU)
    "openvino_device": "CPU",     # Set to "CPU" for OpenVINO inference (since we're CPU-only)
    "num_epochs": 10,
    "batch_size": 8,
    "learning_rate": 1e-4,
    "input_size": 224,           # Input resolution for the model
    "num_classes": 7             # Number of emotion classes (FER-2013)
}
model_cache_path = "vit_emotion_cpu.pth"
onnx_path = "vit_emotion.onnx"

# --------------------------
# Utility: Print Elapsed Time
# --------------------------
def print_time(message, start_time):
    elapsed = time.time() - start_time
    print(f"{message} - {elapsed:.2f} seconds")
    return time.time()

# --------------------------
# Model Loading and Modification
# --------------------------
def load_pretrained_deit():
    """
    Loads a pre-trained DeiT model via torch.hub and adapts its head for FER-2013.
    """
    print("Loading pre-trained DeiT model...")
    start_time = time.time()
    model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
    # Adapt the classification head for FER-2013 emotion classification
    model.head = nn.Linear(model.head.in_features, config["num_classes"])
    print_time("Model loaded", start_time)
    return model

# --------------------------
# Training and Evaluation Functions with tqdm
# --------------------------
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    start_time = time.time()
    for images, labels in tqdm(dataloader, desc="Training Batches", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_time = time.time() - start_time
    return epoch_loss, epoch_time

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    start_time = time.time()
    for images, labels in tqdm(dataloader, desc="Evaluation Batches", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    accuracy = correct / total
    eval_time = time.time() - start_time
    return accuracy, eval_time

def train_model(train_dataset, val_dataset, model_cache_path, device_name = "cpu") :
    # Force CPU-only device.
    device = torch.device(device_name)
    print(f"Using device: {device}")
    
    # --------------------------
    # Data Loading and Preprocessing
    # --------------------------
    print("Loading FER-2013 datasets...")
    start_time = time.time()
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4)
    print_time("Datasets loaded", start_time)
    
    # --------------------------
    # Model Loading
    # --------------------------
    model = load_pretrained_deit()
    model = model.to(device)
    
    # Check for a cached model; if available, load it.
    if os.path.exists(model_cache_path):
        print(f"Loading cached model from {model_cache_path}...")
        model.load_state_dict(torch.load(model_cache_path, map_location=device))
        print("Cached model loaded successfully.")
    else:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
        
        # Enable IPEX CPU optimization if configured
        if config["enable_ipex"]:
            try:
                import intel_extension_for_pytorch as ipex
                start_time = time.time()
                # For CPU, we can simply pass dtype=torch.float32 (or torch.bfloat16 if supported)
                model, optimizer = ipex.optimize(model, optimizer=optimizer, dtype=torch.float32)
                print_time("IPEX CPU optimization applied", start_time)
            except Exception as e:
                print("Error during IPEX CPU optimization:", e)
        
        # Fine-tuning loop with progress logging
        for epoch in range(config["num_epochs"]):
            print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
            epoch_start = time.time()
            train_loss, train_time = train(model, train_loader, criterion, optimizer, device)
            print(f"Training Loss: {train_loss:.4f}, Batch Time: {train_time:.2f} sec")
            val_acc, eval_time = evaluate(model, val_loader, device)
            print(f"Validation Accuracy: {val_acc:.4f}, Eval Time: {eval_time:.2f} sec")
            print(f"Epoch total time: {time.time() - epoch_start:.2f} seconds")
        
        final_acc, final_eval_time = evaluate(model, val_loader, device)
        print(f"\nFinal Accuracy on FER-2013: {final_acc:.4f}, Final Eval Time: {final_eval_time:.2f} sec")
        torch.save(model.state_dict(), model_cache_path)
        print(f"Trained model saved to {model_cache_path}")
    
    return device, model

# --------------------------
# Main Pipeline for CPU-only with IPEX CPU Optimization and OpenVINO Inference
# --------------------------
def main():
    overall_start = time.time()

    transform = transforms.Compose([
        transforms.Resize((config["input_size"], config["input_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    train_dataset = datasets.ImageFolder(root='Dataset/FER-2013/train', transform=transform)
    val_dataset = datasets.ImageFolder(root='Dataset/FER-2013/test', transform=transform)  
    
    device, model = train_model(train_dataset, val_dataset, model_cache_path, "xpu")  
    # --------------------------
    # OpenVINO Conversion for Inference Optimization (CPU)
    # --------------------------
    if config["enable_openvino"]:
        try:
            print("\nExporting model to ONNX for OpenVINO conversion...")
            start_time = time.time()
            dummy_input = torch.randn(1, 3, config["input_size"], config["input_size"]).to(device)
            torch.onnx.export(model, dummy_input, onnx_path, opset_version=14,
                              input_names=["input"], output_names=["output"])
            print_time("Model exported to ONNX", start_time)
            
            from openvino.runtime import Core
            start_time = time.time()
            ie = Core()
            ov_model = ie.read_model(model=onnx_path)
            compiled_model = ie.compile_model(model=ov_model, device_name=config["openvino_device"])
            print_time(f"OpenVINO model compiled on {config['openvino_device']}", start_time)
            
            sample_img, _ = val_dataset[0]
            sample_np = sample_img.unsqueeze(0).numpy()
            start_time = time.time()
            ov_results = compiled_model([sample_np])
            print_time("OpenVINO inference completed", start_time)
            print("OpenVINO inference results:", ov_results)
        except Exception as e:
            print("OpenVINO conversion error:", e)
    
    print_time("Total pipeline time", overall_start)

if __name__ == "__main__":
    main()

