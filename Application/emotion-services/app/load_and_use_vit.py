import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import os

# Thiết bị
device = torch.device("xpu")

# Khởi tạo mô hình ViT-B/16
vit_model = models.vit_b_16(weights=None)  # Không load pretrained weights
vit_model.heads = nn.Linear(768, 7)  # 7 lớp cho FER-2013
vit_model = vit_model.to(device)

# Load state_dict từ file .pth
model_path = "models/vit_b16_fer2013_optimized.pth"  # Thay bằng đường dẫn thực tế đến file .pth
vit_model.load_state_dict(torch.load(model_path, map_location=device))
vit_model.eval()  # Chuyển sang chế độ đánh giá
print("ViT-B/16 model loaded successfully!")

# Transform cho ảnh đầu vào
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Hàm dự đoán trên một ảnh
def predict_image(model, image_path):
    # Load và transform ảnh
    image = Image.open(image_path).convert('RGB')  # Đảm bảo ảnh là RGB
    image = transform(image).unsqueeze(0)  # Thêm batch dimension
    image = image.to(device)

    # Dự đoán
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return predicted.item()

# Ví dụ sử dụng
image_path = "models/portrait-of-smiling-young-man-DISF002246.jpg"  # Thay bằng đường dẫn đến ảnh cần dự đoán
predicted_class = predict_image(vit_model, image_path)
print(f"Predicted class: {predicted_class}")