import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# 定义Unet网络架构
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.middle = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, out_channels, kernel_size=2, stride=2)
        )

    def forward(self, x):
        enc1 = self.encoder(x)
        enc2 = self.encoder(enc1)
        middle = self.middle(enc2)
        dec1 = self.decoder(middle)
        dec2 = self.decoder(dec1)
        return dec2

# 使用设备（CPU或GPU）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 创建Unet实例
in_channels = 3  # 输入图像通道数
out_channels = 1  # 输出分割图像通道数
model = UNet(in_channels, out_channels).to(device)

# 数据加载和预处理
def load_and_preprocess_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image

# 加载模型和权重
model = UNet(in_channels, out_channels).to(device)
model.load_state_dict(torch.load('unet_model_weights.pth'))  # 请替换为实际的权重文件路径

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载并预测图像
image_path = 'C:\\Users\\zxs\\PycharmProjects\\transtab-main\\1.jpg'  # 请替换为实际的图像文件路径
input_image = load_and_preprocess_image(image_path, transform).unsqueeze(0).to(device)
with torch.no_grad():
    model.eval()
    output = model(input_image)

# 获取分割预测
predicted_segmentation = torch.argmax(output, dim=1).squeeze().cpu().numpy()

# 显示原始图像和分割后的图像
original_image = Image.open(image_path)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(original_image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(predicted_segmentation, cmap='gray')
plt.title('Predicted Segmentation')
plt.axis('off')

plt.tight_layout()
plt.show()
