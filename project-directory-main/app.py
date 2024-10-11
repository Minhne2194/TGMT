import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, render_template, redirect, url_for
import torchvision.models as models

# Khởi tạo Flask
app = Flask(__name__)

# Thư mục lưu ảnh tải lên
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



# Bảng ánh xạ giữa chỉ số và tên bệnh
disease_labels = {
    0: 'Bệnh ghẻ táo (Apple scab)',
    1: 'Bệnh thối đen (Black rot)',
    2: 'Bệnh gỉ sắt cây táo và cây tuyết tùng (Cedar apple rust)',
    3: 'Táo khỏe mạnh',
    4: 'Việt quất khỏe mạnh',
    5: 'Bệnh phấn trắng (Powdery mildew) trên anh đào',
    6: 'Bệnh đốm lá Cercospora trên ngô (Cercospora leaf spot)',
    7: 'Bệnh gỉ sắt thông thường (Common rust) trên ngô',
    8: 'Bệnh cháy lá miền Bắc (Northern Leaf Blight) trên ngô',
    9: 'Ngô khỏe mạnh',
    10: 'Bệnh thối đen (Black rot) trên nho',
    11: 'Bệnh Esca (Black Measles) trên nho',
    12: 'Bệnh đốm lá Isariopsis trên nho (Leaf blight)',
    13: 'Nho khỏe mạnh',
    14: 'Bệnh vàng lá gân xanh (Citrus greening) trên cam',
    15: 'Bệnh đốm vi khuẩn (Bacterial spot) trên đào',
    16: 'Đào khỏe mạnh',
    17: 'Bệnh đốm vi khuẩn (Bacterial spot) trên ớt',
    18: 'Ớt khỏe mạnh',
    19: 'Bệnh cháy sớm (Early blight) trên khoai tây',
    20: 'Bệnh cháy muộn (Late blight) trên khoai tây',
    21: 'Khoai tây khỏe mạnh',
    22: 'Dâu tây đen khỏe mạnh',
    23: 'Đậu nành khỏe mạnh',
    24: 'Bệnh phấn trắng trên bí ngô (Powdery mildew)',
    25: 'Bệnh cháy lá (Leaf scorch) trên dâu tây',
    26: 'Dâu tây khỏe mạnh',
    27: 'Bệnh đốm vi khuẩn (Bacterial spot) trên cà chua',
    28: 'Bệnh cháy sớm (Early blight) trên cà chua',
    29: 'Bệnh cháy muộn (Late blight) trên cà chua',
    30: 'Bệnh nấm mốc lá (Leaf Mold) trên cà chua',
    31: 'Bệnh đốm lá Septoria trên cà chua',
    32: 'Nhện đỏ hai đốm trên cà chua (Spider mites)',
    33: 'Bệnh đốm mục tiêu trên cà chua (Target Spot)',
    34: 'Virus xoăn lá vàng cà chua (Tomato Yellow Leaf Curl Virus)',
    35: 'Virus khảm cà chua (Tomato mosaic virus)',
    36: 'Cà chua khỏe mạnh'
}

class ImageClassificationBase(nn.Module):
    def __init__(self):
        super().__init__()

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool=False):
        super(ConvBlock, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class ResNet9(ImageClassificationBase):
    def __init__(self, in_channels, num_diseases):
        super().__init__()

        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True)  # out_dim: 128 x 64 x 64 
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))

        self.conv3 = ConvBlock(128, 256, pool=True)  # out_dim: 256 x 16 x 16
        self.conv4 = ConvBlock(256, 512, pool=True)  # out_dim: 512 x 4 x 4
        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))

        self.classifier = nn.Sequential(
            nn.MaxPool2d(4),
            nn.Flatten(),
            nn.Linear(512, num_diseases)
        )

    def forward(self, xb):  # xb is the loaded batch
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out  # Residual connection
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out  # Residual connection
        out = self.classifier(out)
        return out

# Tải model từ tệp .pth
model = torch.load('models\plant-disease-model-complete.pth', map_location=torch.device('cpu'))
model.eval()

# Hàm tiền xử lý ảnh
def transform_image(image_path):
    transformation = transforms.Compose([
        transforms.Resize((256, 256)),  # Tăng kích thước ảnh lên
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path)
    return transformation(image).unsqueeze(0)

# Hàm lấy dự đoán
def get_prediction(image_path):
    tensor = transform_image(image_path)
    with torch.no_grad():
        outputs = model(tensor)
        _, predicted_class = outputs.max(1)
    
    # Trả về tên bệnh từ bảng ánh xạ
    return disease_labels[predicted_class.item()]

# Route trang chủ
@app.route('/')
def index():
    return render_template('index.html')

# Route xử lý dự đoán
@app.route('/predict', methods=['POST'])
def upload_and_predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Dự đoán kết quả
        predicted_disease = get_prediction(filepath)

        return render_template('index.html', prediction=predicted_disease, image_path=filepath)

if __name__ == "__main__":
    app.run(debug=True)