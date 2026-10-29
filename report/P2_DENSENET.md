# Báo cáo Phần 2: Thực nghiệm DenseNet121 Baseline

## 1. Mục tiêu và Cách làm
* **Bài toán:** Phân loại 10 lớp trên tập dữ liệu CIFAR-10.
* **Mô hình:** DenseNet121 pretrained từ ImageNet.
* **Chế độ:** Mode `head` (Đóng băng toàn bộ các tầng rút trích đặc trưng `features`, chỉ huấn luyện lớp phân loại mới `classifier`).
* **Tiền xử lý & Phân chia:** Resize ảnh 224x224, chuẩn hóa ImageNet. Tập train 45.000 ảnh, validation 5.000 ảnh (phân tầng, seed 42) và tập test độc lập 10.000 ảnh.
* **Siêu tham số:** Optimizer SGD (`lr=0.001`, `momentum=0.9`), `batch_size=32`, `epochs=5`, `seed=42`.

## 2. Giải thích Kiến trúc và Lớp Classifier
* **Cơ chế Dense Connections:** Không giống ResNet sử dụng phép cộng x + F(x), DenseNet121 kết nối tất cả các tầng phía trước trực tiếp vào các tầng phía sau bằng phép ghép nối chuỗi kênh (concatenation). Điều này giúp tăng cường khả năng tái sử dụng đặc trưng (feature reuse) và giảm thiểu biến mất đạo hàm.
* **Lớp Classifier mới:** Lớp `classifier` gốc nhận 1.024 kênh đầu ra. Nhóm thay thế bằng `Linear(1024, 10)`.
* **Phân tích tham số:**
  * **Tổng tham số (Total):** 6.964.106 tham số.
  * **Tham số huấn luyện (Trainable):** 10.250 tham số (`classifier.weight`: 10x1024, `classifier.bias`: 10).

## 3. Lệnh chạy
```bash
# Huấn luyện DenseNet121 Baseline
python -m src.train --run-name p2_densenet_baseline --model densenet121 --mode head --lr 0.001 --batch-size 32 --epochs 5 --seed 42

# Đánh giá trên tập Test độc lập
python -m src.evaluate --run-dir results/p2_densenet_baseline
```
## 4. Kết quả thực nghiệm
## Chỉ số tổng hợp chính (results/p2_densenet_baseline)
• **Environment:** PyTorch 2.11.0+cu130, Python 3.13
• **Thiết bị chạy:** NVIDIA Tesla T4 GPU (Google Colab / CUDA)
• **Best Validation Epoch:** Epoch 5
• **Train Loss / Accuracy:** 0.3019 / 89.73%
• **Validation Loss / Accuracy:** 0.3046 / 89.86%
• **Test Loss / Accuracy:** 0.3275 / 88.81%
• **Thời gian huấn luyện:** 825.41 giây (~13.7 phút)

## Diễn biến huấn luyện qua 5 Epochs (history.csv)
**Epoch 1:** Train Loss: 0.5239 | Train Acc: 84.04% | Val Loss: 0.3709 | Val Acc: 88.18%
**Epoch 2:** Train Loss: 0.3566 | Train Acc: 88.14% | Val Loss: 0.3284 | Val Acc: 89.30%
**Epoch 3:** Train Loss: 0.3288 | Train Acc: 88.86% | Val Loss: 0.3210 | Val Acc: 89.32%
**Epoch 4:** Train Loss: 0.3116 | Train Acc: 89.45% | Val Loss: 0.3030 | Val Acc: 89.82%
**Epoch 5 (Best):** Train Loss: 0.3019 | Train Acc: 89.73% | Val Loss: 0.3046 | Val Acc: 89.86%

## 5. Quan sát và Nhận xét Đồ thị TensorBoard (Smoothing = 0)
## Các hình ảnh đồ thị
**Train Accuracy:**
**Train Loss:**
**Validation Accuracy:**
**Validation Loss:**

## Phân tích tiến trình học
**Tiến trình học:** Khi cài đặt Smoothing = 0, đồ thị phản ánh chính xác số liệu thô từ file CSV. Cả train_loss (0.5239 → 0.3019) và val_loss (0.3709 → 0.3046) giảm liên tục qua 5 epoch.
**Hiện tượng Overfitting:** Chênh lệch giữa Train Loss và Val Loss ở epoch 5 rất nhỏ (~0.0027). val_accuracy (89.86%) bám rất sát train_accuracy (89.73%), chứng tỏ mô hình không bị overfitting khi freeze backbone.
**Khả năng tổng quát:** Kết quả đánh giá độc lập trên 10.000 ảnh tập Test đạt 88.81% accuracy, tương thích cao với chỉ số trên tập Validation.