# Phần 1 - Pipeline và ResNet18 baseline

## Mục tiêu và cách làm

Phần này chuẩn bị pipeline dùng chung và một kết quả đối chiếu ban đầu cho các thí nghiệm của nhóm. Dữ liệu chọn là CIFAR-10, bài toán phân loại 10 lớp. Nhóm dùng ResNet18 với trọng số ImageNet, thay lớp fc từ 1.000 đầu ra bằng Linear(in_features, 10). Toàn bộ trọng số backbone được đóng băng; optimizer chỉ cập nhật fc.weight và fc.bias.

Ảnh được resize về 224x224, chuyển tensor và chuẩn hóa theo ImageNet. Trong 50.000 ảnh thuộc tập huấn luyện gốc, nhóm tách phân tầng 45.000 ảnh train và 5.000 ảnh validation với seed 42. Tập test chính thức 10.000 ảnh được giữ riêng để đánh giá cuối, không dùng chọn tham số.

Pipeline dùng CrossEntropyLoss trên logits và SGD với learning rate 0.001, momentum 0.9. Batch size 32, ngân sách ban đầu 5 epoch. Các lựa chọn này là kế hoạch nhóm, không phải con số bắt buộc được nêu trong Practice 2.

Ở mode head, backbone được đặt eval để giữ running statistics của BatchNorm; chỉ classifier được đặt train. Với mỗi epoch, code ghi loss và accuracy train/validation theo số mẫu. Checkpoint tốt nhất được chọn theo validation accuracy. TensorBoard ghi bốn đường cong phục vụ quan sát quá trình học.

## Lệnh chạy

```bash
python -m src.train --run-name p1_resnet_baseline --model resnet18 --mode head --lr 0.001 --batch-size 32 --epochs 5 --seed 42
# Chỉ chạy sau khi chốt cấu hình
python -m src.evaluate --run-dir results/p1_resnet_baseline
```

## Kiến trúc và tham số

ResNet18 dùng các residual blocks để cộng nhánh đầu vào với đặc trưng đã biến đổi. Trong torchvision, lớp fc nhận vector 512 chiều. Sau khi đổi thành 10 lớp, số tham số trainable là 512*10+10 = 5.130. Tổng tham số của mô hình sau khi thay head: 11.181.642. Các con số này được kiểm tra bằng code; xem parameters.json của run thật để lưu bằng chứng.

## Kết quả thực nghiệm - cần điền sau khi chạy

| Nội dung | Giá trị |
|---|---|
| Commit và environment | Chờ chạy |
| GPU / CPU thực hiện | Chờ chạy |
| Best validation epoch | Chờ chạy |
| Train loss / acc tại epoch đó | Chờ chạy |
| Validation loss / acc | Chờ chạy |
| Test loss / acc cuối | Chờ chạy |
| Thời gian train | Chờ chạy |

Chèn ảnh TensorBoard vào đây và giải thích cụ thể: loss thay đổi ra sao, train/validation có lệch nhiều không, lý do chọn epoch. Không dùng kết quả smoke test để điền bảng này.

## Bàn giao

Người 2 giữ pipeline và thay DenseNet121 để so sánh kiến trúc. Người 3 giữ ResNet18, thử learning rate, batch size và mode finetune trong các lượt độc lập. Không sửa split/preprocessing âm thầm. Người 1 tổng hợp kết quả và review PR.

## Tài liệu tham khảo

- Practice 2, slide 23-24, Chapter 3 của lớp.
- https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
