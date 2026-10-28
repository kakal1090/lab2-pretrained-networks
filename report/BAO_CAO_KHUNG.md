# Lab 2 - Pretrained Neural Networks

> Khung báo cáo, chưa có kết quả thực nghiệm. Thay các ô chờ bằng số liệu thật.

## 1. Thành viên và phân công

| Thành viên | Vai trò | Bằng chứng |
|---|---|---|
| Người 1 (điền tên/MSSV) | Pipeline, ResNet baseline, GitHub và tổng hợp | PR, notebook 01, P1_BASELINE.md |
| Người 2 | DenseNet và so sánh kiến trúc | PR, notebook 02, P2_DENSENET.md |
| Người 3 | Learning rate, batch size, fine-tuning | PR, notebook 03, P3_TUNING.md |

## 2. Mục tiêu

Giải thích pretrained, transfer learning, fine-tuning; nêu cách đáp ứng các yêu cầu Practice 2.

## 3. Dữ liệu và phương pháp

CIFAR-10, số lớp, train/validation/test, seed, resize/normalize, optimizer, loss, epochs, GPU và phiên bản phần mềm. Nêu rõ những lựa chọn do nhóm tự đề xuất.

## 4. Kiến trúc

Mô tả ResNet18 và DenseNet121; classifier thay thế; tổng tham số / trainable; các lớp đóng/mở. Tham chiếu architecture.txt và parameters.json.

## 5. Kết quả

| Run | LR | Batch | Mode | Best val acc (%) | Best epoch | Test acc nếu đã chốt (%) | Thời gian |
|---|---|---|---|---|---|---|---|
| p1_resnet_baseline | 0.001 | 32 | head | Chờ chạy | | | |
| p2_densenet_baseline | 0.001 | 32 | head | Chờ chạy | | | |
| p3_lr_01 | 0.01 | 32 | head | Chờ chạy | | | |
| p3_batch64 | 0.001 | 64 | head | Chờ chạy | | | |
| p3_finetune | 0.001 | 32 | finetune | Chờ chạy | | | |

Không điền test cho mọi cấu hình nếu chưa có kế hoạch test; ghi “không đánh giá” thay vì suy đoán. Đính kèm các đồ thị TensorBoard đủ nhãn.

## 6. Phân tích

So sánh kiến trúc, ảnh hưởng LR và batch, hiệu quả fine-tuning, dấu hiệu overfitting. Chênh lệch accuracy dùng điểm phần trăm. Mọi nhận xét phải dẫn tới run và số liệu cụ thể.

## 7. Kết luận và hạn chế

Cấu hình được chọn bằng validation, test cuối, thời gian/tài nguyên, một seed, ngân sách nhỏ. Không kết luận model luôn tốt nhất từ một lần thử.

## 8. Cách chạy lại và tài liệu

Ghi commit SHA, lệnh chạy, environment và các nguồn tham khảo thực sự đã dùng. Đối chiếu quy định giảng viên về sử dụng công cụ hỗ trợ; mỗi thành viên phải hiểu và giải thích phần mình.
