# Kiểm tra đã thực hiện

Ngày 28/10/2026. Python 3.12, torch 2.14.1+cpu, torchvision 0.29.1+cpu, tensorboard 2.21.0 trong môi trường kiểm tra này. Đây không phải lockfile bắt buộc cho Colab; nhóm lưu pip freeze từ môi trường chạy thật của mình.

## Đã đạt

- Python compile cho src và tests.
- Kiểm tra cú pháp các code cell của 3 notebook (bỏ qua magic TensorBoard khi dùng parser Python).
- 3 unittest: classifier cập nhật nhưng backbone weights và buffers giữ nguyên (ResNet18, DenseNet121); phạm vi finetune ResNet đúng layer4+fc và train/eval mode đúng; split 45.000/5.000, không giao nhau, đủ 500 validation/lớp và tái lập cùng seed.
- 2 smoke run đầy đủ train/validation/checkpoint/log: ResNet18 head và DenseNet121 finetune, mỗi run 1 epoch với FakeData và random weights.
- EventAccumulator đọc được đủ 4 scalar TensorBoard, đối chiếu khớp history.csv; summary đánh dấu synthetic_smoke_only.
- Khởi tạo ImageNet pretrained ResNet18 thành công. Tải CIFAR-10 thật bị chậm, đã dừng kiểm tra đó; chưa có thực nghiệm đầy đủ trên CIFAR-10.
- git diff --check đạt.

## Chưa xác nhận

- Chưa train đầy đủ 5 epoch trên CIFAR-10; không có test accuracy thực nghiệm.
- Chưa chạy toàn bộ notebook trên Google Colab hoặc GPU của nhóm.
- Chưa push hoặc mở PR trên GitHub khi không có phiên đăng nhập được cấp quyền.
- Chưa xác nhận rule main đã bật trên server; ảnh repo không thể hiện settings đó.

## Chạy lại kiểm tra offline

```bash
python -m unittest discover -s tests -v
python -m src.train --run-name smoke_resnet --smoke --epochs 1 --batch-size 2 --workers 0 --output-root smoke_results
python -m src.train --run-name smoke_densenet --model densenet121 --mode finetune --smoke --epochs 1 --batch-size 2 --workers 0 --output-root smoke_results
```

Smoke không cần tải dataset hoặc pretrained weights. Không nộp smoke accuracy như kết quả lab. Chạy từ thư mục gốc repo.
