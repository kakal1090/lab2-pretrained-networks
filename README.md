# Lab 2 - Pretrained Neural Networks

Bài nhóm 3 người, bám Practice 2 (slide 23-24, Chapter 3).

## Yêu cầu đề và lựa chọn của nhóm

Đề yêu cầu: nạp pretrained từ torchvision; xem kiến trúc / tham số; thay classifier; transfer learning hoặc fine-tuning; chuẩn bị dữ liệu / DataLoader; train và test; thử nhiều kiến trúc, điều chỉnh hyperparameter, theo dõi TensorBoard.

CIFAR-10 là ví dụ trong slide, không phải ràng buộc được ghi rõ. Nhóm chọn CIFAR-10. Đề không ghi số model, epoch, định dạng nộp hoặc hạn nộp; kế hoạch dưới đây là đề xuất của nhóm, cần đối chiếu hướng dẫn riêng của giảng viên.

## Bắt đầu

Khuyên chạy notebook trên Google Colab có GPU. Mở `notebooks/01_baseline_colab.ipynb` để làm phần người 1. Trong Colab chọn Runtime > Change runtime type > GPU (nếu khả dụng). Notebook chỉ clone main sau khi gói nền tảng đã được đưa vào main; trước đó đổi BRANCH sang feat/lab2-foundation nếu nhánh đã có trên GitHub.

Chạy tại máy cá nhân, từ thư mục repo:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.train --run-name p1_resnet_baseline --model resnet18 --mode head --lr 0.001 --batch-size 32 --epochs 5
python -m tensorboard.main --logdir results
```

Nếu cần GPU trên máy riêng, cài cặp torch/torchvision phù hợp CUDA theo https://pytorch.org/get-started/locally/ trước khi cài requirements. Ghi lại môi trường bằng `python -m pip freeze > results/ENVIRONMENT.txt`.

## Thống nhất thực nghiệm

- CIFAR-10: 45.000 train, 5.000 validation (500 ảnh/lớp), 10.000 test riêng.
- Split phân tầng với seed 42, lưu chỉ số ở split.json.
- Resize 224x224; normalize ImageNet; preprocessing giống nhau giữa các lượt. Không augmentation trong baseline để giữ phạm vi gọn.
- SGD, momentum 0.9; không scheduler / weight decay trong kế hoạch này.
- Mỗi cấu hình chạy 5 epoch, seed 42. Đây là ngân sách khởi đầu; nếu đổi, thống nhất rồi chạy lại các cấu hình so sánh tương ứng.
- Mỗi lượt khởi tạo lại từ ImageNet; head mới. Fine-tuning mở layer4 và head từ đầu, KHÔNG warm-start baseline trong so sánh chính để giữ cùng 5 epoch. Warm-start là mở rộng, phải ghi riêng tổng ngân sách.
- Chọn epoch tốt nhất theo validation accuracy. Chỉ test cấu hình cuối của mỗi hướng sau khi chọn trên validation; không dùng test để chọn hyperparameter.
- accuracy trong file nằm trong [0, 1]; nhân 100 khi trình bày phần trăm.
- Frozen backbone luôn eval để BatchNorm không đổi running statistics. Fine-tuning chỉ bật train cho khối được mở.
- Một seed là giới hạn của lab; không kết luận chênh lệch nhỏ có ý nghĩa thống kê.

## Ma trận chạy

| Người | Run | Model | Mode | LR | Batch | Epoch |
|---|---|---|---|---|---|---|
| 1 | p1_resnet_baseline | ResNet18 | head | 0.001 | 32 | 5 |
| 2 | p2_densenet_baseline | DenseNet121 | head | 0.001 | 32 | 5 |
| 3 | p3_lr_01 | ResNet18 | head | 0.01 | 32 | 5 |
| 3 | p3_batch64 | ResNet18 | head | 0.001 | 64 | 5 |
| 3 | p3_finetune | ResNet18 | finetune | 0.001 | 32 | 5 |

Người 3 so sánh từng lượt với baseline; không đổi nhiều yếu tố một lúc. LR cho fine-tuning nếu bất ổn có thể thử 0.0001 nhưng phải ghi là lượt bổ sung.

## Đầu ra tự động của mỗi run

`results/<run>/`: config.json, architecture.txt, parameters.json, split.json, history.csv, summary.json, best.pt, tensorboard/. Folder có sẵn sẽ bị từ chối để tránh ghi đè. Đổi run-name khi chạy lại. Checkpoint và TensorBoard event không commit; nộp ảnh TensorBoard, JSON/CSV/TXT và nhận xét. Không sửa số liệu thủ công.

TensorBoard theo dõi 4 scalar: train_loss, train_accuracy, val_loss, val_accuracy. Test mặc định không chạy. Sau khi chốt cấu hình, chạy notebook cell đánh giá checkpoint (không train lại).

## Phân công và nộp

- `docs/01_NGUOI_1.md`: phần của trưởng nhóm và hướng dẫn hiểu code.
- `docs/02_GIAO_NGUOI_2.md`: nhiệm vụ so sánh DenseNet.
- `docs/03_GIAO_NGUOI_3.md`: nhiệm vụ tuning / fine-tuning.
- `docs/GIT_WORKFLOW.md`: nộp PR, review, merge và thiết lập bảo vệ main.
- `report/BAO_CAO_KHUNG.md`: khung tổng hợp, chưa có kết quả thực nghiệm.

## Trạng thái trung thực

Code nền tảng được chuẩn bị sẵn. Chưa có kết quả train CIFAR-10 trong repo. Smoke test dùng dữ liệu giả và weights ngẫu nhiên chỉ kiểm tra pipeline; không được dùng làm accuracy của bài lab. Xem docs/VALIDATION.md cho kiểm tra thực tế đã thực hiện.

## Tài liệu

- Practice 2, slide 23-24 trong tài liệu lớp; không đưa PDF lên repo công khai.
- https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
- https://docs.pytorch.org/vision/stable/models.html
