# Giao người 3 - Hyperparameters và fine-tuning

## Mục tiêu gắn với đề

Thực hiện Exercises 2 và 3: đổi hyperparameter, theo dõi TensorBoard; minh họa fine-tuning so với frozen feature extractor. Dùng cùng ResNet18 baseline của người 1 để xác định tác động từng yếu tố.

## Chuẩn bị

Đợi nền tảng merge main, tạo nhánh feat/tuning. Dùng notebooks/03_tuning_colab.ipynb. Không sửa dữ liệu/split/preprocessing/optimizer chung nếu chưa thống nhất với trưởng nhóm.

## Ba lượt phải làm

Mỗi lượt 5 epoch, seed 42, khởi tạo lại từ ImageNet, head mới. So sánh từng lượt với p1_resnet_baseline (head, lr 0.001, batch 32).

```bash
# A: chỉ đổi learning rate 0.001 -> 0.01
python -m src.train --run-name p3_lr_01 --model resnet18 --mode head --lr 0.01 --batch-size 32 --epochs 5 --seed 42

# B: chỉ đổi batch size 32 -> 64
python -m src.train --run-name p3_batch64 --model resnet18 --mode head --lr 0.001 --batch-size 64 --epochs 5 --seed 42

# C: chỉ đổi nhóm tham số được học: mở layer4 + fc
python -m src.train --run-name p3_finetune --model resnet18 --mode finetune --lr 0.001 --batch-size 32 --epochs 5 --seed 42
```

Đây là fine-tuning từ pretrained ImageNet ngay từ đầu. Không dùng --init-checkpoint trong bảng chính: học baseline 5 epoch rồi fine-tune thêm 5 epoch sẽ không cùng ngân sách 5 epoch với baseline. Nếu muốn warm-start, làm thí nghiệm mở rộng, ghi rõ checkpoint nguồn và tổng epoch 10.

## Việc cụ thể

1. Trước mỗi run, đọc config; giữ đúng seed và split. Sau khi chạy, kiểm tra đủ 5 dòng history.
2. Với C, xem parameters.json: layer4.* và fc.* trainable; lớp trước frozen. Không mở toàn bộ mạng.
3. Mở TensorBoard chung cả 3 run, so sánh đồ thị trên cùng trục epoch. Chụp đủ 4 scalar.
4. Lập bảng baseline + A + B + C: best validation accuracy, best epoch, train/validation loss tại best epoch, thời gian, trainable params.
5. Viết nhận xét: LR lớn có làm dao động không? Batch 64 ảnh hưởng accuracy và thời gian ra sao? Batch lớn giảm số optimizer steps mỗi epoch, nên nêu giới hạn so sánh. Fine-tuning có cải thiện trong ngân sách này không? Có dấu hiệu overfitting không?
6. Chọn cấu hình tốt nhất dựa trên validation accuracy, nếu hòa thì ghi quy tắc chọn (ví dụ ít trainable params hơn). Chỉ sau đó chạy test cho cấu hình đã chọn:

```bash
# Thay RUN_DA_CHON bằng tên run thật, không lấy test làm tiêu chí chọn
python -m src.evaluate --run-dir results/RUN_DA_CHON
```

7. Viết report/P3_TUNING.md: lệnh, môi trường/GPU, bảng kết quả, ảnh và nhận xét 400-600 từ (gợi ý). Không cần accuracy tăng mới là bài đúng: kết quả không cải thiện vẫn có giá trị nếu phân tích trung thực.

## File phải gửi trong PR

- Notebook 03 có phần phân tích của bạn.
- Mỗi run: config.json, history.csv, summary.json, architecture.txt, parameters.json, split.json.
- test_metrics.json của cấu hình đã chọn; ghi lý do chọn trước khi test.
- Ảnh TensorBoard trong results/<run>/figures/ và environment.txt.
- report/P3_TUNING.md.

Không commit best.pt, dataset hoặc event files. Lưu checkpoint riêng để còn evaluate sau khi Colab ngắt.

## Tiêu chí duyệt và nộp

Có 3 lượt tách biệt, chỉ thay một yếu tố so với baseline; trainable layers đúng; đủ TensorBoard và số liệu; không chọn bằng test; nhận xét cụ thể khớp kết quả. Push feat/tuning, PR base main, request kakal1090 review. Sửa trên cùng nhánh nếu bị yêu cầu; không tự merge.
