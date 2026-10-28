# Giao người 2 - So sánh kiến trúc DenseNet121 với ResNet18

## Mục tiêu gắn với đề

Thực hiện Exercises 1: thử pretrained architecture khác; đồng thời xem kiến trúc, thay classifier, train và đánh giá. Không chỉ đổi tên model rồi chụp accuracy: phải giải thích được thay đổi.

## Điểm bắt đầu

Đợi nền tảng được merge vào main. Clone/pull main, tạo nhánh feat/densenet. Dùng notebooks/02_densenet_colab.ipynb hoặc CLI. Không sửa notebook 01 của người 1. Baseline đối chiếu là results/p1_resnet_baseline, cùng seed 42, preprocessing 224x224, SGD momentum 0.9, lr 0.001, batch 32, 5 epoch.

## Việc cụ thể

1. Xem build_model trong src/train.py: DenseNet121 dùng ImageNet weights; features frozen; classifier mới 10 đầu ra.
2. Chạy:

```bash
python -m src.train --run-name p2_densenet_baseline --model densenet121 --mode head --lr 0.001 --batch-size 32 --epochs 5 --seed 42
```

3. Mở architecture.txt, chỉ ra features, denseblock, classifier. Trong report giải thích ngắn ResNet dùng residual connection, DenseNet nối các đặc trưng qua dense connections. Không cần tự xây lại kiến trúc.
4. Kiểm tra parameters.json: chỉ classifier.weight và classifier.bias được học ở mode head. Ghi tổng tham số và tham số trainable.
5. Mở TensorBoard: chụp đủ train_loss, val_loss, train_accuracy, val_accuracy; có tên run và trục epoch.
6. So sánh với baseline bằng validation accuracy tốt nhất, loss tại epoch đó, epoch tốt nhất, số tham số và thời gian. Nếu khác GPU, chỉ báo thời gian mô tả, không kết luận tốc độ kiến trúc công bằng.
7. Sau khi chốt DenseNet, đánh giá test một lần:

```bash
python -m src.evaluate --run-dir results/p2_densenet_baseline
```

8. Điền report/P2_DENSENET.md: lệnh thực tế, phiên bản, GPU, bảng kết quả, ảnh, nhận xét 300-500 từ (gợi ý độ dài, không phải yêu cầu giảng viên). Nói rõ mô hình nào tốt hơn trong lượt thử, chênh lệch bao nhiêu điểm %, có overfitting không, hạn chế một seed/ít epoch.

## File phải gửi trong PR

- Notebook 02 có phần giải thích/nhận xét của bạn; tránh lưu output dài và dữ liệu nhạy cảm.
- results/p2_densenet_baseline/{config.json,history.csv,summary.json,parameters.json,architecture.txt,split.json,test_metrics.json}.
- results/p2_densenet_baseline/figures/ chứa ảnh TensorBoard.
- results/p2_densenet_baseline/environment.txt từ pip freeze.
- report/P2_DENSENET.md.

best.pt và event TensorBoard giữ riêng, không upload Git. Đừng sửa số liệu hoặc ghi kết luận khi chưa chạy.

## Tiêu chí để trưởng nhóm duyệt

Chạy được từ runtime sạch; dùng pretrained thật; 10 đầu ra; đúng freezing; cùng điều kiện baseline; đủ số liệu và TensorBoard; nhận xét khớp số liệu. Nếu gặp lỗi, báo nguyên thông báo, lệnh chạy và environment, không đổi cấu hình âm thầm.

## Nộp

Commit/push nhánh feat/densenet, mở PR base main, điền PR template, chọn reviewer kakal1090. Nếu bị Request changes, sửa trên cùng nhánh và push tiếp. Chờ trưởng nhóm accept; không push main và không tự merge.
