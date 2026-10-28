# Người 1 - Trưởng nhóm: nền tảng, baseline và tích hợp

## Phần đã chuẩn bị trong gói này

- src/train.py: split dữ liệu, preprocessing, DataLoader, kiến trúc, freezing, train/validation, best checkpoint, TensorBoard, log thực nghiệm.
- src/evaluate.py: chỉ đánh giá best checkpoint trên test sau khi chọn cấu hình.
- Notebook 01: chạy baseline trên Colab, mở TensorBoard, tải kết quả.
- README, phân công người 2/3, khung báo cáo, PR template và CODEOWNERS.

Chưa được coi là hoàn thành thực nghiệm: bạn cần chạy CIFAR-10 thật, giữ kết quả và tự viết nhận xét. Gói này không chứa accuracy giả.

## Bạn làm theo thứ tự

1. Đưa gói nền tảng vào repo bằng nhánh feat/lab2-foundation; đọc thay đổi trước khi merge.
2. Kiểm tra CODEOWNERS là `* @kakal1090` và bật protection main như docs/GIT_WORKFLOW.md.
3. Chạy notebook 01 trên GPU. Chạy smoke nếu muốn kiểm tra cài đặt, sau đó chạy baseline đầy đủ.
4. Đọc `architecture.txt`: tìm backbone và `fc`; giải thích vì sao fc từ 1.000 lớp đổi thành 10.
5. Đọc `parameters.json`: đối chiếu trainable_names chỉ gồm fc.weight và fc.bias trong mode head.
6. Chụp 4 đồ thị TensorBoard; ghi nhận loss giảm hay không, khoảng cách train/validation và epoch tốt nhất.
7. Lưu dependency thực tế với pip freeze; ghi GPU và thời gian chạy trong report/P1_BASELINE.md.
8. Chạy test sau khi chốt baseline, không sửa cấu hình dựa vào test.
9. Commit JSON/CSV/TXT, ảnh và nhận xét vào nhánh feat/p1-results, mở PR. Checkpoint/log event không commit, giữ bản sao riêng để tránh Colab reset.
10. Khi baseline có kết quả, báo hai người kia dùng cùng pipeline, chia sẻ đường dẫn results/p1_resnet_baseline. Họ không cần checkpoint của bạn cho ma trận chính.

## Các khái niệm cần tự giải thích được

- Pretrained: trọng số đã học ImageNet; mode head giữ chúng, chỉ học classifier mới.
- requires_grad=False: không tính gradient cho trọng số frozen; eval giữ trạng thái BatchNorm ổn định.
- CrossEntropyLoss nhận logits, không tự thêm softmax trước loss.
- optimizer chỉ nhận tham số có requires_grad=True.
- train là nơi cập nhật trọng số; validation chọn cấu hình/epoch; test là đánh giá cuối.
- Learning rate: kích thước bước cập nhật; batch size: số ảnh mỗi bước; epoch: một lượt toàn bộ tập train.
- run_epoch tính loss/accuracy theo số mẫu, không trung bình sai giữa các batch kích thước khác nhau.

## Tiêu chí hoàn thành

Có baseline 5 epoch trên 45.000/5.000 ảnh, log thực, ảnh TensorBoard, test cuối, cách chạy lại và nhận xét do bạn kiểm tra. Nếu GPU không đủ thời gian, thống nhất ngân sách mới với cả nhóm và ghi rõ; không tự dùng tập con rồi báo như full dataset.

## Checklist trưởng nhóm trước khi accept PR

- Không có token, dataset, checkpoint lớn hay PDF tài liệu lớp.
- Mỗi kết quả truy ngược được config và history.
- Đúng seed, split, epoch; không đổi pipeline âm thầm.
- Đúng số lớp và đúng nhóm tham số được cập nhật.
- Có TensorBoard, nhận xét và cách chạy lại.
- Không chỉ gửi ảnh accuracy; phải có file số liệu gốc.
- Nếu thay code chung, chạy lại smoke và xem tác động đến baseline.
