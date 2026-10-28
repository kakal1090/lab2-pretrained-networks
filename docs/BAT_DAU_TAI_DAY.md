# Làm theo thứ tự này

## A. Đưa nền tảng lên GitHub

Gói này đã viết code, notebook và phân công; chưa chạy full CIFAR-10 và chưa tự đưa lên GitHub khi không có phiên đăng nhập. Không coi các ô Chờ chạy là kết quả hoàn thành.

Cách dễ bằng VS Code + Git trên Windows:

1. Giải nén gói ZIP ở Downloads, mở thư mục con lab2-pretrained-networks.
2. Cài Git từ https://git-scm.com/downloads nếu chưa có, mở lại VS Code sau khi cài.
3. Mở terminal ở thư mục khác (ví dụ Desktop) và clone repo:

```powershell
git clone https://github.com/kakal1090/lab2-pretrained-networks.git
cd lab2-pretrained-networks
git switch -c feat/lab2-foundation
```

4. Copy TOÀN BỘ nội dung trong thư mục lab2-pretrained-networks của ZIP vào thư mục vừa clone. Bao gồm .github và .gitignore. Không xóa/copy đè .git trong thư mục clone.
5. Trong terminal repo, nếu lần đầu dùng Git thì cấu hình tên/email của chính bạn:

```powershell
git config user.name "Ten cua ban"
git config user.email "Email GitHub cua ban"
git status
git add README.md requirements.txt .gitignore .github src docs notebooks report tests
git diff --cached --stat
git commit -m "Add Practice 2 foundation and team assignments"
git push -u origin feat/lab2-foundation
```

Git có thể mở trình duyệt để bạn đăng nhập GitHub. Dùng tài khoản kakal1090. Không gửi mật khẩu/token cho ai. Nếu branch đã tồn tại trên remote, đừng force push; lấy nhánh đó về hoặc dùng tên mới và cập nhật PR tương ứng.

6. Mở repo > Compare & pull request > base main / compare feat/lab2-foundation > Create pull request.
7. Đọc Files changed. PR này do chính bạn tạo, không tự approve được. Nhờ thành viên xem, rồi chủ repo merge theo quyền admin hiện có; không tắt bảo vệ main cho cả nhóm. Nếu rule không cho bypass, cần người review hợp lệ / điều chỉnh mô hình Code Owners trước khi tiếp tục, không push thẳng để né rule.
8. Kiểm tra main hiện src/, docs/, notebooks/ và CODEOWNERS.

## B. Bạn làm phần 1

Mở https://colab.research.google.com/ > File > Upload notebook > chọn notebooks/01_baseline_colab.ipynb trong ZIP. Chọn Runtime > Change runtime type > GPU. Chạy lần lượt các ô; notebook lấy code từ main sau khi bạn merge. Chi tiết docs/01_NGUOI_1.md. Chạy thật có thể cần thời gian và internet để tải dataset/weights. Giữ checkpoint riêng trước khi runtime ngắt.

Nếu chưa muốn upload GitHub trước: mở Colab, upload ZIP, giải nén rồi đổi thư mục đến nơi có src/train.py; bỏ qua cell clone và chạy pip install -r requirements.txt. Code vẫn cần internet để tải dữ liệu/weights.

## C. Giao việc

Mời hai tài khoản trong Settings > Collaborators. Gửi link repo cùng nội dung file docs/02_GIAO_NGUOI_2.md cho người 2 và docs/03_GIAO_NGUOI_3.md cho người 3. Mỗi người đọc README trước. Không cần tự tạo model/pipeline từ đầu.

Người 2 tạo feat/densenet; người 3 tạo feat/tuning. Họ chạy notebook, bổ sung nhận xét, lưu số liệu, push nhánh riêng và mở PR. Họ không push main.

## D. Bạn accept

Đọc checklist docs/01_NGUOI_1.md. PR > Files changed > Review changes > Approve > Submit review > Squash and merge. Nếu chưa đủ, Request changes; họ sửa và push lại cùng nhánh. Kiểm tra rule Code Owners đã bật để hai người không duyệt thay bạn. Xem docs/GIT_WORKFLOW.md.
