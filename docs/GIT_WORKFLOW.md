# GitHub - nộp bài qua PR và trưởng nhóm duyệt

Repo: https://github.com/kakal1090/lab2-pretrained-networks

## Chủ repo kiểm tra trước

Ảnh ban đầu chỉ cho thấy repo public, main, README và .gitignore; chưa đủ bằng chứng protection đã lưu. Kiểm tra Settings > Branches > rule main. Bật Require a pull request before merging; Require approvals = 1; Require review from Code Owners; Dismiss stale approvals; Require conversation resolution. Không bật force push hoặc deletion. CODEOWNERS trong main phải là `* @kakal1090`.

Một approval bất kỳ không đảm bảo trưởng nhóm duyệt; phải bật Code Owners. Với mô hình một chủ repo này, admin giữ bypass để tự merge PR nền tảng của mình, vì tác giả không tự approve được. Không cấp quyền admin/bypass cho thành viên. Quy tắc chặn người khác trước khi chủ repo duyệt; sau approval, collaborator có thể có quyền bấm merge, nên nhóm thống nhất trưởng nhóm là người bấm. Nếu cần cưỡng chế chỉ một người có quyền merge, cần thiết kế quyền khác (ví dụ organization và hạn chế push phù hợp).

Test bằng PR sửa README của thành viên: chưa có owner approve thì bị chặn; thành viên còn lại approve vẫn chưa đủ; owner approve mới đủ; thêm commit mới phải xin duyệt lại.

Mời thành viên: Settings > Collaborators > Add people > đúng username > họ accept lời mời.

## Mỗi thành viên

Cài Git, mở VS Code terminal. Đặt user.name/user.email của chính mình. Khi GitHub yêu cầu đăng nhập, dùng luồng đăng nhập được Git hướng dẫn; không dán token vào file hoặc chat.

```bash
git clone https://github.com/kakal1090/lab2-pretrained-networks.git
cd lab2-pretrained-networks
git switch main
git pull origin main
git switch -c feat/densenet
# Người 3 dùng feat/tuning; người 1 kết quả dùng feat/p1-results
```

Nếu chạy Colab, tải kết quả về và giải nén vào repo trên máy; chạy git status để kiểm tra không thêm data/weights. Commit các đường dẫn đúng phần mình:

```bash
git status
git add notebooks/02_densenet_colab.ipynb report/P2_DENSENET.md results/p2_densenet_baseline
git diff --cached --stat
git commit -m "Add DenseNet experiment and analysis"
git push -u origin feat/densenet
```

Người 3 thay nhánh và đường dẫn bằng phần của mình. Nếu file còn thiếu, tạo đủ rồi mới git add; không bỏ qua lỗi.

Trên GitHub: Pull requests > New pull request > base main, compare nhánh của bạn > điền template > Create pull request > request kakal1090 review. Nếu chưa xong chọn Draft. Sửa theo góp ý trên cùng nhánh; git add, commit, push; PR tự cập nhật.

## Trưởng nhóm accept

Mở Files changed; đối chiếu checklist docs/01_NGUOI_1.md. Review changes > Request changes nếu chưa đạt, hoặc Approve > Submit review nếu đạt. Sau đó Squash and merge > Confirm. Với PR của chính chủ repo, nhờ người khác đọc và review; chủ repo dùng bypass khi cần, không tắt bảo vệ main cho cả nhóm.

Sau merge, mọi người cập nhật:

```bash
git switch main
git pull origin main
```

Nếu nhánh mình đang làm bị tụt main, commit phần đang làm trước rồi:

```bash
git switch feat/tuning
git fetch origin
git merge origin/main
```

Nếu conflict, đọc hai thay đổi và giải quyết; không dùng force push hay chọn hàng loạt ours/theirs khi chưa hiểu. Với notebook, tránh cùng sửa một file. Một PR thay code chung cần được duyệt riêng trước khi chạy hàng loạt.

## Đưa gói nền tảng lên nếu ChatGPT chưa kết nối GitHub

1. Tải ZIP được cung cấp và giải nén (không có thư mục .git trong ZIP).
2. Clone repo GitHub ở một thư mục riêng, tạo nhánh feat/lab2-foundation.
3. Copy nội dung gói (README, src, docs, notebooks, report, requirements, .github, .gitignore) vào repo đã clone; nhớ file/thư mục bắt đầu bằng dấu chấm. Không copy đè thư mục .git.
4. Kiểm tra diff rồi commit và push:

```bash
git status
git add README.md requirements.txt .gitignore .github src docs notebooks report tests
git diff --cached --stat
git commit -m "Add Practice 2 foundation and team assignments"
git push -u origin feat/lab2-foundation
```

5. Mở PR vào main, đọc thay đổi, merge nền tảng; sau đó hai người kia mới pull và làm việc.

Không chạy upload trực tiếp vào main để né rule; không thay settings để xử lý lỗi code.

Tài liệu GitHub: https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches ; https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners
