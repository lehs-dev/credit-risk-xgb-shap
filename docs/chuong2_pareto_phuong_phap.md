# Bản nháp Chương 2 — mục 2.10 và 2.11. Pareto front và chọn nghiệm đại diện

## 2.10. Pareto dominance và Pareto front

Sau lượt tối ưu đa mục tiêu ở mục 2.9, xét tập \(C\) gồm các trial có trạng thái `COMPLETE` trong study NSGA-II chính. Mỗi trial \(t\in C\) có cặp giá trị \(F(t)=(f_1(t),f_2(t))\), trong đó \(f_1\) là ROC-AUC trung bình trên năm fold Core và \(f_2\) là độ ổn định thứ hạng SHAP tính trên cùng năm mô hình. Cả hai mục tiêu đều cần tối đa hóa. Các trial pilot, trial lỗi và trial của nhánh TPE đơn mục tiêu không thuộc \(C\).

Trial \(a\) **chi phối** trial \(b\), ký hiệu \(a\succ b\), khi

\[
f_1(a)\geq f_1(b),\qquad f_2(a)\geq f_2(b),
\qquad [f_1(a)>f_1(b)\ \lor\ f_2(a)>f_2(b)].
\]

Hai trial có cùng cặp giá trị mục tiêu không chi phối nhau. Quan hệ chi phối dùng các giá trị mục tiêu đã lưu với phép so sánh chính xác, không thêm ngưỡng sai khác tùy chọn. Số nghiệm chi phối một trial được tính bằng

\[
d(b)=\sum_{a\in C,\,a\ne b}\mathbf{1}[a\succ b].
\]

Tập nghiệm không bị chi phối trong các cấu hình đã thử là \(P(C)=\{b\in C:d(b)=0\}\). Quy trình kiểm tra đọc các trial `COMPLETE` từ study SQLite, tự so sánh từng cặp \((f_1,f_2)\) để tính \(d(b)\), rồi đối chiếu tập mã trial \(P(C)\) với `study.best_trials` của Optuna. Nếu hai tập khác nhau, cần kiểm tra dữ liệu, trạng thái trial và cách xử lý điểm trùng trước khi phân tích tiếp. Các trial có cùng cặp mục tiêu đều được giữ trong \(P(C)\); phép gộp điểm trùng chỉ áp dụng ở bước chọn cấu hình đại diện.

Biểu đồ Pareto đặt \(f_1\) trên trục hoành và \(f_2\) trên trục tung; thể hiện mọi trial `COMPLETE`, làm nổi bật \(P(C)\) và ghi mã của các nghiệm đại diện. XGBoost mặc định và nghiệm TPE đơn mục tiêu có thể được đặt trên cùng hệ trục để đối chiếu, nhưng không được thêm vào \(C\) khi xác định front của NSGA-II. Cần ghi rõ đây là **front thực nghiệm trong ngân sách và không gian tìm kiếm đã khảo sát**, không phải biên tối ưu toàn cục. Hình dạng front có thể thay đổi khi tăng ngân sách hoặc đổi giao thức đánh giá; không mặc nhiên kết luận hai mục tiêu xung đột chỉ vì đã dùng tối ưu đa mục tiêu.

## 2.11. Phương pháp lựa chọn nghiệm đại diện từ Pareto front

Việc lựa chọn được khóa bằng giá trị trên Development trước khi đánh giá Test độc lập. Chỉ các trial thuộc \(P(C)\) mới đủ điều kiện; không dùng điểm Test, Jaccard top-5, AP hay metric bổ sung để chọn. Nếu nhiều trial trên front có đúng cùng \((f_1,f_2)\), giữ trial có mã nhỏ nhất làm đại diện của điểm đó. Gọi \(U\) là tập các điểm Pareto khác nhau sau bước này.

Ba vai trò được chọn theo thứ tự xác định:

1. **Ưu tiên AUC:** chọn điểm trong \(U\) có \(f_1\) lớn nhất. Nếu đồng hạng, ưu tiên \(f_2\) cao hơn, sau đó mã trial nhỏ hơn.
2. **Ưu tiên ổn định:** chọn điểm trong \(U\) có \(f_2\) lớn nhất. Nếu đồng hạng, ưu tiên \(f_1\) cao hơn, sau đó mã trial nhỏ hơn. Nếu trùng điểm đã chọn ở bước 1, không chọn lại.
3. **Cân bằng:** xét các điểm còn lại trong \(U\). Chuẩn hóa riêng từng mục tiêu theo min–max của **toàn bộ \(U\)**:

   \[
   z_j(t)=\frac{f_j(t)-\min_{u\in U}f_j(u)}{\max_{u\in U}f_j(u)-\min_{u\in U}f_j(u)},\qquad j\in\{1,2\}.
   \]

   Nếu một mục tiêu có cùng giá trị ở mọi điểm thuộc \(U\), đặt \(z_j(t)=1\) cho tất cả các điểm. Chọn điểm còn lại có khoảng cách Euclid nhỏ nhất tới điểm lý tưởng \((1,1)\):

   \[
   D(t)=\sqrt{[1-z_1(t)]^2+[1-z_2(t)]^2}.
   \]

   Nếu khoảng cách bằng nhau, ưu tiên lần lượt \(f_1\) cao hơn, \(f_2\) cao hơn và mã trial nhỏ hơn.

Việc loại các điểm cực trị đã chọn khỏi bước cân bằng nhằm tạo tối đa ba cấu hình **khác nhau** để đối chiếu. Nếu front có ít hơn ba điểm mục tiêu khác nhau, chỉ báo cáo số nghiệm hiện có; không bổ sung trial bị chi phối để đủ ba vai trò. Bảng lựa chọn cần ghi mã trial, hai mục tiêu CV, siêu tham số, vai trò, hai giá trị chuẩn hóa và \(D\). Các chỉ số bổ sung chỉ được phân tích sau khi danh sách nghiệm đã khóa. Các nghiệm được chọn sau đó cần kiểm tra bằng điều kiện lặp mới để đánh giá độ nhạy của \(f_1\) và \(f_2\) trước khi dùng tập Test cho đánh giá dự đoán cuối cùng.

## Kiểm tra triển khai (tách khỏi phần phương pháp)

Study NSGA-II chính hiện có 64/64 trial `COMPLETE`. Phép so sánh từng cặp độc lập trên các giá trị lưu trong SQLite tìm được sáu trial không bị chi phối: **22, 44, 53, 56, 57, 61**. Tập mã này trùng với `study.best_trials` của Optuna. Đây là kiểm tra tính nhất quán của bước lọc Pareto; các giá trị mục tiêu, hình vẽ và phân tích đánh đổi cụ thể thuộc Chương 3.
