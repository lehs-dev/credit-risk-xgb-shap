# Bản nháp Chương 2 — mục 2.1–2.5 và 2.12

## 2.1. Phát biểu bài toán

Bộ dữ liệu *Default of Credit Card Clients* gồm 30.000 quan sát, 23 biến dự báo, một mã định danh `ID` và nhãn nhị phân `default.payment.next.month`. Ký hiệu (x_i\in\mathbb{R}^{23}) là thông tin khách hàng thứ (i) trước các bước mã hóa, (y_i\in\{0,1\}) là nhãn, trong đó (y_i=1) biểu thị default trong kỳ tiếp theo. Mã `ID` chỉ dùng để nhận diện bản ghi và không đưa vào mô hình. XGBoost ước lượng (p_\theta(y=1\mid x)) với cấu hình siêu tham số (\theta\in\Theta); chất lượng phân biệt hai lớp được đánh giá bằng ROC-AUC, không quy đổi trực tiếp thành lợi nhuận hoặc quyết định tín dụng.

Đề tài chọn (\theta) theo hai tiêu chí: hiệu năng dự đoán (f_1(\theta)) và độ ổn định thứ hạng quan trọng đặc trưng theo SHAP (f_2(\theta)). Nhánh đơn mục tiêu tối đa hóa (f_1); nhánh đa mục tiêu tìm những cấu hình không bị chi phối theo cặp ([f_1,f_2]). Giá trị của cả hai tiêu chí chỉ được tính từ quy trình kiểm định chéo trên dữ liệu phát triển, theo cùng phân hoạch, tập tham chiếu và quy tắc tiền xử lý. Tập Test độc lập được giữ để đánh giá cuối sau khi khóa cách chọn cấu hình; các mục 2.5–2.11 xác định cụ thể hai hàm mục tiêu và phép chọn nghiệm.

## 2.2. Kiến trúc tổng thể của phương pháp đề xuất

Quy trình bắt đầu từ tệp CSV UCI đã có trong `data/raw/default_credit_card.csv`: kiểm tra cấu trúc và nhãn, chia phân tầng thành Development và Test, rồi tách tập tham chiếu SHAP khỏi Development. Phần còn lại, gọi là Core, được chia thành năm fold phân tầng cố định. Trong mỗi fold, bộ tiền xử lý được khớp trên tập train, biến đổi tập validation và tập tham chiếu; mô hình chỉ học từ tập train. Cách tổ chức này giữ mọi bước có tham số học được bên trong fold và dùng cùng các khách hàng tham chiếu khi so sánh giải thích giữa các mô hình.

Sáu mô hình Logistic Regression, Random Forest, Gradient Boosting, LightGBM, CatBoost và XGBoost cấu hình mặc định được đánh giá trên cùng năm fold Core để tạo đối chứng về dự đoán. Phần nghiên cứu sâu dùng XGBoost với cùng không gian tìm kiếm cho TPE đơn mục tiêu và NSGA-II đa mục tiêu. Mỗi trial tạo năm mô hình, tính ROC-AUC trên validation và TreeSHAP trên tập tham chiếu; từ đó tổng hợp hai mục tiêu. Các trial đa mục tiêu hoàn thành tạo tập không bị chi phối, rồi ba nghiệm đại diện được chọn theo quy tắc ở mục 2.11. Việc kiểm tra độ nhạy sau lựa chọn và đánh giá Test thuộc giai đoạn thực nghiệm của Chương 3.

## 2.3. Phân chia dữ liệu, protocol seed/fold và reference set cho SHAP

Từ 30.000 quan sát, phép chia ngẫu nhiên phân tầng với seed 42 tạo Development 24.000 quan sát (80%) và Test 6.000 quan sát (20%). Từ Development, lấy phân tầng 1.000 quan sát với seed 42 làm tập tham chiếu SHAP cố định. Core gồm 23.000 quan sát Development còn lại; Core, reference và Test đôi một không giao nhau. Test không đi vào fit, kiểm định chéo, tính SHAP, tối ưu hoặc chọn nghiệm.

Trên Core, `StratifiedKFold` gồm năm fold, có trộn mẫu và seed 42, được tạo một lần rồi dùng cho mọi mô hình và mọi trial. Mỗi fold có 18.400 quan sát train và 4.600 quan sát validation; mỗi quan sát Core xuất hiện đúng một lần ở phần validation. Các phép biến đổi có tham số được fit riêng trên train của từng fold, sau đó mới transform validation và reference. Nhóm `EDUCATION` 0, 5, 6 được gộp về 4; `MARRIAGE` 0 được gộp về 3; ba biến danh mục `SEX`, `EDUCATION`, `MARRIAGE` được mã hóa one-hot. Logistic Regression dùng chuẩn hóa các biến số theo thống kê train; các mô hình cây trong đối chứng và XGBoost tối ưu không dùng bước chuẩn hóa số. Không có quan sát reference nào tham gia train hoặc validation trong các trial.

Giao thức lưu chỉ số Development, Test, Core, reference và năm fold để các lượt chạy tái sử dụng đúng cùng các quan sát. Seed và fold cố định giúp các chênh lệch giữa trial ít bị nhiễu bởi việc chia dữ liệu khác nhau; độ ổn định (f_2) ở đây phản ánh thay đổi tập train giữa năm fold, trong khi seed của XGBoost được giữ ở 42. Tập reference là tập **được giải thích** bằng SHAP, không phải dữ liệu nền riêng cung cấp cho TreeSHAP.

## 2.4. Không gian siêu tham số XGBoost

Không gian (\Theta) gồm chín siêu tham số điều khiển số cây, độ sâu, tốc độ học, điều kiện tách và mức regularization. Hai nhánh HPO dùng cùng miền giá trị và cách lấy mẫu trong `configs/xgb_search_space.yaml`; các tham số còn lại giữ cố định với `objective=binary:logistic`, `eval_metric=logloss`, `tree_method=hist` và `random_state=42`.

| Siêu tham số | Miền giá trị và cách lấy mẫu |
|---|---|
| `n_estimators` | Số nguyên 100–1.000, bước 50 |
| `max_depth` | Số nguyên 2–10 |
| `learning_rate` | Số thực 0,01–0,30 theo thang log |
| `min_child_weight` | Số nguyên 1–15 |
| `subsample` | Số thực 0,50–1,00, bước 0,05 |
| `colsample_bytree` | Số thực 0,50–1,00, bước 0,05 |
| `gamma` | Số thực 0–10, bước 0,5 |
| `reg_alpha` | Số thực (10^{-4})–10 theo thang log |
| `reg_lambda` | Số thực (10^{-3})–100 theo thang log |

Các giới hạn này xác định phạm vi được khảo sát, không hàm ý tối ưu toàn cục của XGBoost ngoài miền đã chọn. Cách tổ chức trial, sampler và ngân sách đánh giá được trình bày ở mục 2.9.

## 2.5. Objective 1: predictive performance

Với cấu hình (\theta), ký hiệu (\operatorname{AUC}_r(\theta)) là ROC-AUC tính từ xác suất default dự đoán trên 4.600 quan sát validation của fold (r); mô hình và bộ tiền xử lý tương ứng chỉ được fit trên 18.400 quan sát train. Mục tiêu dự đoán cần tối đa hóa là trung bình không trọng số của năm fold cố định:

\[
f_1(\theta)=\frac{1}{5}\sum_{r=1}^{5}\operatorname{ROC\text{-}AUC}\!\left(y_{\mathrm{val},r},\hat p_{\theta,r}\right).
\]

ROC-AUC đo khả năng xếp hạng một quan sát default cao hơn một quan sát không default trên nhiều ngưỡng; nó không đòi hỏi chọn trước ngưỡng quyết định. Để bổ sung bối cảnh cho lớp default, benchmark và đánh giá sau lựa chọn còn báo cáo Average Precision (AP), Precision, Recall, F1-score, Balanced Accuracy và Brier score. AP được tính bằng `average_precision_score`, nên gọi đúng là AP, không mặc định đồng nhất với diện tích Precision–Recall tính bằng phép tích phân hình thang. Những chỉ số phụ thuộc ngưỡng dùng ngưỡng xác suất 0,5 trong benchmark; chúng không tham gia hàm mục tiêu HPO (f_1).

## 2.12. Tóm tắt chương

Chương 2 xác định quy trình lựa chọn siêu tham số XGBoost theo ROC-AUC kiểm định chéo và độ ổn định thứ hạng SHAP toàn cục. Toàn bộ trial dùng cùng năm fold Core, cùng 1.000 quan sát tham chiếu, cùng quy tắc tiền xử lý và cùng không gian chín siêu tham số. TPE tạo đối chứng đơn mục tiêu; NSGA-II khảo sát tập nghiệm không bị chi phối, từ đó chọn các cấu hình đại diện theo quy tắc khóa trên Development.

Các giới hạn diễn giải cũng được xác định trước thực nghiệm: (f_2) chỉ đo độ nhất quán của thứ hạng quan trọng trên tập tham chiếu và năm lần thay đổi train đã định, còn Pareto front thu được chỉ mô tả những trial đã đánh giá. Chương 3 trình bày dữ liệu, môi trường chạy, kết quả đối chứng và HPO, kiểm tra độ nhạy của nghiệm được chọn bằng điều kiện lặp mới, sau đó đánh giá cuối trên Test độc lập.
