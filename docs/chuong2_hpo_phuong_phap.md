# Bản nháp Chương 2 — mục 2.9. Thiết kế tối ưu siêu tham số XGBoost

## 2.9.1. Bài toán và điều kiện đánh giá chung

Gọi \(\theta\) là một cấu hình siêu tham số XGBoost trong không gian tìm kiếm \(\Theta\). Hai giá trị được tính cho mọi cấu hình là ROC-AUC kiểm định chéo và độ ổn định thứ hạng SHAP:

\[
f_1(\theta)=\frac{1}{5}\sum_{r=1}^{5}\operatorname{ROC\text{-}AUC}_{r}(\theta),
\qquad
f_2(\theta)=\frac{2}{5(5-1)}\sum_{1\leq a<b\leq5}
\rho_s\!\left(I^{(a)}(\theta),I^{(b)}(\theta)\right).
\]

Trong đó, \(I^{(r)}\) là vector độ quan trọng TreeSHAP toàn cục của 23 biến gốc, tính trên cùng 1.000 quan sát tham chiếu từ mô hình được huấn luyện ở fold \(r\). Năm fold phân tầng được cố định trên Core; tập tham chiếu nằm ngoài train và validation của mọi fold. Năm mô hình của một cấu hình được dùng đồng thời để tính \(f_1\) và \(f_2\). Nguồn biến thiên của phép đo ổn định là năm tập huấn luyện khác nhau, còn seed của XGBoost được giữ ở 42. Cách tính TreeSHAP, gộp cột mã hóa và xử lý đồng hạng tuân theo mục 2.6–2.8.

Nhánh tối ưu đơn mục tiêu tìm \(\max_{\theta\in\Theta} f_1(\theta)\). Nhánh đa mục tiêu tìm \(\max_{\theta\in\Theta}[f_1(\theta),f_2(\theta)]\). Tập Test độc lập không đi vào phép fit, tính mục tiêu hay chọn siêu tham số và nghiệm Pareto.

## 2.9.2. Không gian tìm kiếm

Hai nhánh dùng cùng không gian chín siêu tham số trong `configs/xgb_search_space.yaml`. Các khoảng dưới đây bám theo đề cương; bước lấy mẫu là quy định triển khai của cấu hình hiện tại.

| Siêu tham số | Khoảng | Cách lấy mẫu |
|---|---:|---|
| `n_estimators` | 100–1.000 | Số nguyên, bước 50 |
| `max_depth` | 2–10 | Số nguyên, bước 1 |
| `learning_rate` | 0,01–0,30 | Số thực theo thang log |
| `min_child_weight` | 1–15 | Số nguyên, bước 1 |
| `subsample` | 0,50–1,00 | Số thực, bước 0,05 |
| `colsample_bytree` | 0,50–1,00 | Số thực, bước 0,05 |
| `gamma` | 0–10 | Số thực, bước 0,5 |
| `reg_alpha` | 10⁻⁴–10 | Số thực theo thang log |
| `reg_lambda` | 10⁻³–100 | Số thực theo thang log |

Các tham số khác giữ theo cấu hình cố định: `objective=binary:logistic`, `eval_metric=logloss`, `tree_method=hist`, `random_state=42`. Số luồng huấn luyện XGBoost được ghi nhận cùng cấu hình phần cứng khi báo cáo thực nghiệm. Pilot không dẫn đến thay đổi khoảng tìm kiếm trong lần triển khai này. Mọi thay đổi sau này phải được khóa trước hai lượt tối ưu chính và áp dụng giống nhau cho cả hai nhánh.

## 2.9.3. Hai chiến lược tìm kiếm và ngân sách

Optuna được dùng để quản lý trial. Nhánh đơn mục tiêu dùng `TPESampler(seed=42, n_startup_trials=16)` và chỉ trả \(f_1\) cho sampler. Nhánh đa mục tiêu dùng `NSGAIISampler(seed=42, population_size=16)` và trả cặp \((f_1,f_2)\), với cả hai hướng đều là `maximize`. Mỗi nhánh có một study riêng được lưu bằng SQLite để giữ siêu tham số, trạng thái, giá trị mục tiêu và thông tin chạy. Mức song song của Optuna là `n_jobs=1`, tránh các trial đồng thời tranh chấp tài nguyên và làm khó diễn giải thời gian chạy.

Để so sánh trong cùng ngân sách đánh giá, **mọi trial ở cả hai nhánh đều tính đủ năm ROC-AUC và năm vector SHAP**, dù sampler đơn mục tiêu chỉ sử dụng ROC-AUC để đề xuất cấu hình tiếp theo. Nhánh đơn mục tiêu lưu \(f_2\), Jaccard top-5 và các điểm theo fold như thông tin kèm trial để đánh giá sau tối ưu; chúng không được đưa vào quyết định lấy mẫu của TPE. Như vậy, cùng số trial hoàn thành tương ứng cùng số lần fit và cùng số lần tính SHAP theo giao thức, nhưng thời gian thực tế của hai thuật toán vẫn cần được đo và báo cáo riêng.

Sau pilot, ngân sách chính được khóa ở **64 trial hoàn thành** (`COMPLETE`) cho mỗi nhánh, TPE dùng 16 trial khởi tạo và NSGA-II dùng cỡ quần thể 16. Với năm mô hình và năm lần tính SHAP trong mỗi trial, ngân sách này tương ứng 320 lần fit và 320 lần giải thích cho mỗi nhánh, tổng cộng 640 lần fit và 640 lần giải thích cho hai nhánh, chưa tính pilot và các lần đánh giá đối chứng. Cỡ quần thể 16 cho phép NSGA-II đi qua nhiều quần thể trong giới hạn 64 trial.

Pilot riêng đã kiểm tra 8 trial cho mỗi nhánh; cả hai nhánh đều có **8/8 trial `COMPLETE`**. Mỗi trial dùng đủ năm fold. Hai nhánh cùng kiểm tra cấu hình biên 100 cây, độ sâu 2 và cấu hình biên 1.000 cây, độ sâu 10 để quan sát chi phí ở hai đầu phạm vi. Thời gian của các trial pilot được tóm tắt dưới đây; đây là số liệu chuẩn bị, **không phải kết quả tối ưu chính thức**.

| Nhánh pilot | Trung vị/trial | Phân vị 90%/trial | Lâu nhất/trial | Tổng thời gian các trial |
|---|---:|---:|---:|---:|
| TPE đơn mục tiêu | 3,12 giây | 13,87 giây | 29,67 giây | 52,57 giây |
| NSGA-II đa mục tiêu | 4,67 giây | 20,48 giây | 49,71 giây | 79,81 giây |

Với chỉ tám trial mỗi nhánh, phân vị 90% còn nhạy với vài trial chậm và không đủ để ước lượng chắc thời gian chạy chính. Chênh lệch thời gian giữa hai hàng, kể cả trên các cấu hình biên chung, không chứng minh bản thân sampler này nhanh hơn sampler kia: chi phí huấn luyện phụ thuộc cấu hình được thử và điều kiện thực thi. Pilot không được gộp vào hai study chính hoặc tập Pareto được báo cáo. Không gian tìm kiếm chín tham số giữ nguyên sau pilot. Khi báo cáo lượt chính, ghi cả số trial thử, số trial hoàn thành/thất bại, số lần fit, số lần tính SHAP, thời gian thực tế, phiên bản phần mềm và phần cứng.

## 2.9.4. Trình tự và trạng thái của một trial

1. Sampler đề xuất \(\theta\) từ cùng không gian tìm kiếm; cấu hình cố định và bộ seed/fold được ghép vào mô hình.
2. Với từng fold Core, bộ tiền xử lý được fit chỉ trên phần train. XGBoost được huấn luyện trên train, tính ROC-AUC trên validation và giải thích cùng 1.000 quan sát tham chiếu bằng TreeSHAP.
3. Sau khi đủ năm fold, tính \(f_1\), \(f_2\) và Jaccard top-5; lưu các giá trị, thời gian và cấu hình vào study. Trial chỉ được đánh dấu `COMPLETE` khi cả hai mục tiêu hữu hạn và nằm trong miền hợp lệ: \(0\leq f_1\leq1\), \(-1\leq f_2\leq1\).
4. Nhánh đơn mục tiêu trả \(f_1\); nhánh đa mục tiêu trả \((f_1,f_2)\). Các trial hoàn thành được dùng cho các phân tích so sánh và Pareto.

Lỗi huấn luyện, lỗi tính SHAP, sai kích thước đầu ra hoặc giá trị mục tiêu không hữu hạn làm trial mang trạng thái `FAIL`, kèm loại lỗi để kiểm tra. Không gán điểm thay thế như \((0,0)\), vì giá trị giả có thể làm sai phân bố mục tiêu và biên Pareto. Vector SHAP có mọi phần tử bằng nhau được xử lý theo quy tắc đã định ở mục 2.7: các cặp liên quan nhận Spearman bằng 0; bản thân trường hợp này không phải trial lỗi. Không cắt tỉa trial theo kết quả một phần của các fold, vì hai mục tiêu chỉ được so sánh sau cùng năm lần huấn luyện. Khi một trial lỗi, script dừng để kiểm tra nguyên nhân; trial được lưu là `FAIL` cùng loại và thông điệp lỗi. Sau khi khắc phục, có thể chạy lại để tiến tới đủ ngân sách `COMPLETE`, trong giới hạn `max_attempts` đã cấu hình. Chỉ chạy một tiến trình cho mỗi study; trial `RUNNING` còn sót sau khi process bị dừng đột ngột phải được kiểm tra trước khi tiếp tục.

SQLite lưu lịch sử trial, nhưng khởi tạo lại sampler sau khi dừng rồi chạy tiếp không bảo đảm chuỗi cấu hình đề xuất giống hệt một lượt chạy liên tục: trạng thái bộ sinh số ngẫu nhiên của sampler có thể không được khôi phục chỉ từ lịch sử study. Vì vậy, báo cáo seed, lịch sử trial thực tế và mọi lần tiếp tục chạy; khi cần tái lập đúng chuỗi đề xuất, thực hiện lượt chính trong một tiến trình liên tục hoặc lưu thêm trạng thái sampler bằng cơ chế đã kiểm chứng.

## 2.9.5. Tập nghiệm không bị chi phối và giới hạn diễn giải

Với hai mục tiêu cần tối đa hóa, cấu hình \(\theta_a\) **chi phối** \(\theta_b\) khi

\[
f_1(\theta_a)\geq f_1(\theta_b),\qquad
f_2(\theta_a)\geq f_2(\theta_b),
\]

và có ít nhất một bất đẳng thức nghiêm ngặt. Với \(C\) là tập trial đa mục tiêu hoàn thành, tập nghiệm không bị chi phối được xác định bởi

\[
P(C)=\left\{\theta\in C:\nexists\theta'\in C\text{ chi phối }\theta\right\}.
\]

\(P(C)\) chỉ là xấp xỉ Pareto trong **các cấu hình đã đánh giá**, không phải biên tối ưu toàn cục của toàn bộ không gian \(\Theta\). Giá trị của nghiệm đơn mục tiêu và XGBoost mặc định được đặt lên cùng đồ thị mục tiêu để đối chiếu; chúng không làm thay đổi lịch sử đề xuất của study đa mục tiêu. Quy tắc chọn nghiệm ưu tiên AUC, ưu tiên stability và cân bằng được khóa trên Development trước khi đánh giá Test. Nếu thực nghiệm không cho thấy đánh đổi rõ hoặc không tìm được nghiệm đa mục tiêu tốt hơn đối chứng theo tiêu chí đặt trước, kết quả đó vẫn được báo cáo.

Độ ổn định ở đây phản ánh thứ hạng SHAP toàn cục khi thay đổi tập train giữa năm fold cố định. Nó không đo độ ổn định của lời giải thích từng khách hàng, độ lớn hoặc dấu SHAP, và không cho phép kết luận về các bộ dữ liệu khác. Việc chọn nhiều cấu hình dựa trên cùng Core và reference có thể thích nghi quá mức với giao thức; các nghiệm đại diện cần được kiểm tra bằng điều kiện lặp mới sau khi lựa chọn, rồi mới dùng Test độc lập cho đánh giá cuối.
