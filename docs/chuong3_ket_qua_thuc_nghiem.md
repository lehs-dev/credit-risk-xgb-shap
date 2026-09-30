# Chương 3. Kết quả thực nghiệm

## 3.1. Mục tiêu và phạm vi báo cáo

Chương này trình bày kết quả thực nghiệm của bài toán dự đoán khả năng khách hàng vỡ nợ thẻ tín dụng trong tháng kế tiếp. Các kết quả được báo cáo theo đúng giao thức đã khóa ở Chương 2: dữ liệu được chia thành tập Development gồm 24.000 quan sát và tập Final Test gồm 6.000 quan sát. Mọi bước lựa chọn mô hình, tối ưu siêu tham số, xác định Pareto front và kiểm tra độ nhạy đều được thực hiện trên Development. Final Test chỉ được dùng một lần ở cuối để đánh giá năm cấu hình đã khóa.

Chỉ số chính là ROC-AUC. Average Precision (AP), F1, Precision, Recall, Balanced Accuracy, Brier score và ma trận nhầm lẫn được dùng để bổ sung diễn giải. Các chỉ số phụ thuộc ngưỡng dùng ngưỡng xác suất 0,5. Độ ổn định giải thích được đo bằng tương quan Spearman giữa các vector global SHAP feature importance trên cùng tập tham chiếu SHAP.

## 3.2. Kết quả các mô hình đối chứng trên Development

Bảng 3.1 trình bày kết quả 5-fold cross-validation trên Core đối với sáu mô hình đối chứng. CatBoost đạt ROC-AUC trung bình cao nhất trong nhóm đối chứng, theo sau là Random Forest, Gradient Boosting và LightGBM. XGBoost mặc định thấp hơn các mô hình boosting mạnh nhất ở ROC-AUC, nhưng được giữ lại vì đây là điểm xuất phát trực tiếp cho giai đoạn tối ưu siêu tham số.

**Bảng 3.1. Kết quả benchmark mô hình đối chứng trên Core CV**

| Mô hình | ROC-AUC | AP | F1 | Precision | Recall | Balanced Accuracy | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| CatBoost | 0,784666 ± 0,006131 | 0,561650 | 0,479076 | 0,669979 | 0,373043 | 0,660450 | 0,133521 |
| Random Forest | 0,782083 ± 0,007861 | 0,559917 | 0,469098 | 0,674078 | 0,359872 | 0,655204 | 0,134151 |
| Gradient Boosting | 0,781476 ± 0,007005 | 0,551224 | 0,483310 | 0,667367 | 0,379135 | 0,662714 | 0,134605 |
| LightGBM | 0,780781 ± 0,007269 | 0,558117 | 0,482163 | 0,676199 | 0,374809 | 0,661919 | 0,134069 |
| XGBoost mặc định | 0,772620 ± 0,007096 | 0,546795 | 0,478584 | 0,661302 | 0,375202 | 0,660301 | 0,136651 |
| Logistic Regression | 0,726933 ± 0,004152 | 0,506727 | 0,366932 | 0,709027 | 0,247645 | 0,609391 | 0,144338 |

Kết quả này cho thấy các mô hình cây tăng cường phù hợp hơn Logistic Regression cho bộ dữ liệu đang xét. Tuy nhiên, khác biệt giữa các mô hình boosting không quá lớn trên Core CV. Vì đề tài tập trung vào XGBoost và mục tiêu ổn định giải thích SHAP, các phần sau đi sâu vào XGBoost mặc định, XGBoost tối ưu đơn mục tiêu bằng TPE và XGBoost tối ưu đa mục tiêu bằng NSGA-II.

## 3.3. XGBoost mặc định và tối ưu đơn mục tiêu

XGBoost mặc định đạt ROC-AUC CV bằng 0,772620 và Spearman SHAP bằng 0,816008 trên năm fold Core. Sau khi tối ưu đơn mục tiêu bằng TPE với mục tiêu ROC-AUC, trial tốt nhất là trial 50, đạt ROC-AUC CV bằng 0,787436. So với XGBoost mặc định, TPE cải thiện ROC-AUC khoảng 0,014815 điểm và đồng thời có Spearman SHAP cao hơn, từ 0,816008 lên 0,901482.

**Bảng 3.2. XGBoost mặc định và nghiệm TPE tốt nhất trên Development**

| Cấu hình | Trial | ROC-AUC CV | Spearman SHAP | Jaccard top-5 |
|---|---:|---:|---:|---:|
| XGBoost mặc định | - | 0,772620 | 0,816008 | 0,866667 |
| TPE tối ưu ROC-AUC | 50 | 0,787436 | 0,901482 | 0,733333 |

Mặc dù TPE chỉ tối ưu ROC-AUC, kết quả ghi nhận thêm độ ổn định SHAP để phục vụ so sánh sau tối ưu. Giá trị Jaccard top-5 của TPE thấp hơn XGBoost mặc định, cho thấy chỉ số ổn định theo tập năm biến quan trọng nhất không nhất thiết biến thiên cùng chiều với Spearman toàn cục.

## 3.4. Kết quả tối ưu đa mục tiêu và Pareto front

Nhánh NSGA-II tối ưu đồng thời ROC-AUC CV và Spearman SHAP. Trong 64 trial hoàn thành, phép kiểm tra Pareto độc lập tìm được sáu nghiệm không bị chi phối: trial 22, 44, 53, 56, 57 và 61. Từ Pareto front này, ba nghiệm đại diện được chọn theo quy tắc đã khóa ở Chương 2: ưu tiên AUC, ưu tiên ổn định và cân bằng.

**Bảng 3.3. Các nghiệm đại diện và đối chứng trên Development**

| Nguồn | Vai trò | Trial | ROC-AUC CV | Spearman SHAP | Jaccard top-5 | Chênh ROC-AUC so với TPE | Chênh Spearman so với TPE |
|---|---|---:|---:|---:|---:|---:|---:|
| XGBoost mặc định | Đối chứng | - | 0,772620 | 0,816008 | 0,866667 | -0,014815 | -0,085474 |
| TPE | Tối ưu AUC | 50 | 0,787436 | 0,901482 | 0,733333 | 0,000000 | 0,000000 |
| NSGA-II | Ưu tiên AUC | 44 | 0,787721 | 0,933498 | 0,733333 | +0,000286 | +0,032016 |
| NSGA-II | Ưu tiên ổn định | 57 | 0,775843 | 0,973594 | 0,766667 | -0,011593 | +0,072112 |
| NSGA-II | Cân bằng | 61 | 0,786497 | 0,950198 | 0,595238 | -0,000938 | +0,048715 |

Nghiệm Pareto ưu tiên AUC đạt ROC-AUC CV cao nhất trong các cấu hình được chọn, đồng thời có Spearman SHAP cao hơn TPE. Nghiệm Pareto cân bằng chỉ thấp hơn TPE 0,000938 điểm ROC-AUC trên CV, nhưng cao hơn 0,048715 điểm Spearman SHAP. Nghiệm ưu tiên ổn định đạt Spearman SHAP cao nhất, đổi lại ROC-AUC thấp hơn rõ hơn. Kết quả này thể hiện vai trò của tối ưu đa mục tiêu: thay vì chỉ cung cấp một nghiệm có ROC-AUC cao, quá trình tìm kiếm tạo ra nhiều cấu hình thể hiện các mức đánh đổi khác nhau giữa hiệu năng dự đoán và độ ổn định giải thích.

## 3.5. Kiểm tra độ nhạy sau lựa chọn nghiệm

Sau khi khóa ba nghiệm Pareto, nghiệm TPE và XGBoost mặc định, năm cấu hình được đánh giá lại trên 10 cách chia 5-fold mới của Core. Tổng cộng có 250 lần fit XGBoost và 250 lần tính TreeSHAP. Bảng 3.4 tổng hợp kết quả theo từng phân hoạch năm fold.

**Bảng 3.4. Kết quả kiểm tra độ nhạy trên Development**

| Cấu hình | ROC-AUC CV | Spearman SHAP | Jaccard top-5 |
|---|---:|---:|---:|
| Pareto ưu tiên AUC | 0,787594 ± 0,000682 | 0,933913 ± 0,015876 | 0,782857 ± 0,092954 |
| Pareto cân bằng | 0,787178 ± 0,000542 | 0,947777 ± 0,010718 | 0,699524 ± 0,037219 |
| Pareto ưu tiên ổn định | 0,776891 ± 0,000393 | 0,972586 ± 0,002856 | 0,873333 ± 0,095219 |
| TPE tối ưu AUC | 0,787643 ± 0,000585 | 0,911472 ± 0,017231 | 0,767619 ± 0,047562 |
| XGBoost mặc định | 0,774809 ± 0,001354 | 0,837589 ± 0,039689 | 0,790000 ± 0,047258 |

So với TPE, cả ba nghiệm Pareto đều có Spearman SHAP cao hơn trong cả 10/10 phân hoạch. Nghiệm ưu tiên AUC có ROC-AUC trung bình gần như tương đương TPE, với chênh lệch trung bình -0,000049. Nghiệm cân bằng thấp hơn TPE 0,000465 điểm ROC-AUC trung bình nhưng cao hơn 0,036304 điểm Spearman. Nghiệm ưu tiên ổn định thể hiện đánh đổi mạnh nhất: ROC-AUC thấp hơn TPE 0,010752 điểm, nhưng Spearman cao hơn 0,061113 điểm.

Kết quả kiểm tra độ nhạy củng cố nhận xét rằng tối ưu đa mục tiêu tạo ra các cấu hình ổn định hơn về thứ hạng SHAP trên Development. Tuy nhiên, đây vẫn là đánh giá có điều kiện trên Core và tập tham chiếu SHAP cố định; kết quả này không thay thế đánh giá cuối trên Final Test.

## 3.6. Đánh giá cuối trên Final Test

Sau khi danh sách cấu hình được khóa, năm cấu hình được huấn luyện lại một lần trên toàn bộ 24.000 quan sát Development và đánh giá một lần trên 6.000 quan sát Final Test. Tập Test có 1.327 quan sát default và 4.673 quan sát không default, tương ứng tỷ lệ default 22,1167%.

**Bảng 3.5. Kết quả ROC-AUC, AP và Brier trên Final Test**

| Cấu hình | Spearman SHAP trên Development | ROC-AUC Test | AP Test | Brier Test |
|---|---:|---:|---:|---:|
| Pareto ưu tiên AUC | 0,933913 | 0,779784 | 0,557473 | 0,134759 |
| Pareto ưu tiên ổn định | 0,972586 | 0,768113 | 0,537826 | 0,144236 |
| Pareto cân bằng | 0,947777 | 0,781791 | 0,559020 | 0,134622 |
| TPE tối ưu AUC | 0,911472 | 0,781011 | 0,558185 | 0,134681 |
| XGBoost mặc định | 0,837589 | 0,769664 | 0,539973 | 0,137585 |

Trong năm cấu hình đã khóa, nghiệm Pareto cân bằng đạt ROC-AUC Test cao nhất, AP Test cao nhất và Brier thấp nhất. So với TPE, mức chênh của nghiệm Pareto cân bằng rất nhỏ: ROC-AUC cao hơn 0,000780, AP cao hơn 0,000835 và Brier thấp hơn 0,000060. Vì Final Test chỉ được dùng một lần và các chênh lệch này nhỏ, kết quả nên được hiểu theo hướng mô tả thay vì khẳng định ưu thế thống kê.

**Bảng 3.6. Chỉ số phụ thuộc ngưỡng trên Final Test**

| Cấu hình | Precision | Recall | F1 | Balanced Accuracy | TN/FP/FN/TP |
|---|---:|---:|---:|---:|---:|
| Pareto ưu tiên AUC | 0,665260 | 0,356443 | 0,464181 | 0,652756 | 4435 / 238 / 854 / 473 |
| Pareto ưu tiên ổn định | 0,750000 | 0,149209 | 0,248900 | 0,567543 | 4607 / 66 / 1129 / 198 |
| Pareto cân bằng | 0,664773 | 0,352675 | 0,460857 | 0,651086 | 4437 / 236 / 859 / 468 |
| TPE tối ưu AUC | 0,667586 | 0,364732 | 0,471735 | 0,656580 | 4432 / 241 / 843 / 484 |
| XGBoost mặc định | 0,663087 | 0,372268 | 0,476834 | 0,659278 | 4422 / 251 / 833 / 494 |

Ở ngưỡng 0,5, XGBoost mặc định có F1 và Balanced Accuracy cao nhất trong nhóm năm cấu hình, chủ yếu do recall cao hơn. Ngược lại, nghiệm Pareto ưu tiên ổn định có precision cao nhất nhưng recall thấp nhất, chỉ phát hiện 198 trong 1.327 trường hợp default. Điều này cho thấy việc so sánh bằng các chỉ số phụ thuộc ngưỡng cần được đặt trong bối cảnh ngưỡng đang dùng; ngưỡng 0,5 không được tối ưu riêng cho chi phí tín dụng hoặc tỷ lệ phát hiện default.

## 3.7. Tổng hợp nhận xét

Các kết quả thực nghiệm cho thấy tối ưu siêu tham số cải thiện rõ rệt XGBoost so với cấu hình mặc định trên Development. TPE đạt ROC-AUC cao, nhưng các nghiệm Pareto từ NSGA-II cung cấp thêm các lựa chọn có độ ổn định SHAP cao hơn. Trong nhóm Pareto, nghiệm ưu tiên AUC gần TPE nhất về ROC-AUC, nghiệm ưu tiên ổn định thể hiện đánh đổi mạnh giữa hiệu năng và ổn định, còn nghiệm cân bằng đạt mức trung gian hợp lý giữa hai mục tiêu.

Trên Final Test, nghiệm Pareto cân bằng có kết quả tổng quát tốt nhất theo ROC-AUC, AP và Brier trong năm cấu hình đã khóa, nhưng khoảng cách với TPE rất nhỏ. Vì vậy, kết luận chính không nên là NSGA-II luôn vượt trội về dự đoán, mà là tối ưu đa mục tiêu giúp thu được các cấu hình có độ ổn định giải thích cao hơn trong khi vẫn giữ hiệu năng dự đoán cạnh tranh. Đây là kết quả phù hợp với mục tiêu của đề tài: không chỉ tối đa hóa ROC-AUC, mà còn xem xét tính ổn định của diễn giải SHAP khi lựa chọn mô hình rủi ro tín dụng.

Các giới hạn cần lưu ý gồm: Final Test chỉ là một phép chia giữ lại duy nhất; dữ liệu thuộc một bộ UCI cố định; độ ổn định SHAP được đo trên global feature importance với tập tham chiếu cố định; và các chỉ số ở ngưỡng 0,5 chưa phản ánh trực tiếp chi phí kinh doanh của quyết định cấp tín dụng. Những giới hạn này là cơ sở cho phần thảo luận và hướng phát triển ở chương sau.
