# Bản nháp Chương 2 — mục 2.6 đến 2.8

## Các lựa chọn phương pháp đã chốt

| Thành phần | Quy tắc |
|---|---|
| Tập được giải thích | 1.000 quan sát tham chiếu lấy phân tầng từ Development; cố định cho mọi cấu hình và mọi fold |
| Mô hình lặp | 5 mô hình XGBoost huấn luyện trên 5 tập train của CV phân tầng trên Core; mỗi tập train có 18.400 quan sát |
| Nguồn biến thiên | Thay đổi tập huấn luyện giữa các fold; `random_state=42` của XGBoost giữ cố định |
| Thuật toán giải thích | TreeSHAP chính xác qua `Booster.predict(pred_contribs=True, approx_contribs=False)` |
| Thang đầu ra | Raw margin (log-odds); không diễn giải đóng góp như mức thay đổi xác suất |
| Dữ liệu nền | Không cấp tập nền riêng; dùng cách tính dựa trên đường đi của cây. Tập tham chiếu chỉ là tập quan sát được giải thích |
| Đơn vị xếp hạng | 23 biến gốc; cộng các SHAP của cột one-hot thuộc cùng biến trên từng quan sát rồi mới lấy trị tuyệt đối |
| Độ ổn định chính | Spearman trung bình của 10 cặp bảng xếp hạng từ 5 mô hình |
| Phân tích bổ sung | Top-5 Jaccard trung bình; Kendall's W chưa đưa vào mục tiêu |
| Trường hợp đồng hạng/suy biến | Spearman dùng hạng trung bình; cặp có vector độ quan trọng hằng nhận 0. Top-5 đồng hạng ở ranh giới được phá theo thứ tự 23 biến gốc; cặp có vector hằng nhận 0 |

## 2.6. Phương pháp tính SHAP/TreeSHAP

Với mỗi cấu hình siêu tham số \(\theta\), mô hình XGBoost được huấn luyện riêng trên phần train của từng fold thuộc Core. Bộ tiền xử lý chỉ được khớp trên phần train tương ứng; sau đó cùng bộ tiền xử lý biến đổi tập tham chiếu SHAP cố định. Không có quan sát tham chiếu nào tham gia huấn luyện hoặc validation. Điều kiện này bảo đảm các giá trị giải thích của các mô hình được tính trên cùng 1.000 khách hàng, dù mô hình được huấn luyện từ những tập con khác nhau.

Đề tài tính các đóng góp TreeSHAP bằng chức năng `pred_contribs` của XGBoost, với `approx_contribs=False`. Đầu ra được giải thích là raw margin của bộ phân loại nhị phân, tương ứng với thang log-odds. Mỗi hàng đóng góp gồm các giá trị SHAP của các cột sau tiền xử lý và một giá trị cơ sở. Tổng các thành phần này được kiểm tra bằng raw margin do mô hình dự đoán. Phép tính không dùng tập dữ liệu nền riêng; thông tin về phân bố nền được suy ra từ các đường đi của cây đã học. Vì vậy, tập tham chiếu SHAP cần được hiểu là những quan sát được giải thích, không phải một tập nền được đưa vào bộ giải thích.

Đối với biến phân loại được mã hóa one-hot, gọi \(G_j\) là tập các cột sau mã hóa của biến gốc \(j\). Trên khách hàng \(i\), đóng góp của biến gốc được xác định bằng \(\phi_{ij}^{(r)}=\sum_{c\in G_j}\phi_{ic}^{(r)}\). Với các biến không mã hóa, \(G_j\) chỉ gồm một cột. Độ quan trọng toàn cục của biến \(j\) trong lần huấn luyện \(r\) là

\[
I_j^{(r)}=\frac{1}{N_{ref}}\sum_{i=1}^{N_{ref}}\left|\phi_{ij}^{(r)}\right|,\qquad N_{ref}=1000.
\]

Quy tắc cộng đóng góp theo từng khách hàng **trước khi** lấy trị tuyệt đối được áp dụng giống nhau cho mọi fold, để mỗi mô hình tạo một vector \(I^{(r)}\) gồm đúng 23 biến gốc. Đại lượng này phản ánh cường độ đóng góp trung bình, không cho biết chiều tăng hoặc giảm rủi ro của biến.

## 2.7. Xây dựng độ đo SHAP stability

Năm tập train của CV cố định tạo ra \(R=5\) mô hình cho cùng cấu hình \(\theta\). Mỗi tập train có 18.400 quan sát Core; 4.600 quan sát còn lại của fold dùng để tính ROC-AUC. Seed của XGBoost giữ ở 42, do đó nguồn biến thiên được khảo sát là thay đổi mẫu huấn luyện giữa các fold. Với mỗi mô hình, độ quan trọng toàn cục được tính trên cùng tập tham chiếu và sắp theo cùng thứ tự 23 biến gốc.

Độ ổn định thứ hạng là trung bình tương quan Spearman giữa mọi cặp vector độ quan trọng:

\[
S(\theta)=\frac{2}{R(R-1)}\sum_{1\leq a<b\leq R}\rho_s\!\left(I^{(a)},I^{(b)}\right),\qquad R=5.
\]

Có 10 cặp tương quan trong giao thức này. Khi các biến có độ quan trọng bằng nhau, Spearman dùng hạng trung bình. Nếu một vector có mọi phần tử bằng nhau, tương quan thứ hạng không được xác định; cặp chứa vector đó được gán điểm 0 theo quy tắc thận trọng, thay vì xem là ổn định hoàn hảo. Các cặp dùng chung mô hình nên không được xem là 10 quan sát thống kê độc lập.

Để kiểm tra sự nhất quán của các biến quan trọng nhất, đề tài báo cáo thêm Jaccard trung bình giữa các tập top-5 của 5 mô hình. Nếu có đồng hạng tại ranh giới top-5, thứ tự 23 biến gốc trong cấu hình dữ liệu được dùng để phá đồng hạng; cặp chứa vector hằng nhận điểm 0. Jaccard không xét thứ tự bên trong top-5 và không thay thế chỉ số Spearman chính.

## 2.8. Objective 2: explanation stability

Mục tiêu thứ hai là tối đa hóa \(f_2(\theta)=S(\theta)\). Cùng năm mô hình fold được dùng để tính mục tiêu thứ nhất \(f_1(\theta)\), tức ROC-AUC validation trung bình. Mọi cấu hình sử dụng cùng phân hoạch Core, cùng tập tham chiếu, seed mô hình, thang SHAP, quy tắc gộp biến và cách xử lý đồng hạng. Tập Test không tham gia tính hai mục tiêu hoặc lựa chọn cấu hình.

Giá trị \(S(\theta)\) chỉ diễn đạt mức ổn định của **thứ hạng độ quan trọng SHAP toàn cục** theo giao thức thay đổi mẫu huấn luyện này. Nó không đo độ ổn định về độ lớn đóng góp, dấu của đóng góp hay lời giải thích cho một khách hàng riêng lẻ. Sau khi chọn nghiệm trên Development, các cấu hình đại diện cần được kiểm tra lại bằng điều kiện lặp mới trong Chương 3 để đánh giá mức phụ thuộc vào năm fold dùng trong HPO.

## Nguồn triển khai cần đối chiếu khi đưa vào DOCX

- [Tài liệu SHAP TreeExplainer](https://shap.readthedocs.io/en/stable/generated/shap.TreeExplainer.html): raw margin của XGBoost, cách tính dựa trên đường đi của cây và vai trò của dữ liệu nền.
- [Tài liệu dự đoán của XGBoost](https://xgboost.readthedocs.io/en/stable/prediction.html): kích thước đầu ra `pred_contribs`, cột bias và raw margin.
- Báo cáo giữa kỳ, mục 1.6–1.7 và các tài liệu [6], [7], [19] trong danh mục tham khảo của báo cáo.

Số liệu từ lần chạy thử XGBoost mặc định thuộc Chương 3, không đưa vào đoạn phương pháp ở trên.
