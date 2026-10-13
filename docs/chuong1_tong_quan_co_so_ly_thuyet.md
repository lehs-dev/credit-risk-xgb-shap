# Chương 1. Tổng quan đề tài và cơ sở lý thuyết

**Đề tài:** Tối ưu đa mục tiêu siêu tham số XGBoost cho dự đoán rủi ro tín dụng cá nhân xét đến hiệu năng dự đoán và độ ổn định của giải thích SHAP.  
**Sinh viên thực hiện:** Lê Hoàng Sơn, Phạm Thiên Phúc — **Giảng viên hướng dẫn:** TS. Phan Tấn Quốc.

---

## 1.1. Bối cảnh nghiên cứu và bài toán quản trị rủi ro tín dụng cá nhân

### 1.1.1. Tầm quan trọng của quản trị rủi ro tín dụng trong ngân hàng thương mại
Rủi ro tín dụng (Credit Risk) là rủi ro phát sinh tổn thất tài chính khi khách hàng vay vốn không thực hiện đúng hoặc không có khả năng thực hiện nghĩa vụ trả nợ (gốc và lãi) theo hợp đồng đã cam kết. Trong hoạt động của hệ thống ngân hàng thương mại và các tổ chức tài chính hiện đại, cấp tín dụng luôn là nghiệp vụ kinh doanh nòng cốt mang lại phần lớn nguồn thu thu nhập lãi thuần, song cũng là nguồn gốc chủ yếu dẫn đến các biến cố kiệt quệ tài chính và đổ vỡ hệ thống nếu không được kiểm soát chặt chẽ.

Đối với phân khúc tín dụng tiêu dùng cá nhân (Retail Credit / Consumer Lending), đặc biệt là các sản phẩm thẻ tín dụng (Credit Card) và cho vay tiêu dùng tín chấp, bài toán quản trị rủi ro đối mặt với những thách thức đặc thù:
1. **Quy mô danh mục lớn và tính phân tán cao:** Số lượng hồ sơ vay vốn lên tới hàng trăm nghìn hoặc hàng triệu khách hàng, với giá trị từng khoản vay tương đối nhỏ so với tín dụng doanh nghiệp, đòi hỏi quy trình phê duyệt phải được tự động hóa ở mức độ cao.
2. **Thông tin phi đối xứng (Asymmetric Information):** Tổ chức tín dụng chỉ quan sát được thông tin nhân khẩu học cơ bản và lịch sử tín dụng quá khứ, trong khi trạng thái tài chính thực tế và ý định trả nợ của khách hàng mang tính chất tiềm ẩn.
3. **Tính nhạy cảm chu kỳ kinh tế:** Rủi ro vỡ nợ của chủ thẻ liên kết chặt chẽ với các biến động kinh tế vĩ mô (lạm phát, thất nghiệp, lãi suất), dẫn đến sự biến đổi nhanh chóng của phân phối xác suất vỡ nợ.

### 1.1.2. Khung pháp lý Basel II/III và Chuẩn mực Báo cáo Tài chính Quốc tế IFRS 9
Dưới góc độ điều hành an toàn vĩ mô và vi mô, Hiệp ước Vốn Basel (Basel II và Basel III) cùng Chuẩn mực Kế toán Quốc tế số 9 (IFRS 9) đã thiết lập các tiêu chuẩn khắt khe về việc định lượng rủi ro tín dụng.

Theo phương pháp tiếp cận dựa trên xếp hạng nội bộ (Internal Ratings-Based - IRB) của Basel và nguyên tắc trích lập dự phòng theo tổn thất tín dụng dự kiến (Expected Credit Loss - ECL) của IFRS 9, giá trị tổn thất dự kiến của một danh mục tín dụng được mô hình hóa theo công thức tích lũy ba yếu tố:

$$
ECL = PD \times LGD \times EAD
$$

Trong đó:
- **$PD$ (Probability of Default - Xác suất vỡ nợ):** Xác suất một khách hàng vay vốn không trả được nợ trong khoảng thời gian xác định (thường là chu kỳ 12 tháng kế tiếp). Đây chính là biến mục tiêu trọng tâm cần dự báo của đề tài.
- **$LGD$ (Loss Given Default - Tỷ lệ tổn thất khi vỡ nợ):** Tỷ lệ phần trăm dư nợ bị tổn thất thực tế sau khi đã thực hiện các biện pháp thu hồi nợ và xử lý tài sản bảo đảm. Đối với sản phẩm thẻ tín dụng tín chấp không có tài sản bảo đảm, tỷ lệ $LGD$ thường rất cao, dao động từ 45% đến 85%.
- **$EAD$ (Exposure at Default - Tổng dư nợ tại thời điểm vỡ nợ):** Tổng giá trị nghĩa vụ nợ của khách hàng tại thời điểm phát sinh sự kiện vỡ nợ, bao gồm dư nợ gốc hiện hữu, lãi dồn tích và hạn mức thẻ chưa sử dụng dự kiến sẽ được rút thêm trước khi vỡ nợ.

Do $LGD$ và $EAD$ phụ thuộc vào tính chất hợp đồng và biện pháp thu hồi nợ sau khi vỡ nợ xảy ra, việc mô hình hóa chính xác **Xác suất vỡ nợ ($PD$)** ngay từ khâu thẩm định ban đầu đóng vai trò quyết định hàng đầu trong việc giảm thiểu tổn thất và tối ưu hóa chi phí dự phòng vốn rủi ro của ngân hàng.

### 1.1.3. Bản chất toán học của bài toán chấm điểm tín dụng
Bài toán chấm điểm tín dụng cá nhân (Credit Scoring) được phát biểu dưới dạng bài toán phân loại nhị phân có giám sát (Supervised Binary Classification). Giả sử không gian đầu vào là $\mathcal{X} \subseteq \mathbb{R}^d$ đại diện cho $d$ biến giải thích (gồm thông tin nhân khẩu học, lịch sử thanh toán, dư nợ hóa đơn và số tiền thanh toán) và không gian nhãn $\mathcal{Y} = \{0, 1\}$. 

Ký hiệu $y_i = 1$ biểu thị khách hàng thứ $i$ rơi vào trạng thái vỡ nợ trong kỳ đánh giá tiếp theo (Default), và $y_i = 0$ biểu thị khách hàng hoàn thành đầy đủ nghĩa vụ trả nợ (Non-default). Mô hình học máy có nhiệm vụ học một hàm ánh xạ:

$$
f: \mathcal{X} \to [0, 1], \qquad \hat{p}_i = f(x_i) = P(y_i = 1 \mid x_i)
$$

Xác suất dự báo $\hat{p}_i$ phản ánh mức độ rủi ro vỡ nợ của hồ sơ $x_i$, từ đó hỗ trợ hệ thống ra quyết định chấp thuận cấp tín dụng, từ chối hoặc điều chỉnh hạn mức thẻ phù hợp.

### 1.1.4. Bộ dữ liệu chuẩn Default of Credit Card Clients và các nhóm đặc trưng
Để giải quyết bài toán thực nghiệm, nghiên cứu sử dụng bộ dữ liệu chuẩn mực *Default of Credit Card Clients* được thu thập bởi Yeh & Lien (2009) từ một ngân hàng thương mại phát hành thẻ tín dụng tại Đài Loan, lưu trữ tại UCI Machine Learning Repository. Bộ dữ liệu bao gồm 30.000 hồ sơ khách hàng, 23 đặc trưng giải thích phản ánh toàn diện các khía cạnh tài chính và nhân khẩu học, cùng 1 biến mục tiêu nhị phân `default.payment.next.month`:

**Bảng 1.1: Các thuộc tính trong bộ dữ liệu Default of Credit Card Clients.**

| Nhóm đặc trưng | Tên biến | Kiểu dữ liệu | Ý nghĩa nghiệp vụ và miền giá trị |
|:---|:---|:---:|:---|
| **Hạn mức tín dụng** | `LIMIT_BAL` | Liên tục (NT$) | Hạn mức tín dụng được cấp (bao gồm cả tín dụng cá nhân và gia đình). |
| **Nhân khẩu học** | `SEX` | Rời rạc | Giới tính (1 = Nam, 2 = Nữ). |
| | `EDUCATION` | Rời rạc | Trình độ học vấn (1 = Sau đại học, 2 = Đại học, 3 = Phổ thông, 4/5/6/0 = Khác). |
| | `MARRIAGE` | Rời rạc | Tình trạng hôn nhân (1 = Đã kết hôn, 2 = Độc thân, 3/0 = Khác). |
| | `AGE` | Liên tục (Năm) | Độ tuổi của chủ thẻ tín dụng. |
| **Lịch sử trả nợ**<br>(Tháng 4 – Tháng 9/2005) | `PAY_0` | Thứ bậc | Trạng thái trả nợ tháng 9/2005 (-1 = Đúng hạn, 1 = Trễ 1 tháng, 2 = Trễ 2 tháng,...). |
| | `PAY_2` | Thứ bậc | Trạng thái trả nợ tháng 8/2005. |
| | `PAY_3` | Thứ bậc | Trạng thái trả nợ tháng 7/2005. |
| | `PAY_4` | Thứ bậc | Trạng thái trả nợ tháng 6/2005. |
| | `PAY_5` | Thứ bậc | Trạng thái trả nợ tháng 5/2005. |
| | `PAY_6` | Thứ bậc | Trạng thái trả nợ tháng 4/2005. |
| **Dư nợ sao kê hóa đơn**<br>(Tháng 4 – Tháng 9/2005) | `BILL_AMT1` | Liên tục (NT$) | Số dư sao kê hóa đơn tháng 9/2005. |
| | `BILL_AMT2` | Liên tục (NT$) | Số dư sao kê hóa đơn tháng 8/2005. |
| | `BILL_AMT3` | Liên tục (NT$) | Số dư sao kê hóa đơn tháng 7/2005. |
| | `BILL_AMT4` | Liên tục (NT$) | Số dư sao kê hóa đơn tháng 6/2005. |
| | `BILL_AMT5` | Liên tục (NT$) | Số dư sao kê hóa đơn tháng 5/2005. |
| | `BILL_AMT6` | Liên tục (NT$) | Số dư sao kê hóa đơn tháng 4/2005. |
| **Số tiền đã thanh toán**<br>(Tháng 4 – Tháng 9/2005) | `PAY_AMT1` | Liên tục (NT$) | Số tiền đã thanh toán trong tháng 9/2005. |
| | `PAY_AMT2` | Liên tục (NT$) | Số tiền đã thanh toán trong tháng 8/2005. |
| | `PAY_AMT3` | Liên tục (NT$) | Số tiền đã thanh toán trong tháng 7/2005. |
| | `PAY_AMT4` | Liên tục (NT$) | Số tiền đã thanh toán trong tháng 6/2005. |
| | `PAY_AMT5` | Liên tục (NT$) | Số tiền đã thanh toán trong tháng 5/2005. |
| | `PAY_AMT6` | Liên tục (NT$) | Số tiền đã thanh toán trong tháng 4/2005. |
| **Biến mục tiêu** | `default.payment.next.month` | Nhị phân | 1: Vỡ nợ tháng 10/2005 (Default), 0: Không vỡ nợ (Non-default). |

---

## 1.2. Ma trận chi phí sai lầm bất đối xứng trong thẩm định tín dụng

### 1.2.1. Phân loại sai lầm thống kê trong thẩm định tín dụng
Khi áp dụng một ngưỡng quyết định $\tau \in [0, 1]$ (thông thường là $\tau = 0{,}5$) để chuyển đổi xác suất dự báo $\hat{p}_i$ thành quyết định nhị phân $\hat{y}_i = \mathbf{1}[\hat{p}_i \ge \tau]$, không gian dự báo được chia thành 4 khả năng theo ma trận nhầm lẫn (Confusion Matrix):

**Bảng 1.2: Ma trận chi phí sai lầm bất đối xứng trong thẩm định tín dụng.**

| Thực tế \ Dự báo | Dự báo Tốt ($\hat{y} = 0$) | Dự báo Vỡ nợ ($\hat{y} = 1$) |
|:---|:---:|:---:|
| **Thực tế Tốt ($y = 0$)** | Đúng âm ($TN$ - True Negative) | **Dương giả ($FP$ - False Positive / Sai lầm loại I)** |
| **Thực tế Vỡ nợ ($y = 1$)** | **Âm giả ($FN$ - False Negative / Sai lầm loại II)** | Đúng dương ($TP$ - True Positive) |

### 1.2.2. Tính bất đối xứng nghiêm trọng về thiệt hại tài chính
Trong bối cảnh ngân hàng, hai loại sai lầm thống kê mang lại hậu quả tài chính và kinh doanh hoàn toàn khác biệt:
1. **Sai lầm loại I (False Positive - $FP$):** Hệ thống dự báo khách hàng tốt là khách hàng vỡ nợ, dẫn đến quyết định từ chối cấp thẻ tín dụng một cách oan uổng. Hậu quả là ngân hàng đánh mất một khách hàng tiềm năng, mất doanh thu thu nhập lãi và phí quẹt thẻ trong tương lai. Chi phí thiệt hại này tương đương chi phí cơ hội, được chuẩn hóa ở mức tương đối là **$C_{FP} \approx 1\times$**.
2. **Sai lầm loại II (False Negative - $FN$):** Hệ thống dự báo khách hàng vỡ nợ là khách hàng tốt, dẫn đến việc cấp thẻ hoặc giải ngân cho người không có khả năng chi trả. Khi khách hàng vỡ nợ, ngân hàng đối mặt với việc mất toàn bộ vốn gốc giải ngân cùng chi phí xử lý nợ xấu đắt đỏ ($EAD \times LGD$). Trong thực tế ngành tài chính tiêu dùng, thiệt hại của sai lầm loại II nghiêm trọng gấp nhiều lần sai lầm loại I, với tỷ lệ chi phí tổn thất thường nằm trong khoảng:

$$
C_{FN} \approx 5 \times C_{FP} \quad \text{đến} \quad 20 \times C_{FP}
$$

### 1.2.3. Hệ quả phương pháp luận đối với độ đo đánh giá
Do tính chất bất đối xứng sâu sắc giữa $C_{FP}$ và $C_{FN}$ cùng hiện tượng mất cân bằng lớp tự nhiên trong dữ liệu tín dụng (tỷ lệ default thông thường chỉ chiếm từ 15% đến 25%), độ chính xác tổng thể (Accuracy) tại một ngưỡng tùy ý như $0{,}5$ hoàn toàn không có ý nghĩa thực tiễn: một mô hình tầm thường luôn dự đoán tất cả khách hàng là không vỡ nợ vẫn đạt độ chính xác ~78%, nhưng lại gây ra tổn thất tín dụng tối đa do bỏ sót 100% hồ sơ vỡ nợ ($FN$).

Do đó, các độ đo phân tách phi phụ thuộc ngưỡng (threshold-independent metrics) đóng vai trò sống còn trong việc đánh giá và lựa chọn mô hình:
- **ROC-AUC (Area Under the Receiver Operating Characteristic Curve):** Thước đo khả năng xếp hạng tổng quát, phản ánh xác suất mà mô hình gán điểm rủi ro cho một khách hàng vỡ nợ ngẫu nhiên cao hơn một khách hàng không vỡ nợ ngẫu nhiên. ROC-AUC là chỉ số khách quan và độc lập với việc lựa chọn ngưỡng cắt.
- **Average Precision (AP / PR-AUC):** Đo lường diện tích dưới đường cong Precision-Recall, tập trung trực tiếp vào chất lượng nhận diện của lớp thiểu số (lớp vỡ nợ), đặc biệt nhạy cảm với các biến động dương giả ($FP$) khi tỷ lệ lớp mất cân bằng.
- **Balanced Accuracy và Brier Score:** Đánh giá độ chính xác trung bình có trọng số giữa hai lớp và sai số trung bình bình phương của xác suất dự báo đã hiệu chuẩn.

---

## 1.3. Mô hình học máy dạng cây và thuật toán XGBoost

### 1.3.1. Sự phát triển từ mô hình tuyến tính đến mô hình học kết hợp (Ensemble Learning)
Trong nhiều thập kỷ, Hồi quy Logistic (Logistic Regression) là chuẩn mực công nghiệp trong mô hình hóa thẻ điểm tín dụng (Scorecard) nhờ tính chất tường minh, dễ triển khai dưới dạng bảng điểm tuyến tính và khả năng tuân thủ quy chuẩn Basel. Tuy nhiên, Logistic Regression tồn tại hạn chế căn bản: giả định mối quan hệ tuyến tính giữa log-odds và các biến giải thích, đòi hỏi các bước biến đổi biến thủ công phức tạp (WoE - Weight of Evidence, binning rời rạc hóa) và không thể tự động khai thác các tương tác phi tuyến bậc cao giữa các đặc trưng tài chính.

Sự trỗi dậy của các mô hình học máy dạng cây kết hợp (Tree-based Ensemble Methods) đã tạo ra bước đột phá về hiệu năng phân loại trong dữ liệu dạng bảng (Tabular Data):
- **Random Forest (Breiman, 2001):** Sử dụng kỹ thuật Bagging (Bootstrap Aggregating) kết hợp chọn ngẫu nhiên tập đặc trưng để giảm phương sai (variance) của các cây quyết định độc lập.
- **Gradient Boosting Decision Trees - GBDT (Friedman, 2001):** Sử dụng cơ chế Boosting tuần tự, trong đó mỗi cây quyết định mới được huấn luyện để xấp xỉ phần dư (residuals) hoặc gradient âm của hàm mất mát từ các cây phía trước.

### 1.3.2. Cấu trúc giải thuật XGBoost (Chen & Guestrin, 2016)
XGBoost (Extreme Gradient Boosting) là bản cài đặt tối ưu hóa mở rộng của GBDT, bổ sung các cơ chế chính quy hóa chặt chẽ và thuật toán tối ưu phần cứng để đạt hiệu năng vượt trội.

#### 1. Hàm mục tiêu chính quy hóa (Regularized Objective)
Tại vòng lặp thứ $t$, mô hình dự đoán $\hat{y}_i^{(t)} = \hat{y}_i^{(t-1)} + f_t(x_i)$, trong đó $f_t$ là cây quyết định mới cần học. Hàm mục tiêu tổng quát được xác định bởi:

$$
\mathcal{L}^{(t)} = \sum_{i=1}^n \ell\left(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)\right) + \Omega(f_t)
$$

Thành phần kiểm soát độ phức tạp mô hình $\Omega(f_t)$ bao gồm phạt số lượng nút lá $T$ và phạt độ lớn trọng số nút lá $w \in \mathbb{R}^T$ theo cả hai chuẩn $L_1$ và $L_2$:

$$
\Omega(f_t) = \gamma T + \alpha \sum_{j=1}^T |w_j| + \frac{\lambda}{2} \sum_{j=1}^T w_j^2
$$

Trong đó $\gamma$ là ngưỡng tối thiểu để cho phép chia nhánh mới, $\alpha$ là hệ số phạt chính quy hóa $L_1$ (Lasso), và $\lambda$ là hệ số phạt chính quy hóa $L_2$ (Ridge).

#### 2. Khai triển Taylor bậc hai của hàm mục tiêu
XGBoost xấp xỉ hàm mất mát $\ell$ bằng khai triển Taylor bậc hai quanh giá trị dự báo ở bước trước $\hat{y}_i^{(t-1)}$:

$$
\mathcal{L}^{(t)} \approx \sum_{i=1}^n \left[ \ell(y_i, \hat{y}_i^{(t-1)}) + g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) \right] + \Omega(f_t)
$$

Với gradient bậc một $g_i$ và gradient bậc hai (Hessian) $h_i$ đối với hàm mất mát Binary Cross-Entropy (`binary:logistic`):

$$
g_i = \frac{\partial \ell(y_i, \hat{p}_i^{(t-1)})}{\partial \hat{z}_i^{(t-1)}} = \hat{p}_i^{(t-1)} - y_i, \qquad h_i = \frac{\partial^2 \ell(y_i, \hat{p}_i^{(t-1)})}{\partial (\hat{z}_i^{(t-1)})^2} = \hat{p}_i^{(t-1)}(1 - \hat{p}_i^{(t-1)})
$$

#### 3. Nghiệm giải tích trọng số nút lá và tiêu chuẩn phân nhánh (Gain)
Gọi $I_j = \{i : q(x_i) = j\}$ là tập hợp các chỉ số mẫu rơi vào nút lá $j$. Đặt $G_j = \sum_{i \in I_j} g_i$ và $H_j = \sum_{i \in I_j} h_i$. Khi xét phạt $L_2$ thuần túy ($\alpha = 0$), trọng số tối ưu tại lá $j$ có dạng nghiệm đóng:

$$
w_j^* = -\frac{G_j}{H_j + \lambda}
$$

Giá trị hàm mục tiêu tối ưu tương ứng trên toàn bộ cây là:

$$
\mathcal{L}_{\mathrm{opt}}^{(t)} = -\frac{1}{2} \sum_{j=1}^T \frac{G_j^2}{H_j + \lambda} + \gamma T
$$

Khi phân chia một nút lá thành hai nút con Trái ($L$) và Phải ($R$), độ lợi phân nhánh (Gain) được tính toán để lựa chọn điểm cắt tốt nhất:

$$
\mathrm{Gain} = \frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} \right] - \gamma
$$

Thuật toán chỉ thực hiện phân chia nếu $\mathrm{Gain} > 0$, ngăn chặn hiện tượng quá khớp (overfitting) ngay từ cấu trúc cây.

---

## 1.4. Tối ưu hóa siêu tham số (HPO) và bài toán tối ưu đa mục tiêu

### 1.4.1. Không gian siêu tham số và thách thức tối ưu hóa
Mô hình XGBoost có cấu trúc siêu tham số phức tạp điều khiển trực tiếp năng lực biểu diễn và khả năng tổng quát hóa:
1. **Quy mô và năng lực cây:** `n_estimators` (số lượng cây tăng cường), `max_depth` (độ sâu tối đa của từng cây).
2. **Tốc độ học và co cụm:** `learning_rate` ($\eta$ - tốc độ co hẹp đóng góp của cây mới).
3. **Lấy mẫu ngẫu nhiên chống quá khớp:** `subsample` (tỷ lệ lấy mẫu hàng quan sát), `colsample_bytree` (tỷ lệ lấy mẫu cột đặc trưng tại mỗi cây).
4. **Kiểm soát phân nhánh và chính quy hóa:** `min_child_weight` (tổng Hessian tối thiểu của nút lá), `gamma` (phạt số nút lá), `reg_alpha` ($L_1$), `reg_lambda` ($L_2$).

Việc tìm kiếm cấu hình tối ưu bằng lưới vét cạn (Grid Search) gặp hiện tượng bùng nổ tổ hợp chiều (Curse of Dimensionality), trong khi tìm kiếm ngẫu nhiên (Random Search) không khai thác được lịch sử thử nghiệm trước đó.

### 1.4.2. Thuật toán tối ưu hóa Bayes qua TPE (Tree-structured Parzen Estimator)
TPE (Bergstra et al., 2011) là giải thuật tối ưu hóa Bayes phi tham số được tích hợp trong thư viện Optuna (Akiba et al., 2019). Thay vì mô hình hóa hàm mục tiêu $p(y \mid x)$ bằng quá trình Gauss (Gaussian Process), TPE sử dụng quy tắc Bayes để mô hình hóa phân phối tham số có điều kiện dựa trên một ngưỡng phân vị $\gamma$:

$$
p(x \mid y) = \begin{cases} \ell(x) & \text{nếu } y < y^* \\ g(x) & \text{nếu } y \ge y^* \end{cases}
$$

Thuật toán tối đa hóa độ đo Expected Improvement ($EI$):

$$
EI(x) \propto \frac{\ell(x)}{g(x)}
$$

Bằng cách ưu tiên lấy mẫu tại các vùng không gian có tỷ số $\ell(x) / g(x)$ cao, TPE tập trung khai thác các vùng tham số có xác suất sinh ra hiệu năng vượt trội một cách hiệu quả về mặt ngân sách tính toán.

### 1.4.3. Thuật toán di truyền tối ưu đa mục tiêu NSGA-II và khái niệm Pareto
Trong nhiều bài toán thực tế, mục tiêu nghiên cứu không chỉ là tối đa hóa hiệu năng dự đoán mà còn phải cân bằng đồng thời với các tiêu chí phi hiệu năng (độ ổn định, tính công bằng, chi phí tính toán).

#### 1. Khái niệm Pareto Dominance (Chi phối Pareto)
Xét bài toán tối ưu đa mục tiêu với $m$ hàm mục tiêu cần tối đa hóa: $\max_{\theta \in \Theta} F(\theta) = [f_1(\theta), f_2(\theta), \dots, f_m(\theta)]$.
- Một nghiệm $\theta_a$ được gọi là **chi phối** (dominate) nghiệm $\theta_b$ (ký hiệu $\theta_a \succ \theta_b$) khi và chỉ khi:

$$
\forall k \in \{1, \dots, m\}: f_k(\theta_a) \ge f_k(\theta_b) \quad \land \quad \exists j \in \{1, \dots, m\}: f_j(\theta_a) > f_j(\theta_b)
$$

- Tập hợp tất cả các nghiệm không bị chi phối bởi bất kỳ nghiệm nào khác trong không gian tìm kiếm được gọi là **Tập nghiệm Pareto** (Pareto Optimal Set), và hình chiếu của chúng trong không gian mục tiêu được gọi là **Biên Pareto** (Pareto Front).

#### 2. Cơ chế của NSGA-II (Deb et al., 2002)
NSGA-II (Non-dominated Sorting Genetic Algorithm II) duy trì một quần thể ứng viên qua các thế hệ nhờ hai cơ chế cốt lõi:
- **Phân hạng không chi phối nhanh (Fast Non-dominated Sorting):** Phân chia quần thể thành các tầng Pareto (Front 1, Front 2,...), trong đó Front 1 gồm các cá thể không bị chi phối bởi bất kỳ cá thể nào.
- **Khoảng cách chen chúc (Crowding Distance):** Ước lượng mật độ các cá thể xung quanh một điểm trên biên mục tiêu, ưu tiên lựa chọn các cá thể ở vùng thưa thớt nhằm bảo tồn tính đa dạng và trải rộng của tập nghiệm Pareto.

---

## 1.5. Trí tuệ nhân tạo có thể giải thích (XAI) và phương pháp SHAP

### 1.5.1. Yêu cầu pháp lý và bài toán "Hộp đen" (Black-Box Problem)
Mặc dù XGBoost mang lại hiệu năng phân loại vượt trội so với Hồi quy Logistic, bản chất kết hợp của hàng trăm cây quyết định biến mô hình thành một hệ thống "hộp đen" phức tạp. Trong lĩnh vực tài chính tín dụng, tính đóng kín này vi phạm nghiêm trọng các khung pháp lý quốc tế:
- **Hoa Kỳ:** Đạo luật Bình đẳng Cơ hội Tín dụng (Equal Credit Opportunity Act - ECOA) và Đạo luật Báo cáo Tín dụng Công bằng (Fair Credit Reporting Act - FCRA) bắt buộc tổ chức tài chính phải gửi Thông báo Hành động Bất lợi (*Adverse Action Notice*), nêu rõ các lý do chính khiến hồ sơ bị từ chối cấp tín dụng.
- **Liên minh Châu Âu (EU):** Điều 22 của Quy chế Bảo vệ Dữ liệu Chung (GDPR) trao cho công dân quyền được nhận lời giải thích có ý nghĩa (*Right to Explanation*) đối với các quyết định được đưa ra hoàn toàn bằng quy trình tự động.
- **Ủy ban Basel:** Các nguyên tắc về quản trị rủi ro mô hình (Model Risk Management - SR 11-7) đòi hỏi mô hình nội bộ phải được kiểm toán, thẩm định và chứng minh được tính vững chắc của các biến số tác động.

### 1.5.2. Giá trị Shapley trong lý thuyết trò chơi hợp tác
Để giải quyết bài toán diễn giải mô hình phức tạp một cách có cơ sở toán học vững chắc, Lundberg & Lee (2017) đã phát triển khung phương pháp SHAP (SHapley Additive exPlanations) dựa trên khái niệm giá trị Shapley từ lý thuyết trò chơi hợp tác (Shapley, 1953).

Xét tập hợp $N = \{1, 2, \dots, d\}$ người chơi (đại diện cho $d$ đặc trưng đầu vào) hợp tác để tạo ra một giá trị tổng thể $v(N)$ (giá trị dự báo của mô hình). Giá trị Shapley của đặc trưng thứ $j$ là mức đóng góp biên trung bình của đặc trưng đó qua tất cả các liên minh có thể có $S \subseteq N \setminus \{j\}$:

$$
\phi_j = \sum_{S \subseteq N \setminus \{j\}} \frac{|S|! (|N| - |S| - 1)!}{|N|!} \left[ v(S \cup \{j\}) - v(S) \right]
$$

Giá trị Shapley là phương pháp phân bổ duy nhất thỏa mãn đồng thời 4 tiên đề công bằng toán học:
1. **Tính hiệu quả (Efficiency):** Tổng các đóng góp của các đặc trưng bằng hiệu số giữa dự báo mô hình và kỳ vọng cơ sở: $\sum_{j=1}^d \phi_j = f(x) - \mathbb{E}[f(x)]$.
2. **Tính đối xứng (Symmetry):** Hai đặc trưng có đóng góp biên như nhau vào mọi liên minh sẽ nhận giá trị Shapley bằng nhau.
3. **Thuộc tính người chơi giả (Dummy / Null Player):** Đặc trưng không mang lại đóng góp biên vào bất kỳ liên minh nào sẽ có giá trị đóng góp bằng 0.
4. **Tính cộng được (Additivity):** Nếu một trò chơi là tổng của hai trò chơi độc lập, giá trị Shapley bằng tổng giá trị Shapley của từng trò chơi riêng rẽ.

### 1.5.3. Thuật toán TreeSHAP (Lundberg et al., 2020)
Mặc dù giá trị Shapley có nền tảng toán học chặt chẽ, việc tính toán trực tiếp theo định nghĩa đòi hỏi độ phức tạp hàm mũ $O(d \cdot 2^d)$, hoàn toàn bất khả thi đối với các tập dữ liệu có hàng chục biến.

TreeSHAP (Lundberg et al., 2018, 2020) là giải thuật đột phá cho phép tính toán chính xác giá trị Shapley cho các mô hình học máy dạng cây trong thời gian đa thức $O(T L D^2)$, trong đó $T$ là số lượng cây, $L$ là số nút lá tối đa và $D$ là độ sâu tối đa của cây.
- **Thang đo đầu ra:** Đối với bài toán phân loại nhị phân bằng XGBoost, TreeSHAP tính toán đóng góp trên thang đo raw margin (log-odds), bảo toàn tính cộng tính chặt chẽ trước khi qua hàm phi tuyến sigmoid.
- **Cơ chế tính toán dựa trên đường đi (Tree-path dependent):** Ước lượng kỳ vọng có điều kiện $\mathbb{E}[f(x) \mid x_S]$ bằng cách lần theo tỷ lệ trọng số mẫu quan sát được tại các nút phân nhánh trong cấu trúc cây đã học, loại bỏ sự cần thiết phải tích hợp qua một tập dữ liệu nền ngoại suy nhân tạo.

---

## 1.6. Thách thức về độ bất ổn định của giải thích SHAP trong mô hình Ensemble

### 1.6.1. Hiện tượng bất ổn định giải thích (Explanation Instability)
Mặc dù SHAP đảm bảo tính duy nhất và công bằng đối với một mô hình cố định, trong thực tiễn triển khai chu trình học máy, mô hình không bao giờ là bất biến. Khi dữ liệu huấn luyện trải qua các biến động nhỏ (ví dụ: thay đổi mẫu ngẫu nhiên giữa các fold kiểm định chéo, bổ sung dữ liệu mới theo chu kỳ hàng tháng hoặc thay đổi hạt giống ngẫu nhiên `random_state`), cấu trúc phân nhánh của các cây quyết định trong XGBoost có thể thay đổi đáng kể.

Sự biến động cấu trúc cây dẫn đến hiện tượng **Bất ổn định của lời giải thích (Explanation Instability)**: thứ hạng quan trọng của các đặc trưng (Feature Importance Ranking) bị xáo trộn nghiêm trọng giữa các lần huấn luyện, mặc dù hiệu năng dự báo tổng thể (ROC-AUC) của mô hình hầu như không thay đổi.

### 1.6.2. Căn nguyên kỹ thuật của sự bất ổn định
1. **Tính đa cộng tuyến (Multicollinearity):** Trong dữ liệu tài chính, các biến số lịch sử tín dụng thường có tương quan nội tại rất cao (ví dụ: số dư hóa đơn `BILL_AMT1` đến `BILL_AMT6`, lịch sử trả nợ `PAY_0` đến `PAY_6`). Khi hai biến có tương quan mạnh, tiêu chuẩn phân nhánh (Gain) tại một nút lá có thể chênh lệch rất nhỏ; một biến động mẫu nhỏ trong tập huấn luyện có thể khiến cây chọn biến $A$ thay vì biến $B$, dẫn đến việc SHAP chuyển toàn bộ điểm đóng góp từ biến này sang biến kia.
2. **Cơ chế lấy mẫu ngẫu nhiên (Subsampling & Colsampling):** Các kỹ thuật lấy mẫu ngẫu nhiên hàng (`subsample`) và cột (`colsample_bytree`) được thiết kế để chống quá khớp nhưng đồng thời làm tăng tính ngẫu nhiên của các điểm cắt cây giữa các fold.
3. **Cấu trúc cây sâu và phức tạp:** Cấu hình XGBoost có độ sâu lớn (`max_depth` cao) và ít bị ràng buộc chính quy hóa sẽ học các phân mảnh cục bộ của dữ liệu huấn luyện, làm suy giảm tính nhất quán toàn cục của các giải thích.

### 1.6.3. Rủi ro nghiệp vụ và khoảng trống nghiên cứu
Hiện tượng bất ổn định giải thích tạo ra rủi ro nghiêm trọng cho các định chế tài chính:
- **Rủi ro tuân thủ pháp lý:** Nếu một khách hàng bị từ chối vay vốn vào tháng trước vì lý do "Thời gian trễ hạn thanh toán (`PAY_0`)", nhưng khi hệ thống cập nhật dữ liệu huấn luyện vào tháng sau lại từ chối với lý do "Hạn mức tín dụng quá thấp (`LIMIT_BAL`)", ngân hàng không thể giải trình một cách nhất quán trước cơ quan thanh tra giám sát.
- **Khoảng trống nghiên cứu hiện hữu:** Tuyệt đại đa số các nghiên cứu tối ưu hóa siêu tham số (HPO) truyền thống trong lĩnh vực rủi ro tín dụng chỉ tập trung duy nhất vào mục tiêu hiệu năng dự báo (ROC-AUC hoặc F1-score). Việc bỏ qua độ ổn định của giải thích trong quá trình tìm kiếm cấu hình khiến mô hình được chọn dễ rơi vào trạng thái quá khớp cục bộ về mặt diễn giải.

---

## 1.7. Mục tiêu nghiên cứu, câu hỏi nghiên cứu (RQ1–RQ4) và bố cục đề tài

### 1.7.1. Mục tiêu nghiên cứu
Đề tài hướng tới việc xây dựng và thực nghiệm một quy trình khoa học toàn diện nhằm giải quyết đồng thời hai yêu cầu xung đột trong mô hình hóa rủi ro tín dụng cá nhân:
1. **Tối đa hóa hiệu năng phân loại rủi ro vỡ nợ (ROC-AUC)** trên dữ liệu kiểm định chéo.
2. **Tối đa hóa độ ổn định của thứ hạng giải thích SHAP toàn cục (SHAP Stability)** qua các tập huấn luyện độc lập trên cùng một tập khách hàng tham chiếu cố định.
3. Áp dụng phương pháp tối ưu đa mục tiêu dựa trên giải thuật di truyền NSGA-II để khảo sát không gian đánh đổi (trade-off) Pareto, từ đó thiết lập quy trình lựa chọn nghiệm khách quan phục vụ triển khai thực tế.

### 1.7.2. Các câu hỏi nghiên cứu trọng tâm (Research Questions)
Để định hướng thực nghiệm và kiểm chứng khoa học, đề tài xác lập 4 câu hỏi nghiên cứu:
- **RQ1:** *Các siêu tham số điều khiển cấu trúc và chính quy hóa của XGBoost tác động như thế nào đến hiệu năng dự đoán (ROC-AUC) và độ ổn định của giải thích SHAP?*
- **RQ2:** *Cấu hình siêu tham số tối ưu đơn mục tiêu theo ROC-AUC bằng thuật toán TPE có đồng thời đảm bảo độ ổn định của giải thích SHAP hay không?*
- **RQ3:** *Thuật toán tối ưu hóa đa mục tiêu NSGA-II có thể tìm ra các cấu hình Pareto mang lại sự cân bằng vượt trội giữa hiệu năng dự đoán và độ ổn định giải thích so với phương pháp đơn mục tiêu và cấu hình mặc định hay không?*
- **RQ4:** *Các cấu hình XGBoost tối ưu đa mục tiêu đứng ở vị thế nào khi so sánh đối chuẩn (benchmark) với các mô hình học máy phổ biến khác trong quản trị rủi ro tín dụng (Logistic Regression, Random Forest, Gradient Boosting, LightGBM, CatBoost)?*

### 1.7.3. Đóng góp chính của đề tài
1. **Xây dựng thước đo định lượng độ ổn định giải thích SHAP:** Định nghĩa chỉ số Spearman stability trung bình ghép cặp giữa các vector độ quan trọng TreeSHAP toàn cục của 23 đặc trưng gốc qua 5-fold Cross-Validation trên một tập tham chiếu (Reference Set) 1.000 mẫu được cô lập nghiêm ngặt.
2. **Quy trình tối ưu hóa đa mục tiêu hai nhánh có kiểm soát:** Thực thi song song nhánh đơn mục tiêu TPE và nhánh đa mục tiêu NSGA-II trên cùng một không gian tìm kiếm 9 siêu tham số và cùng ngân sách đánh giá (64 trial hoàn thành = 320 lần fit và 320 lần tính TreeSHAP mỗi nhánh).
3. **Khung lựa chọn nghiệm Pareto không thiên lệch:** Thiết lập quy tắc chọn nghiệm tiên nghiệm (ưu tiên AUC, ưu tiên ổn định, và cân bằng khoảng cách chuẩn hóa min-max tới điểm lý tưởng) hoàn toàn trên tập Development.
4. **Kiểm toán độ nhạy và đánh giá độc lập trên Final Test:** Thực nghiệm kiểm tra độ nhạy qua 10 phân hoạch 5-fold mới (seed 101–110, 250 lần fit/TreeSHAP) và khóa giao thức đánh giá cuối cùng trên 6.000 mẫu Final Test độc lập, đảm bảo tính minh bạch và ngăn ngừa triệt để rò rỉ dữ liệu.

### 1.7.4. Bố cục tổng thể của báo cáo
Báo cáo đề tài được cấu trúc thành 3 chương chính:
- **Chương 1 — Tổng quan đề tài và cơ sở lý thuyết:** Trình bày bối cảnh rủi ro tín dụng, ma trận chi phí sai lầm, cơ sở thuật toán XGBoost, tối ưu hóa siêu tham số Bayes/NSGA-II, lý thuyết giá trị Shapley và thách thức bất ổn định giải thích.
- **Chương 2 — Phương pháp đề xuất:** Mô tả chi tiết kiến trúc giải pháp, giao thức phân chia dữ liệu cô lập chống rò rỉ, không gian siêu tham số, hai hàm mục tiêu toán học ($f_1, f_2$), thuật toán TreeSHAP trên biến phân loại đã nhóm, và quy tắc trích xuất nghiệm đại diện từ Pareto front.
- **Chương 3 — Thực nghiệm và đánh giá:** Trình bày kết quả đối chuẩn 6 mô hình baseline, lịch sử tối ưu TPE và NSGA-II, phân tích Pareto front, so sánh hiệu năng trên Final Test, kiểm định độ nhạy qua 10 phân hoạch mới, thảo luận trade-off, giới hạn nghiên cứu và trả lời chi tiết 4 câu hỏi nghiên cứu RQ1–RQ4.
