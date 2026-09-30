# Đánh giá dự đoán cuối trên Final Test

## Giao thức khóa trước khi xem kết quả Test

Danh sách cấu hình được cố định từ lựa chọn Pareto và nhánh TPE trên Development: Pareto ưu tiên AUC (trial 44), Pareto ưu tiên ổn định SHAP (trial 57), Pareto cân bằng (trial 61), TPE tối ưu ROC-AUC (trial 50) và XGBoost mặc định. Kết quả kiểm tra độ nhạy trên 10 cách chia Core chỉ dùng để mô tả độ biến thiên; không thay danh sách, siêu tham số hay vai trò của các cấu hình. Mọi mã trial và fingerprint phải khớp các bảng HPO, Pareto và độ nhạy đã lưu.

Mỗi cấu hình được huấn luyện lại **một lần** trên toàn bộ 24.000 quan sát Development. Bộ tiền xử lý được fit trên Development rồi áp dụng cho 6.000 quan sát Final Test. Việc dùng lại 1.000 quan sát SHAP reference trong lần fit cuối là hợp lệ sau khi đã khóa cấu hình; reference được tách khỏi Core để đánh giá mục tiêu trong giai đoạn chọn mô hình, còn lần này chỉ đánh giá dự đoán. Không fit encoder, scaler hoặc mô hình trên Test; không dùng nhãn Test để chỉnh tham số, ngưỡng hay chọn lại nghiệm.

Chỉ số chính là ROC-AUC. Chỉ số bổ sung gồm Average Precision (AP), Precision, Recall, F1, Balanced Accuracy, Brier score và ma trận nhầm lẫn. Các chỉ số phân loại dùng ngưỡng xác suất **0,5** cố định từ trước; AP dùng `average_precision_score`. Báo tỷ lệ default của Test để đặt AP trong bối cảnh tỷ lệ lớp. Các chỉ số Test được tính một lần cho từng cấu hình và chỉ dùng để báo cáo/so sánh các cấu hình đã khóa. Không tính độ ổn định SHAP trên một mô hình refit duy nhất và không tính SHAP trên Test; mục tiêu ổn định đã được đánh giá bằng Core CV với tập tham chiếu cố định.

Manifest đánh giá cuối ghi rõ nguồn cấu hình, fingerprint của giao thức, số mẫu và trạng thái hoàn tất. Nếu đầu vào thay đổi hoặc đã có kết quả Test, quy trình dừng để tránh trộn hai giao thức hay vô tình lặp đánh giá. Bảng kết quả Test được trình bày cùng các giá trị Development CV, nhưng không dùng chênh lệch Test để tạo thêm vòng tối ưu. Một phép chia Test duy nhất cho phép ước lượng hiệu năng trên tập giữ lại này; chưa đủ để khẳng định khả năng khái quát cho mọi danh mục tín dụng hoặc mọi thời điểm.

## Kết quả đánh giá đã khóa

Lượt chấm duy nhất hoàn tất năm lần fit trên **24.000 mẫu Development** và năm lần đánh giá trên cùng **6.000 mẫu Test**. Test có 1.327 trường hợp default và 4.673 trường hợp không default (tỷ lệ default **22,1167%**). Bảng này trình bày kết quả Test; cột Spearman SHAP là **trung bình kiểm tra độ nhạy trên Development** ở bước trước, không phải chỉ số tính trên Test.

| Cấu hình | Spearman SHAP trên Development | ROC-AUC Test | AP Test | Brier Test |
|---|---:|---:|---:|---:|
| Pareto ưu tiên AUC (trial 44) | 0.933913 | 0.779784 | 0.557473 | 0.134759 |
| Pareto ưu tiên ổn định (trial 57) | 0.972586 | 0.768113 | 0.537826 | 0.144236 |
| Pareto cân bằng (trial 61) | 0.947777 | 0.781791 | 0.559020 | 0.134622 |
| TPE tối ưu AUC (trial 50) | 0.911472 | 0.781011 | 0.558185 | 0.134681 |
| XGBoost mặc định | 0.837589 | 0.769664 | 0.539973 | 0.137585 |

Các chỉ số phụ thuộc ngưỡng dưới đây dùng cùng ngưỡng xác suất **0,5**. `TN/FP/FN/TP` lần lượt là đúng âm, dương giả, âm giả và đúng dương.

| Cấu hình | Precision | Recall | F1 | Balanced Accuracy | TN/FP/FN/TP |
|---|---:|---:|---:|---:|---:|
| Pareto ưu tiên AUC | 0.665260 | 0.356443 | 0.464181 | 0.652756 | 4435 / 238 / 854 / 473 |
| Pareto ưu tiên ổn định | 0.750000 | 0.149209 | 0.248900 | 0.567543 | 4607 / 66 / 1129 / 198 |
| Pareto cân bằng | 0.664773 | 0.352675 | 0.460857 | 0.651086 | 4437 / 236 / 859 / 468 |
| TPE tối ưu AUC | 0.667586 | 0.364732 | 0.471735 | 0.656580 | 4432 / 241 / 843 / 484 |
| XGBoost mặc định | 0.663087 | 0.372268 | 0.476834 | 0.659278 | 4422 / 251 / 833 / 494 |

Trong năm cấu hình đã khóa, nghiệm Pareto cân bằng có ROC-AUC Test và AP Test cao nhất, còn Brier thấp nhất; mức chênh với TPE nhỏ (ROC-AUC **+0,000780**, AP **+0,000835**, Brier **−0,000060**). Nghiệm ưu tiên ổn định có Spearman SHAP trên Development cao nhất nhưng ROC-AUC Test thấp hơn và recall ở ngưỡng 0,5 chỉ **0,149209**. Điều này cho thấy sự đánh đổi khi ưu tiên độ ổn định giải thích, không phải lý do để điều chỉnh lại siêu tham số hoặc ngưỡng theo Test. Các chỉ số phụ thuộc ngưỡng và Brier được báo cáo để mô tả thêm; không tham gia quy tắc chọn nghiệm ban đầu.

Bảng đầy đủ nằm ở `artifacts/tables/xgb_final_test_summary.csv`; `xgb_final_test_manifest.json` ghi fingerprint, năm cấu hình, `status=complete`, `final_test_evaluated=true`, `test_used_for_selection=false` và `shap_computed_on_test=false`. Lệnh `python scripts/08_run_final_test.py --verify` kiểm tra lại artifact mà không fit hay chấm Test lần nữa. Đây là một tập Test giữ lại duy nhất; các chênh lệch nhỏ giữa cấu hình chỉ được diễn giải mô tả, không suy ra ý nghĩa thống kê hay bảo đảm hiệu năng trên dữ liệu tín dụng khác.
