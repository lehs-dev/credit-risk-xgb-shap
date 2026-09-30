# Kiểm tra độ nhạy sau lựa chọn nghiệm trên Development

## Quy tắc khóa trước khi chạy

Ba nghiệm Pareto đã chọn bằng quy tắc ở Chương 2 được giữ nguyên: trial 44 (ưu tiên AUC), trial 57 (ưu tiên stability) và trial 61 (cân bằng). Hai đối chứng là XGBoost tối ưu ROC-AUC bằng TPE (trial 50) và XGBoost mặc định. Không điều chỉnh siêu tham số, không đổi vai trò nghiệm và không dùng kết quả kiểm tra độ nhạy để chọn lại cấu hình.

Kiểm tra này giữ nguyên Development/Test 80/20, Core 23.000 quan sát và tập tham chiếu SHAP 1.000 quan sát đã tách khỏi Core. Trên cùng Core, tạo **10 phân hoạch StratifiedKFold mới**, mỗi phân hoạch gồm năm fold, dùng lần lượt seed 101–110. Cùng một phân hoạch được áp dụng cho cả năm cấu hình. Mỗi fold fit bộ tiền xử lý và XGBoost chỉ trên train Core, tính ROC-AUC trên validation Core, rồi tính TreeSHAP trên đúng 1.000 quan sát tham chiếu. Seed XGBoost giữ ở 42; nguồn biến thiên cần kiểm tra là cách chia Core thành các tập train/validation.

Với mỗi cấu hình và seed phân hoạch, lưu ROC-AUC trung bình của năm fold, độ ổn định Spearman giữa 10 cặp vector quan trọng SHAP, Jaccard top-5, năm AUC theo fold và thời gian. Tổng ngân sách là 5 cấu hình × 10 phân hoạch × 5 fold = **250 lần fit và 250 lần tính TreeSHAP**. Test không tham gia bất kỳ phép tính nào ở bước này.

## Cách tổng hợp

Đơn vị tổng hợp là **phân hoạch năm fold**, không phải từng fold hoặc từng cặp Spearman. Với mỗi cấu hình, báo trung bình, độ lệch chuẩn, giá trị nhỏ nhất và lớn nhất qua 10 phân hoạch cho từng chỉ số. Tính chênh lệch ghép cặp theo cùng seed phân hoạch giữa mỗi nghiệm Pareto và hai đối chứng; tổng hợp các chênh lệch này bằng cùng bốn đại lượng. So sánh mô tả sự nhạy cảm với cách chia Core, không được xem là một kiểm định độc lập trên 10 bộ dữ liệu khác nhau vì các phân hoạch dùng chung quan sát và reference.

Nếu thứ tự các cấu hình thay đổi giữa các phân hoạch, báo cáo hiện tượng đó cùng độ biến thiên, giữ nguyên ba nghiệm đã khóa. Bước này chỉ kiểm tra độ nhạy có điều kiện theo **tập tham chiếu cố định**; nó không kiểm tra độ nhạy khi thay tập tham chiếu hoặc dịch chuyển phân phối dữ liệu. Sau khi ghi nhận kết quả, mới tiến hành đánh giá cuối trên Test độc lập theo giao thức đã định.

## Kết quả trên Development

Lượt chạy hoàn tất **50 cặp seed/cấu hình**, tức **250 lần fit XGBoost và 250 lần tính TreeSHAP**. Bảng dưới tổng hợp theo 10 phân hoạch năm fold; số sau dấu ± là độ lệch chuẩn với `ddof=0`, khoảng trong ngoặc là giá trị nhỏ nhất–lớn nhất. Đây là kết quả CV trên Core, chưa phải kết quả Final Test.

| Cấu hình đã khóa | ROC-AUC CV | Spearman SHAP | Jaccard top-5 |
|---|---:|---:|---:|
| Pareto ưu tiên AUC (trial 44) | 0.787594 ± 0.000682 (0.786312–0.788426) | 0.933913 ± 0.015876 (0.905040–0.956225) | 0.782857 ± 0.092954 (0.652381–1.000000) |
| Pareto cân bằng (trial 61) | 0.787178 ± 0.000542 (0.786309–0.787965) | 0.947777 ± 0.010718 (0.929743–0.961660) | 0.699524 ± 0.037219 (0.652381–0.766667) |
| Pareto ưu tiên ổn định (trial 57) | 0.776891 ± 0.000393 (0.776271–0.777447) | 0.972586 ± 0.002856 (0.966537–0.976326) | 0.873333 ± 0.095219 (0.733333–1.000000) |
| TPE tối ưu AUC (trial 50) | 0.787643 ± 0.000585 (0.786638–0.788505) | 0.911472 ± 0.017231 (0.879842–0.941008) | 0.767619 ± 0.047562 (0.676190–0.866667) |
| XGBoost mặc định | 0.774809 ± 0.001354 (0.772981–0.776938) | 0.837589 ± 0.039689 (0.748814–0.906028) | 0.790000 ± 0.047258 (0.733333–0.866667) |

Chênh lệch dưới đây được tính **trong cùng seed**, lấy nghiệm Pareto trừ đối chứng. Bảng kết quả lưu cả độ lệch chuẩn và khoảng min–max của từng chênh lệch.

| Nghiệm Pareto | Đối chứng | Δ ROC-AUC trung bình | Δ Spearman SHAP trung bình |
|---|---|---:|---:|
| Ưu tiên AUC | TPE | -0.000049 | +0.022441 |
| Cân bằng | TPE | -0.000465 | +0.036304 |
| Ưu tiên ổn định | TPE | -0.010752 | +0.061113 |
| Ưu tiên AUC | XGBoost mặc định | +0.012786 | +0.096324 |
| Cân bằng | XGBoost mặc định | +0.012369 | +0.110188 |
| Ưu tiên ổn định | XGBoost mặc định | +0.002082 | +0.134997 |

So với TPE, cả ba nghiệm Pareto có Spearman SHAP cao hơn trong cả **10/10 phân hoạch**. Nghiệm ưu tiên AUC có AUC trung bình gần TPE, nhưng chênh lệch AUC đổi dấu giữa các seed (4 seed cao hơn, 6 seed thấp hơn). Nghiệm cân bằng có AUC thấp hơn TPE ở 9/10 seed. Nghiệm ưu tiên ổn định thể hiện đánh đổi rõ nhất: Spearman cao nhất và AUC thấp hơn TPE ở cả 10 seed. Chỉ số Jaccard top-5 không đi cùng chiều với Spearman trong mọi cấu hình; ví dụ nghiệm cân bằng có Jaccard trung bình 0.699524, thấp hơn TPE 0.767619. Vì vậy, kết luận về độ ổn định ở đây gắn với **mục tiêu Spearman đã định**, còn Jaccard là chỉ số bổ sung.

Các số liệu chi tiết và dấu vết tái lập nằm ở `artifacts/tables/xgb_sensitivity_repeats.csv`, `xgb_sensitivity_summary.csv`, `xgb_sensitivity_paired.csv`, `xgb_sensitivity_paired_summary.csv` và `xgb_sensitivity_manifest.json`. Manifest ghi `final_test_evaluated=false`. Các phân hoạch dùng chung Core và tham chiếu SHAP nên 10 dòng mỗi cấu hình không phải 10 bộ dữ liệu độc lập; không suy diễn ý nghĩa thống kê hay hiệu quả trên Final Test từ bảng này.
