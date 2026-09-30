# Credit Risk Modeling with Multi-Objective XGBoost HPO & SHAP Stability

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Dự án nghiên cứu: **"Tối ưu đa mục tiêu siêu tham số XGBoost cho dự đoán rủi ro tín dụng cá nhân xét đến hiệu năng dự đoán và độ ổn định của giải thích SHAP"**.

---

## 📌 Tổng quan Nghiên cứu

- **Tác giả**: Lê Hoàng Sơn, Phạm Thiên Phúc
- **GVHD**: TS. Phan Tấn Quốc
- **Bộ dữ liệu chính**: *Default of Credit Card Clients* (UCI / Yeh & Lien, 2009) gồm 30.000 hồ sơ tín dụng.
- **Bài toán**: Tối ưu đa mục tiêu siêu tham số XGBoost:
  $$\max F(\theta) = [f_1(\theta), f_2(\theta)] = [\text{ROC-AUC}_{CV}(\theta), \text{SHAP-Stability}(\theta)]$$
  trong đó độ ổn định giải thích được định lượng bằng tương quan thứ hạng Spearman ($S$) của global SHAP feature importance qua nhiều lần lặp huấn luyện trên cùng một tập tham chiếu (reference set) cố định.

---

## 📂 Cấu trúc Thư mục

```text
credit-risk-xgb-shap/
├── configs/                  # File cấu hình YAML (data, baselines, search space)
├── data/
│   ├── raw/                  # Dữ liệu gốc (default_credit_card.csv)
│   ├── processed/            # Báo cáo thống kê & dữ liệu tiền xử lý
│   └── splits/               # Giao thức chia tập (outer_split, cv_folds, shap_reference)
├── notebooks/                # Jupyter Notebooks (01_eda.ipynb, ...)
├── src/creditrisk/           # Mã nguồn thư viện tái sử dụng
│   ├── config.py             # Nạp và kiểm định cấu hình
│   ├── data.py               # Tải và xác thực dữ liệu
│   ├── preprocessing.py      # Pipeline tiền xử lý chống rò rỉ (no-leakage)
│   ├── splits.py             # Phân chia mẫu và trích xuất SHAP reference set
│   ├── models.py             # Model factory (LR, RF, GBM, LightGBM, CatBoost, XGBoost)
│   ├── metrics.py            # Tính ROC-AUC, AP, F1, Balanced Accuracy...
│   ├── stability.py          # Thước đo Spearman, Jaccard, Kendall's W cho SHAP
│   └── utils.py              # Tiện ích logging, timer, seeding
├── scripts/                  # Scripts thực thi từng giai đoạn
│   ├── 00_prepare_data.py    # Kiểm định dữ liệu thô
│   ├── 01_make_splits.py     # Sinh các tập split cố định
│   ├── 02_run_baselines.py   # Benchmark 6 mô hình baseline
│   └── 03_run_xgb_default.py # ROC-AUC và SHAP stability của XGBoost mặc định
├── artifacts/                # Kết quả thí nghiệm, models, bảng, figures
├── tests/                    # Bộ kiểm thử tự động (anti-leakage, splits, stability)
├── pyproject.toml            # Cấu hình đóng gói package & pytest
├── requirements.txt          # Danh sách thư viện phụ thuộc
├── Makefile / run.sh         # Lệnh chạy tự động hóa
└── README.md
```

---

## 🚀 Hướng dẫn Cài đặt & Sử dụng

### 1. Kích hoạt môi trường ảo
```bash
source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-deps --no-build-isolation
```

### 2. Các lệnh thực thi nhanh (qua `./run.sh` hoặc `make`)

```bash
# Dùng dữ liệu đã đặt trong data/raw/default_credit_card.csv
./run.sh data

# Kiểm định dữ liệu thô và xuất tóm tắt thống kê
./run.sh prepare

# Khóa protocol phân chia mẫu (Outer Split 80/20, SHAP Reference Set, 5-Fold CV trên Core)
./run.sh split

# Chạy toàn bộ bộ kiểm thử tự động (bộ kiểm thử chống rò rỉ và toán học)
./run.sh test

# Chạy benchmark 6 mô hình đối chứng (Logistic Regression, RF, GBM, LightGBM, CatBoost, XGBoost)
./run.sh baseline

# Chạy thử hai mục tiêu của XGBoost mặc định trên cùng 5 fold Core
.venv/bin/python scripts/03_run_xgb_default.py

# Pilot riêng: 8 trial mỗi nhánh để đo thời gian (có cấu hình nhẹ và nặng)
python scripts/04_run_single_hpo.py --pilot
python scripts/05_run_multi_hpo.py --pilot

# Lượt chính: 64 trial hoàn thành mỗi nhánh, tách khỏi pilot
python scripts/04_run_single_hpo.py
python scripts/05_run_multi_hpo.py

# Kiểm tra Pareto độc lập và chọn ba nghiệm đại diện trên Development
python scripts/06_verify_pareto.py

# Xác thực đầu vào, chạy thử 5 fold, rồi kiểm tra độ nhạy 10 lần chia Core
python scripts/07_run_sensitivity.py --dry-run
python scripts/07_run_sensitivity.py --smoke
python scripts/07_run_sensitivity.py

# Khóa đầu vào và đánh giá một lần trên Final Test sau khi đã cố định cấu hình
python scripts/08_run_final_test.py --dry-run
python scripts/08_run_final_test.py
python scripts/08_run_final_test.py --verify
```

---

## 🔒 Giao thức Chống Rò Rỉ (Strict Anti-Leakage Protocol)

1. **Outer Split**: Tách riêng 20% Final Test set ($N=6.000$) độc lập tuyệt đối. Không sử dụng cho bất kỳ bước tuning nào.
2. **Development Set**: 80% ($N=24.000$) được chia thành Core và SHAP Reference Set rời nhau.
3. **SHAP Reference Set**: 1.000 mẫu được trích xuất phân tầng từ Development Set (seed 42) và giữ cố định để tính SHAP; không tham gia huấn luyện hoặc validation.
4. **Core Set**: 23.000 mẫu còn lại được dùng cho 5-Fold Stratified Cross Validation. Mỗi fold chỉ chứa các chỉ số của Core.
5. **Cô lập Fit/Transform**: Scaler và encoder chỉ được fit trên phần huấn luyện của từng fold, sau đó transform phần validation hoặc test.

Bảng `artifacts/tables/baseline_benchmark.csv` được tạo lại bằng 5-fold CV trên Core. Cột `average_precision` là Average Precision (AP), không phải diện tích đường Precision–Recall tính bằng quy tắc hình thang.

Dữ liệu đầu vào là bản CSV của [UCI Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default%2Bof%2Bcredit%2Bcard%2Bclients), đặt tại `data/raw/default_credit_card.csv` với cột nhãn `default.payment.next.month`.

Bản nháp phương pháp tính SHAP và độ ổn định nằm tại [`docs/chuong2_phuong_phap_shap.md`](docs/chuong2_phuong_phap_shap.md).

Bản nháp mục [2.9 về HPO](docs/chuong2_hpo_phuong_phap.md) mô tả TPE, NSGA-II, ngân sách và cách chọn Pareto. Mỗi trial ở cả hai nhánh đều fit 5 mô hình Core CV và tính SHAP trên cùng 1.000 quan sát tham chiếu; nhánh đơn mục tiêu chỉ đưa ROC-AUC vào sampler. SQLite lưu ở `artifacts/optuna/hpo.sqlite3`; các bảng trial, best/Pareto và manifest được xuất vào `artifacts/tables/`. Khi chạy lại cùng study, script chạy thêm đến đủ số trial `COMPLETE` đã cấu hình. Lịch sử trial được giữ, nhưng chuỗi đề xuất sau khi khởi tạo lại sampler có thể khác một lượt chạy liên tục.

Bản nháp [mục 2.10–2.11 về Pareto và chọn nghiệm](docs/chuong2_pareto_phuong_phap.md) giải thích quy tắc chi phối và ba vai trò lựa chọn. Script 06 xuất các bảng `xgb_pareto_audit.csv`, `xgb_pareto_selected.csv`, `xgb_pareto_comparison.csv` và `xgb_pareto_checks.csv` vào `artifacts/tables/`.

[Giao thức kiểm tra độ nhạy](docs/kiem_tra_do_nhay.md) giữ cố định ba nghiệm Pareto, nghiệm TPE tốt nhất và XGBoost mặc định. Script 07 đánh giá cùng năm cấu hình trên 10 cách chia 5 fold mới của Core (seeds 101–110), dùng chung tập tham chiếu SHAP; không dùng Final Test. `--dry-run` chỉ xác thực đầu vào, `--smoke` ghi kết quả riêng cho một cấu hình trên một seed. Lượt đầy đủ có thể tiếp tục từ các cặp seed/cấu hình đã lưu trong `artifacts/tables/`.

[Giao thức đánh giá Final Test](docs/danh_gia_final_test.md) fit lại năm cấu hình đã khóa trên toàn bộ Development, rồi đo ROC-AUC và các chỉ số bổ sung một lần trên 6.000 quan sát Test. `--dry-run` chỉ kiểm tra đầu vào, không phân tích nhãn Test; sau lượt chấm, `--verify` xác thực artifact mà không fit lại hoặc chấm lại.
