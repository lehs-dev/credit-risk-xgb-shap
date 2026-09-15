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
│   ├── metrics.py            # Tính ROC-AUC, PR-AUC, F1, Balanced Accuracy...
│   ├── stability.py          # Thước đo Spearman, Jaccard, Kendall's W cho SHAP
│   └── utils.py              # Tiện ích logging, timer, seeding
├── scripts/                  # Scripts thực thi từng giai đoạn
│   ├── 00_prepare_data.py    # Kiểm định dữ liệu thô
│   ├── 01_make_splits.py     # Sinh các tập split cố định
│   └── 02_run_baselines.py   # Benchmark 6 mô hình baseline
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
# Nạp và sao chép dữ liệu thô vào data/raw/
./run.sh data

# Kiểm định dữ liệu thô và xuất tóm tắt thống kê
./run.sh prepare

# Khóa protocol phân chia mẫu (Outer Split 80/20, 5-Fold CV, SHAP Reference Set)
./run.sh split

# Chạy toàn bộ bộ kiểm thử tự động (14 bài test chống rò rỉ và toán học)
./run.sh test

# Chạy benchmark 6 mô hình đối chứng (Logistic Regression, RF, GBM, LightGBM, CatBoost, XGBoost)
./run.sh baseline
```

---

## 🔒 Giao thức Chống Rò Rỉ (Strict Anti-Leakage Protocol)

1. **Outer Split**: Tách riêng 20% Final Test set ($N=6.000$) độc lập tuyệt đối. Không sử dụng cho bất kỳ bước tuning nào.
2. **Development Set**: 80% ($N=24.000$) được dùng cho 5-Fold Stratified Cross Validation.
3. **SHAP Reference Set**: 1.000 mẫu được trích xuất phân tầng từ Development Set và cố định xuyên suốt mọi trial HPO.
4. **Cô lập Fit/Transform**: Các phép chuẩn hóa scaler và encoder chỉ được fit trên fold train và transform trên fold validation/test.
