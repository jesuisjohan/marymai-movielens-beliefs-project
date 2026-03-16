# MovieLens Beliefs – Dự đoán Belief Rating

**Tiểu luận cá nhân** – Môn học: Giới thiệu về Máy học  
Đại học Kinh tế – Luật (UEL)

Notebook Google Colab dự đoán **belief rating** (đánh giá kỳ vọng của người dùng với phim chưa xem) dựa trên bộ dữ liệu [MovieLens Beliefs](https://grouplens.org/datasets/movielens/).

---

## Cấu trúc theo Đề Cương Thi

| Mục | Nội dung | Điểm |
|-----|----------|------|
| **1. Giới thiệu về Data** | Mô tả dataset, thống kê mô tả, phân phối, heatmap tương quan | 1.0 |
| **2. Giới thiệu Bài Toán** | Hồi quy (Regression), lý do chọn mô hình, sơ đồ workflow | 0.5 |
| **3. Xử lý Dữ Liệu** | NaN, trùng lặp, kiểm tra hợp lý, feature engineering, tách train/val/test | 2.0 |
| **4. Mô hình Hóa** | RQ1 (Belief Gap), RQ2 (MF Decomposition), RQ3 (Belief→Actual) | 4.0 |
| **5. Đánh giá Model** | Dự báo mẫu, so sánh RMSE/MAE, phân tích lỗi, ablation leakage | 2.0 |
| **6. Kết luận chung** | Trả lời 3 RQ, hạn chế, hướng phát triển | 0.5 |
| **Tổng** | | **10.0** |

---

## Ba Câu Hỏi Nghiên Cứu

### RQ1 – Belief Gap theo Thể Loại
> Người dùng có xu hướng kỳ vọng cao hơn hay thấp hơn so với đánh giá thực tế của cộng đồng, và xu hướng này khác nhau như thế nào giữa các thể loại phim?

**Phương pháp**: Tính `belief_gap = userPredictRating - movie_mean_rating`, phân tích theo genre và theo mức độ certainty.  
**Kết quả mong đợi**: Khám phá genre nào người dùng *overestimate* (lạc quan) hoặc *underestimate* (thận trọng).

---

### RQ2 – Phân tích Nguồn Gốc Belief
> Belief rating chủ yếu được quyết định bởi xu hướng cá nhân (user bias), danh tiếng phim (item popularity), hay tương tác phức tạp user-item (latent factors)?

**Phương pháp**: Tăng dần độ phức tạp mô hình và đo RMSE:

| Level | Mô hình | Thành phần |
|-------|---------|-----------|
| L0 | Global Mean | μ |
| L1 | User Bias | μ + b_u |
| L2 | Additive Bias | μ + b_u + b_i |
| L3 | BiasedMF (SGD) | μ + b_u + b_i + p_u·q_i |
| L4 | SVD++ | μ + b_u + b_i + (p_u + implicit)·q_i |

**Kết quả mong đợi**: Xác định thành phần nào giảm RMSE nhiều nhất.

---

### RQ3 – Belief có giá trị để Dự đoán Actual Rating không?
> Belief rating có giá trị như một feature bổ sung để dự đoán actual rating không?

**Phương pháp**:
1. Tìm *matched pairs*: belief (isSeen=0) + actual rating (belief_tstamp < actual_tstamp)
2. So sánh Ridge/RF có và không có `belief_rating` làm feature
3. Đo RMSE improvement

**Kết quả mong đợi**: Kiểm định xem belief có thực sự cải thiện dự đoán actual rating.

---

## Dataset

Đặt tất cả file CSV trong cùng một thư mục (ví dụ `data/`):

| File | Mô tả | Cột quan trọng |
|------|-------|---------------|
| `belief_data.csv` | Elicitation – 1 dòng = 1 (user, movie) | `userId`, `movieId`, `isSeen`, `userPredictRating`, `userCertainty`, `systemPredictRating`, `tstamp` |
| `user_rating_history.csv` | Lịch sử đánh giá thực tế | `userId`, `movieId`, `rating`, `timestamp` |
| `movies.csv` | Metadata phim | `movieId`, `title`, `genres` |
| `user_recommendation_history.csv` | Log gợi ý (tùy chọn) | `userId`, `movieId`, `predictedRating`, `tstamp` |
| `movie_elicitation_set.csv` | Bộ elicitation (tùy chọn) | `movieId`, `month_idx`, `source` |

---

## Cấu trúc Dự Án

```
.
├── notebooks/
│   └── belief_prediction.ipynb   # Notebook chính (tất cả sections)
├── requirements.txt               # Python dependencies
└── README.md
```

---

## Hướng dẫn Chạy trên Google Colab

1. Mở [Google Colab](https://colab.research.google.com/).
2. Upload hoặc clone repository này.
3. Mở `notebooks/belief_prediction.ipynb`.
4. Chỉnh biến `DATA_DIR` ở **Section 0** trỏ tới thư mục chứa file CSV:
   - **Google Drive**: Mount Drive và set `DATA_DIR = "/content/drive/MyDrive/your_folder"`.
   - **Upload trực tiếp**: Upload file và set `DATA_DIR = "/content"`.
5. Chạy tất cả cells (`Runtime → Run all`).

> Nếu file CSV chưa có → notebook in hướng dẫn và **không bị crash**.

---

## Nguyên tắc Chống Data Leakage

- **`systemPredictRating`** là output của recommender system nội bộ. Dùng làm feature = *policy leakage*. Các mô hình chính **không** dùng cột này.
- Chỉ dùng trong **Section 5.4 (Ablation Demo)** để minh họa tác hại của leakage.
- **`predictedRating`** từ recommendation log cũng bị loại trừ tương tự.

---

## Phân chia Dữ Liệu Theo Thời Gian

Random split làm phồng metric do autocorrelation theo thời gian. Notebook dùng **per-user chronological split** trên `tstamp`:

- **Train**: 70% sớm nhất của mỗi user
- **Validation**: 15% tiếp theo
- **Test**: 15% mới nhất

Điều này mô phỏng deployment thực tế: model huấn luyện trên lịch sử, đánh giá trên tương lai.

---

## Các Mô Hình và Vai Trò

| Mô hình | Vai trò | RQ |
|---------|---------|-----|
| Global Mean | Baseline L0 | RQ2 |
| User Bias | Baseline L1 | RQ2 |
| Additive Bias | Baseline L2 | RQ2 |
| BiasedMF (SGD) | Mô hình chính: latent factors | RQ2 |
| SVD++ | Mở rộng MF với implicit feedback | RQ2 |
| Ridge Regression | Feature-based, so sánh ±belief | RQ3 |
| Random Forest | Non-linear, feature importance | RQ3 |

---

## Deliverables

1. **Data**: CSV files (không commit vào repo)
2. **Code**: `notebooks/belief_prediction.ipynb`
3. **Report**: Báo cáo PDF theo 6 mục của đề cương

---

## Dependencies

```bash
pip install -r requirements.txt
```

Core: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `tqdm`

---

## Hạn Chế

- **Selection bias**: Chỉ user tham gia elicitation study.
- **MNAR**: `isSeen = -1` là non-response, loại bỏ (không phải missing ngẫu nhiên).
- **Leakage**: `systemPredictRating` bị loại khỏi mô hình chính.
- **Cold-start**: Chỉ dùng title + genres, không có external metadata.
- **Matched pairs**: Số lượng có thể nhỏ tùy dataset.
