# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:2A202603566**
- **Họ và tên: Hoàng Lam Minh**
- **SEED** (in ở ô đầu notebook):3792

## 1. Bằng chứng

| Tệp | Có? |
|---|---|
| `day01_lab.ipynb` — notebook đã chạy, còn output | ☑ |
| `lab1_report.json` — do ô "Báo cáo cuối" sinh ra |  ☑ |
| `hand_calc.jpg` — ảnh tính tay có các bước |  ☑ |

## 2. Kết quả chính

| Phương pháp | F1 test | F1 SUB 40 | p95 (ms/tin) | USD/1k tin |
|---|---:|---:|---:|---:|
| BoW + Logistic Regression | 0.905 | 0.919 | 0.118 | 0.0 |
| BoW + MLP | 0.969 | 0.976 | 0.396 | 0.0 |
| MiniLM + kNN | 0.932 | 1.000 | 77.523 | 0.0 |
| MiniLM + Logistic Regression | 0.960 | 1.000 | 58.553 | 0.0 |
| MiniLM + MLP head | 0.970 | 1.000 | 57.789 | 0.0 |
| LLM nhỏ zero-shot | 0.930 | 0.930 | 2366.568 | 0.0 |
| LLM lớn zero-shot | 0.930 | 0.930 | 2393.564 | 0.0 |
| LLM nhỏ few-shot | 0.976 | 0.976 | 2385.092 | 0.0 |

**Nhận xét:** BoW + MLP có F1 test 0.969 và p95 0.396 ms/tin, đáp ứng yêu cầu độ trễ dưới 50 ms trong benchmark. Các phương pháp Embedding và LLM có độ trễ cao hơn giới hạn. Chi phí LLM hiển thị 0.0 có thể do notebook chưa khai báo bảng giá model, không có nghĩa API miễn phí.

## 3. Dự đoán và thực tế

**Dự đoán 1 – Learning rate = 50:** Mình dự đoán loss sẽ dao động mạnh hoặc không hội tụ. Thực tế, loss ban đầu tăng rất cao nhưng sau đó giảm nhanh xuống khoảng 0.005. Điều này cho thấy learning rate lớn có thể gây cập nhật mạnh nhưng mô hình vẫn hội tụ trong lần thử nghiệm này.

**Dự đoán 2 – Model LLM lớn tốt hơn model nhỏ:** Mình dự đoán model lớn có F1 cao hơn. Tuy nhiên, cả hai đều đạt F1 = 0.930 trên tập SUB. Điều này cho thấy model lớn hơn không phải lúc nào cũng mang lại kết quả phân loại tốt hơn.

**Dự đoán 3 – Prompt injection là nhóm khó nhất:** Mình dự đoán LLM dễ bị đánh lừa nhất bởi prompt injection. Thực tế, nhóm "ham looks spam" mới là nhóm khó hơn với LLM zero-shot, chỉ đạt 50% accuracy. Prompt phòng thủ giúp kết quả prompt injection tăng từ 83% lên 100%.

## 4. Phân tích 50 case khó

BoW + Logistic Regression tụt nhiều nhất, F1 giảm từ 0.919 xuống 0.378. BoW + MLP cũng giảm từ 0.976 xuống 0.488. Nguyên nhân là BoW chủ yếu dựa vào sự xuất hiện của từ khóa, chưa hiểu tốt ngữ cảnh và ý nghĩa câu.

Hai ví dụ cụ thể:

- **"FREE pizza in the office kitchen right now, come quick before it's gone"**: Đây là tin nhắn ham nhưng LLM vẫn phân loại nhầm do các từ "FREE" và cách diễn đạt thúc giục dễ bị hiểu là quảng cáo spam.
- **"Hope you're well! Quick reminder that your photos from the event are ready, view & order prints"**: Đây là spam nhưng LLM phân loại nhầm thành ham vì nội dung được viết tự nhiên, lịch sự và không có từ khóa spam rõ ràng.

Nhìn chung, BoW gặp khó với spam diễn đạt tự nhiên và các trường hợp né từ khóa. Embedding có thể hiểu ngữ nghĩa tốt hơn nhưng vẫn nhầm lẫn giữa tin nhắn nói về spam và tin nhắn thực sự là spam.

## 5. Thử thách cá nhân

Thử thách của mình là thay đổi threshold từ 0.5 thành 0.2 và 0.8 để so sánh Precision, Recall và F1.

| Threshold | Precision | Recall | F1 |
|---|---:|---:|---:|
| 0.2 | 0.920 | 0.939 | 0.929 |
| 0.5 | 1.000 | 0.827 | 0.905 |
| 0.8 | 1.000 | 0.663 | 0.798 |

Khi tăng threshold, Precision tăng nhưng Recall giảm vì mô hình thận trọng hơn khi phân loại tin nhắn là spam. Threshold 0.2 có F1 và Recall cao nhất. Nếu ưu tiên tránh chặn nhầm OTP, mình nghiêng về threshold 0.5 vì vẫn giữ Precision = 1.000 và Recall cao hơn threshold 0.8.

## 6. Phương án cho sản phẩm thật

Với yêu cầu xử lý 10 triệu SMS/ngày và p95 dưới 50 ms/tin, mình chọn BoW + MLP vì có F1 test = 0.969, F1 SUB = 0.976 và p95 = 0.396 ms/tin. BoW + Logistic Regression cũng đáp ứng độ trễ nhưng có F1 thấp hơn, ở mức 0.905.

Mình chấp nhận đánh đổi thời gian huấn luyện và độ phức tạp cao hơn của MLP để có chất lượng phân loại tốt hơn. Tuy nhiên, stress test cho thấy mô hình vẫn cần cải thiện trước khi triển khai thực tế.

Nếu sử dụng LLM cho 300 triệu tin nhắn/tháng, với mức giá giả định 0.05 USD/1.000 tin, chi phí sẽ khoảng 15.000 USD/tháng. Vì vậy, mình đề xuất dùng mô hình cục bộ để xử lý chính và chỉ sử dụng LLM để kiểm tra bổ sung bất đồng bộ cho các trường hợp khó.

## 7. AI log và điều đã học

| Phần | Bạn hỏi AI gì (prompt) | AI trả lời/viết gì | Bạn đã sửa / kiểm chứng gì |
|---|---|---|---|
| Benchmark | So sánh các mô hình dựa trên F1, p95 và chi phí để chọn mô hình lọc SMS. | Phân tích ưu nhược điểm và đề xuất mô hình phù hợp. | Đối chiếu số liệu với kết quả notebook. |
| Stress test | Giải thích vì sao mô hình giảm F1 trên 50 case khó. | Phân tích lỗi do từ khóa, ngữ cảnh và cách diễn đạt. | Kiểm tra kết quả theo từng nhóm case. |
| Threshold | Giải thích sự thay đổi Precision, Recall và F1 khi thay đổi threshold. | Phân tích sự đánh đổi giữa chặn nhầm và bỏ sót spam. | Đối chiếu với kết quả thực nghiệm. |
| Báo cáo | Hỗ trợ tổng hợp kết quả và viết báo cáo Day 01. | Đề xuất cách trình bày các phần kết quả và nhận xét. | Kiểm tra và chỉnh sửa theo số liệu notebook. |

**Điều mình học được hôm nay:**

Mình hiểu rằng F1 cao trên tập kiểm thử thông thường chưa đảm bảo mô hình hoạt động tốt với dữ liệu thực tế. Việc lựa chọn mô hình cần cân nhắc đồng thời chất lượng, độ trễ, throughput và chi phí. Ngoài ra, threshold có ảnh hưởng trực tiếp đến Precision và Recall nên cần lựa chọn theo yêu cầu nghiệp vụ.
