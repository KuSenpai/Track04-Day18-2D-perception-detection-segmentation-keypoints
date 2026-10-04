# Báo cáo bonus — Lab Ngày 18

Số liệu lấy từ lần chạy trên Colab (T4, ultralytics 8.4.171, 40 epoch, imgsz 640, seed 0).

## 4C — Val gốc và val lật gương

| Model | Pose mAP50-95 — val gốc | Pose mAP50-95 — val lật gương |
|---|---|---|
| flip_idx giải phẫu | 0.457 | 0.439 |
| flip_idx đồng nhất | 0.417 | 0.298 |

Với model `flip_idx` đồng nhất, trên val lật gương Pose mAP50 chỉ còn 0.878 (box mAP50-95 vẫn 0.894 nên phát hiện hổ không hỏng).

**Metric nào đã che lỗi?** Trên val gốc (toàn hổ quay phải) hai model chỉ chênh 0.04 mAP50-95 (0.457 so với 0.417), và box mAP gần như không bị ảnh hưởng, nên nhìn val gốc thì `flip_idx` đồng nhất trông "gần như ổn". Lỗi chỉ lộ rõ khi hổ quay trái: Pose mAP50-95 của model đồng nhất tụt từ 0.417 xuống 0.298 (giảm khoảng 29%), còn model giải phẫu chỉ tụt từ 0.457 xuống 0.439 (khoảng 4%). Metric bị che nhất là các metric đo ở ngưỡng lỏng và metric box: chúng không phân biệt chân trái với chân phải. Pose mAP50-95 (chấm bằng OKS) nhạy hơn nhưng vẫn cần dữ liệu có cả hai hướng mới thấy lỗi.

**Thiết kế tập val tốt hơn.** Val phải có cả hai hướng quay (hoặc thêm bản lật gương của mọi ảnh với nhãn hoán đổi theo `flip_idx`), báo cáo riêng từng hướng thay vì một con số gộp, và có ảnh che khuất chân. Lưu ý: một lần train, một seed, 53 ảnh, nên chênh lệch nhỏ (ví dụ 0.457 so với 0.417 trên val gốc) chưa chắc là khác biệt thật; khoảng cách 0.439 so với 0.298 trên val lật gương là đủ lớn để kết luận.

## Bài tập về nhà 3 — ONNX, latency trên CPU

Export `yolo26n.pt` sang ONNX hai cách (`end2end=False` cho head one-to-many cần NMS, `end2end=True` cho head one-to-one NMS-free), chạy bằng ONNX Runtime trên CPU của Colab (Intel Xeon 2.0 GHz), trung bình 30 lần trên `bus.jpg`.

| Cấu hình | preprocess (ms) | inference (ms) | postprocess (ms) | số box |
|---|---|---|---|---|
| one-to-many + NMS, conf 0.25 | 4.68 | 73.00 | 1.34 | 5 |
| one-to-one NMS-free, conf 0.25 | 5.41 | 83.13 | 0.48 | 5 |
| one-to-many + NMS, conf 0.001 | 5.98 | 88.43 | 2.54 | 186 |
| one-to-one NMS-free, conf 0.001 | 4.78 | 74.00 | 0.47 | 177 |

Nhận xét: postprocess của NMS-free ổn định ở khoảng 0.5 ms, còn head one-to-many + NMS tăng từ 1.34 ms lên 2.54 ms khi hạ conf từ 0.25 xuống 0.001 (186 box vào NMS). Khoảng chênh ở cột inference (73 so với 83 ms ở conf 0.25, nhưng 88 so với 74 ms ở conf 0.001) đổi chiều giữa hai mức conf, trong khi mạng giống nhau về kích thước, nên đó là nhiễu của CPU dùng chung trên Colab, không phải khác biệt thật. Trên CPU, cả hai cấu hình bị chi phối bởi inference (hơn 70 ms), nên phần NMS tiết kiệm được chỉ khoảng 1–2 ms, nhỏ hơn nhiều so với tổng; lợi ích của NMS-free sẽ rõ hơn với cảnh đông hơn hoặc model nhanh hơn. Số liệu GPU (T4) ở 1C cho kết luận tương tự: postprocess giảm khoảng 2.6–3 lần.
