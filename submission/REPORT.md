# Báo cáo Day 08 — 2A202603349 - Nguyễn Tuấn Minh

> Một file duy nhất cho cả bài (đã gộp recipe, QC và ghi chú cá nhân). Mọi con số kèm **mẫu số** và nguồn (file nào). Viết bằng lời của bạn.

## 1. Model và recipe (Part 2–3)

| Lớp | Model + cỡ | Ngưỡng | Vì sao (dẫn `run_id` trong `model_log.csv`) |
|---|---|---|---|
| person | YOLO26m | 0.20 | `r06`: quét trên 15 vật gold cho recall 93.3% (14/15) và precision 32.6% (14 TP, 29 FP); đây là ngưỡng thấp nhất vẫn đạt recall mục tiêu 90%, dẫn `submission/qc/threshold_sweep.csv` |
| bicycle | YOLO26m | 0.20 | `r06`: trên 6 vật gold, recall 100% (6/6) và precision 85.7% (6 TP, 1 FP), dẫn `threshold_sweep.csv` |
| motorbike | YOLO26m | 0.35 | `r06`: 10/11 vật gold được giữ (recall 90.9%) và precision 83.3% (10 TP, 2 FP) |
| car | YOLO26m | mặc định 0.30 | `r06`: không ngưỡng nào đạt recall 90% trên 8 vật gold; mức tốt nhất là recall 87.5%, nên giữ mặc định 0.30 thay vì ép số theo mẫu nhỏ |
| bus | YOLO26m | mặc định 0.30 | chỉ có 1 vật gold nên không lập ngưỡng riêng; mẫu trong `threshold_sweep.csv` không đủ |
| truck | YOLO26m | mặc định 0.30 | không có vật truck trong 48 đối tượng gold (n_gold = 0), không thể đánh giá ngưỡng |
| three_wheeler | YOLOE-26s; prompt `tuk-tuk`, `cyclo`, `cargo tricycle` | 0.15 | `r06`: trên 7 vật gold, recall 100% (7/7) và precision 77.8% (7 TP, 2 FP); dùng YOLOE vì YOLO26 không có lớp ba bánh chuyên biệt |

- Lệnh `merge_prelabels.py` đã dùng cho recipe lớp: `python scripts/merge_prelabels.py work/prelabel/yoloe_all/pre.coco.json work/prelabel/y26_all/pre.coco.json --iou 0.6 --conf 0.3 --thr "person=0.2,bicycle=0.2,motorbike=0.35,three_wheeler=0.15" --split "batch,gold" --out work/prelabel/merged_review/pre.coco.json`
- **Phiên bản SAM chọn:** SAM 2.1 (`sam2.1_b.pt`) để gộp mask trong Part SAM; log `r04` và `r06` chạy SAM 2.1 trên `img_004.jpg` và `img_013.jpg`, chạy trên CPU, tạo 16 mask cho 2 ảnh. Trong thư mục `submission/masks/` còn có cả SAM 1 và SAM 2.1 để so sánh trực quan ở `img_013`, nhưng không đủ log điểm/đúng sai để khẳng định SAM 1 hoặc 2.1 nhanh hơn hoặc tiết kiệm RAM hơn.
- Lỗi không sửa bằng đổi ngưỡng: tuk-tuk bị YOLO26 gán nhầm thành `car`/`truck` là lỗi ánh xạ lớp, không phải ngưỡng điểm; ví dụ so sánh trong `walkthrough_images/04_model_tour_yolo26_vs_yoloe_img002.jpg` và `work/prelabel/t_sam`. Cách sửa: YOLOE chạy riêng lớp `three_wheeler`, đặt file YOLOE trước YOLO26 trong merge và loại box bị trùng (IoU 0.6) để lớp ba bánh thắng box car/truck.
- Trong mẫu gold có thể quan sát nhầm lớp trên tuk-tuk, nhưng không có bảng ánh xạ confidence–nhãn của từng box trong file báo cáo để xác nhận cặp “điểm cao mà sai / điểm thấp mà đúng” theo ảnh cụ thể. Vì vậy không bịa ví dụ; cần đối chiếu trực tiếp box–confidence nếu đo thêm.
- Giới hạn: ngưỡng được chọn và chấm trên chính 4 ảnh gold, chỉ có 48 đối tượng gold; car/bus/truck và từng ngưỡng có mẫu rất nhỏ. Số đo trên gold do đó lạc quan và chỉ nên đọc cùng khoảng tin cậy.

## 2. AI error profile (từ `qc/review_stats.md`)

| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted | 434 | 434/434 = 100.0% | 434 box máy |
| edited | 0 | 0/434 = 0.0% | 434 box máy |
| deleted | 0 | 0/434 = 0.0% | 434 box máy |
| added | 0 | 0/434 = 0.0% | 434 đối tượng sau duyệt |

- Lớp máy chưa cho thấy mức lỗi cao trong batch review: person có 252 box được giữ, motorbike 88, car 49, three_wheeler 26, bicycle 9, truck 7 và bus 3. Tuy nhiên đánh giá theo sweep trên gold cho thấy `person` có nhiều FP nhất tỉ lệ (precision 32.6% ở ngưỡng 0.20), kèm 29 FP trên 15 vật gold; đây là lớp chưa ổn nhất theo dữ liệu đã có. `truck` không có vật gold nên không thể xếp hạng.
- So với gold: precision / recall của **máy** là **59.1% / 81.2%** (39 TP, 27 FP, 9 FN; mẫu 48 đối tượng gold). Kết quả này lấy trên chính 4 ảnh dùng để chọn ngưỡng; khoảng tin cậy 95% từ `review_stats.md` là khoảng 47–70% cho precision và 68–90% cho recall.

## 3. Thiên kiến tự động hóa và ảnh bẫy

- Bắt bẫy: **0/12 = 0.0%**; cận trên 95% cho phép từ `review_stats.md` là 24%. Ba bẫy thiếu vật (`missing`) đều lọt tại `img_026`, `img_033`, `img_008`; ba box thừa (`extra`) đều lọt tại `img_034`, `img_025`, `img_031`; ba box lỏng (`loose`) lọt tại `img_004`, `img_003`, `img_009`; ba sai lớp (`wrong_class`) lọt tại `img_021`, `img_023`, `img_018`.
- Nguyên nhân khả dĩ của kết quả: batch đã được duyệt thụ động với kết quả 434/434 box đều accepted, trong khi checklist yêu cầu quét ảnh trước để phát hiện missing/extra/loose/wrong_class. Có vẻ bước “quét ảnh trước khi xem nhãn máy” và ngừng xem khi nhìn thấy box đang đúng chưa được thực hiện đầy đủ.
- Accepted 100% cùng bắt bẫy 0% cho thấy profile không phản ánh chất lượng máy: accepted cao không chứng minh máy tốt vì 22 đối tượng nằm quanh bẫy đã được loại khỏi profile. Ngược lại, sai ở 12 bẫy là dấu hiệu mạnh của thiên kiến xác nhận—chỉ kiểm tra box đã có, không chủ động tìm vật bị bỏ sót hay box sai loại.

## 4. Máy có đáng dùng không?

- Thời gian (từ `time_log.csv`): file hiện tại chỉ có header và chưa ghi `gold_manual`/`batch_review`, nên không thể tính phút/ảnh gán tay hay phút/ảnh duyệt. Không suy ra tốc độ từ việc máy đã tạo box.
- Thiên lệch nhỏ: vì tập gold được gán trước khi chọn recipe, nhưng kết luận recall/precision vẫn chỉ là đánh giá in-sample; người gán có thể đã quen vùng nhìn và thêm bias khi xem lại.
- Kết luận: **chưa chứng minh máy đáng dùng cho phát hành**. Máy có thể hỗ trợ tạo nhắc nhãn cho person, bicycle, motorbike, car và three_wheeler, nhưng chưa đạt chứng cứ đủ; precision 59.1% và recall 81.2% trên gold, bắt bẫy 0/12, plus time log/QC chưa hoàn thiện. Riêng truck không có bằng chứng gold; bus chỉ 1 vật gold.
- **Ba quy tắc giảm thiên kiến được kiểm tra bằng số hoặc việc làm:**
  1. Quét ảnh trước, ghi missing/extra trong checklist rồi mới xem box máy; kiểm tra bằng `traps_key.json` có 12 bẫy và mục tiêu bắt ≥10/12.
  2. Chỉ giữ box khi `IoU >= 0.9` đối với box máy; mọi box chỉnh/sửa/xoá phải đếm vào review_stats thay vì auto accepted.
  3. Kiểm QC độc lập trên 5 ảnh được chọn seed 42 (24 ảnh batch, trong đó có các box `can_review`), báo cả lỗi theo đối tượng và theo ảnh; hiện `qc_review.csv` chưa điền `error_type` nên quy tắc chưa vượt qua.

## 5. Mask SAM cho 3 vật (Part 5)

| Ảnh | SAM (phiên bản) | Số click | Mask gồm người lái / hàng hoá? | Lỗi quan sát | File trong `masks/` |
|---|---|---|---|---|---|
| img_004 | SAM 2.1 | chưa ghi số click theo từng vật trong metadata | Mask tuk-tuk bao phần nhìn thấy của xe; người lái/người ngồi và hàng hoá không nằm riêng trong mask xe, nhưng kết quả trực quan cho thấy mask che phần thân xe và có thể tràn vào vùng nội thất | cần kiểm lại lỗ thủng, cạnh mái và bóng đổ; không có tọa độ điểm/loại điểm để đánh giá tái lập | `submission/masks/tuktuk_img004_sam2.jpg` |
| img_013 | SAM 1 | chưa ghi số click trong file; ảnh so sánh có mặt trong batch | Dùng để so sánh bao xe/cạnh biên; người lái/hành khách không được tách khỏi vật xe | chưa có log so sánh metric, chỉ có quan sát trực quan | `submission/masks/tuktuk_img013_sam1.jpg` |
| img_013 | SAM 2.1 | screenshot kèm nhiều điểm xanh/đỏ cho thấy tích cực và âm; không ghi tổng số chính xác | Mask bao tuk-tuk, che đủ phần khung xe nhìn thấy; hai người trong xe không phải vật riêng trong mask xe | SAM 2.1 có thể tràn sang người/đối tượng ngoài tuk-tuk nếu điểm âm đặt thiếu; cần điểm theo từng vật | `submission/masks/tuktuk_img013_sam2.jpg` |
| img_017 | SAM 2.1 | chưa ghi số click trong metadata | Cần đối chiếu thêm mask với người lái/hàng trên tuk-tuk; file có dạng overlay kết quả | chưa ghi một lỗi định lượng; quan sát trực quan là chưa đủ để kết luận mask ổn | `submission/masks/tuktuk_img017_sam2.jpg` |

- Đối chiếu guideline mục 4.2: quy tắc “bám phần nhìn thấy, không tính bóng đổ, mỗi vật không chồng nhau” hữu ích để sửa mask. Quy tắc lỗ thủng cần làm rõ hơn trên tuk-tuk vì mask có thể chui qua khe khung, cửa hở và khu vực giữa người; quy tắc về kính cửa, hàng nhô ra và xem người lái có thuộc xe hay mask riêng vẫn còn mơ hồ.
- Interactor SAM trong CVAT: không có bằng chứng ghi nhận đã thử và không có lỗi/tiến độ cụ thể trong các file đính kèm; vì vậy không kết luận được Interactor hoạt động hay không. Offline dùng SAM 2.1 bằng `sam2_points.py` đã tạo mask ở `submission/masks/`, còn Interactor cần thao tác trực tiếp trong CVAT và tạo mask tương tác tại task.

## 6. Calibration (Part 6)

- Số quyết định: **0 dòng dữ liệu** trong `submission/qc/calibration_decisions.csv` (file hiện tại không có quyết định nào).
- Quan trọng nhất: chưa có `CAL-xx` nào để dẫn; chưa thể xác định quyết định quan trọng nhất.
- Quy tắc mới đã vào `submission/guidelines/guideline_v1.1_calibrated.md`: **chưa có**; file hiện vẫn là khuôn chờ điền, không có ca/ảnh/quy tắc v1.1.
- Nguyên nhân gốc theo bằng chứng hiện có: thiếu calibration thay vì xác định được guideline mơ hồ hay lỗi của người/máy. Có một câu hỏi rõ ràng trong guideline v1.0—xe ba bánh, người trong xe, hàng nhô ra, người lái—đã liên quan tới lỗi tuk-tuk; tuy nhiên các lỗi máy trong review/bẫy cần được chốt thành quyết định trước khi gọi là vấn đề guideline.

## 7. QC và quyết định (Part 6)

- Mẫu: 5 ảnh trong `qc_sample.csv` (`img_003`, `img_006`, `img_014`, `img_031`, `img_035`) gồm tổng 96 đối tượng theo cột `so_doi_tuong`; file không ghi tách số ảnh `can_review` và số ngẫu nhiên. `qc_sample.csv` ghi `la_ca_kho=no` cho cả 5 dòng. Người kiểm: bạn (**hạn chế**: người duyệt batch_reviewed và người kiểm QC có thể cùng thiên kiến).
- Lỗi theo **đối tượng**: chưa tính được; `qc_review.csv` có 5 dòng nhưng cột `error_type` đều trống.
- Lỗi theo **ảnh**: chưa tính được; trong `qc_review.csv` chưa có loại lỗi nào được điền để xác nhận ảnh lỗi hay sạch.
- Ngưỡng Release ≤ 2% theo đối tượng và theo ảnh, nhưng mẫu này chưa có kết quả QC hợp lệ nên không thể chứng minh đạt ngưỡng.
- Top lỗi: chưa xác định được vì `error_type` trống; 12 bẫy cho thấy đủ loại missing/extra/loose/wrong_class để dùng riêng làm kiểm tra review, không thay cho QC.
- **Giới hạn mẫu nhỏ**: `qc_report_auto.md` không có trong thư mục `submission/qc/`, vì vậy không có cận trên 95% do script sinh. Với 5 ảnh và chưa biết lỗi, không được suy ra “không lỗi” hay “đạt ≤2%”; 12 bẫy chỉ cho cận trên 95% 24% ở kết quả 0/12.
- Script kiểm được: tính được số ảnh và số đối tượng mẫu, kiểm được cấu trúc và các file còn trống. Script không thể kiểm được ôm sát của box, không bắt được hai người cùng bỏ sót một vật, không thay checklist của con người.

**Quyết định: Hold**, ngày 2026-11-25. Lý do theo cây Hold → Rework → Release: **Hold** vì bắt bẫy 0/12 dưới mục tiêu 80%, review_stats cho thấy 434/434 box accepted và không có chỉnh/xoá/thêm, đồng thời model_log/time_log/calibration/QC chưa đầy đủ. Cần Rework mới đủ điều kiện xét Release.

Việc cần làm tiếp:
1. Điền `time_log.csv` bằng thời gian thật cho `gold_manual` và `batch_review`.
2. Điền `qc_review.csv` với `error_type` cho 5 ảnh; chạy lại QC report để có tỉ lệ/lưới cận trên 95%.
3. Lập ít nhất 3 dòng calibration, gồm cả `loi_may` và `cau_hoi_mo`, rồi cập nhật guideline v1.1; xem lại các 12 bẫy bằng checklist từ ảnh trước.
4. Sau Rework, duyệt lại batch; chỉ Release nếu bắt bẫy ≥10/12, lỗi ảnh/đối tượng theo QC ≤2%, và file/report/log nhất quán.

## 8. Ghi chú cá nhân

- **Dùng AI để làm bài:** có dùng trợ lý Copilot AI trong VS Code để tổng hợp số liệu và viết báo cáo; mọi số liệu được kiểm tra lại trong `model_log.csv`, `threshold_sweep.csv`, `review_stats.md`, `qc_review.csv`, `time_log.csv` và các file release. Không lấy số từ `traps_key.json` để thay đổi kết quả bắt bẫy.
- Gold có bị nhiễm không: không thể khẳng định hoàn toàn từ file hiện có. Gold đã có 48 đối tượng trước khi tạo profile; `gold_manual.json` không ghi ngày/giai đoạn hoặc bằng chứng không xem pre-label. Nếu dùng lại trước đây, cần ghi rõ trong quá trình nộp.
- Máy / GPU / CVAT dùng: các dòng chạy trong `model_log.csv` ghi CPU, không ghi GPU/CUDA/MPS cụ thể; chưa có thông tin CVAT Docker/version trong file submission. Không mô tả thêm để tránh giả định.
- Dùng cache `data/cache/` thay vì tự chạy: chưa xác nhận từ log. Các run dùng CPU và n_boxes/thời gian; file chạy cụ thể không ghi cache hay tải model, nên phần này để trống đến khi có log môi trường.
- Lỗi / bug script đã gặp: chưa ghi trong các file có liên quan. Trong quá trình kiểm tra nhận thấy `time_log.csv`, `calibration_decisions.csv`, `qc_review.csv.error_type` và `qc_report_auto.md` còn thiếu nội dung; đây là vướng mắc tiến độ, không phải bug script đã xác nhận.
- Một điều làm khác lần sau: gắn checklist ca khó và thời gian ngay trong lúc làm từng Part thay vì chờ lúc cuối; đồng thời ghi lại từng click SAM và dùng cùng một quy tắc QC cho review và bẫy để dữ liệu tái lập được.
