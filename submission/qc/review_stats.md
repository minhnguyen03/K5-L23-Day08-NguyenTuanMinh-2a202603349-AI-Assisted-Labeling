# Số đo review (det)

- Ảnh tính: 24 (batch,gold); box máy: 434; đối tượng sau duyệt: 434; ngưỡng accepted: IoU >= 0.9
- Đã **loại 22 đối tượng nằm quanh các bẫy** khỏi bảng này: lỗi nhân tạo không phản ánh chất lượng máy.

## 1. Bốn trạng thái (AI error profile)
| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted (giữ nguyên) | 434 | 434/434 = 100.0% | 434 box máy |
| edited (sửa) | 0 | 0/434 = 0.0% | 434 box máy |
| deleted (xoá) | 0 | 0/434 = 0.0% | 434 box máy |
| added (vẽ thêm) | 0 | 0/434 = 0.0% | 434 đối tượng sau duyệt |

Trong số deleted + added, **0 trường hợp là đổi lớp** (máy gán sai lớp, box đúng chỗ): đó không phải máy bỏ sót. Số vật máy **sót thật** ≈ added − đổi lớp = 0.

### Theo lớp
| Lớp | accepted | edited | deleted | added |
|---|---|---|---|---|
| person | 252 | 0 | 0 | 0 |
| bicycle | 9 | 0 | 0 | 0 |
| motorbike | 88 | 0 | 0 | 0 |
| car | 49 | 0 | 0 | 0 |
| bus | 3 | 0 | 0 | 0 |
| truck | 7 | 0 | 0 | 0 |
| three_wheeler | 26 | 0 | 0 | 0 |

## 2. So với gold (nhãn gán tay từ đầu)
Ảnh gold: 4; đối tượng gold: 48 (mẫu số của recall)
| | TP | FP | FN | Precision | Recall |
|---|---|---|---|---|---|
| Máy (pre-label) | 39 | 27 | 9 | 59.1% (KTC 95%: 47–70%) | 81.2% (KTC 95%: 68–90%) |

*Lạc quan: ngưỡng được chọn trên chính các ảnh gold này (in-sample), và chỉ có 4 ảnh. Đọc khoảng tin cậy, đừng đọc con số đơn.*


(Ảnh gold không nằm trong nhãn sau duyệt: không tính residual / new error. Đó là chủ ý: duyệt lại ảnh bạn vừa tự gán tay sẽ đo trí nhớ, không đo thiên kiến.)

## 3. Ảnh bẫy
- Bắt được 0/12 = 0.0% (KTC 95%: 0–24%) (ngưỡng đạt: 80%). Chỉ 12 bẫy nên khoảng tin cậy rất rộng: đọc cùng danh sách bẫy lọt, đừng kết luận chỉ từ %.
  - missing: 0/3
  - extra: 0/3
  - loose: 0/3
  - wrong_class: 0/3
  - lọt: img_026.jpg (missing), img_033.jpg (missing), img_034.jpg (extra), img_004.jpg (loose), img_021.jpg (wrong_class), img_025.jpg (extra), img_003.jpg (loose), img_023.jpg (wrong_class), img_008.jpg (missing), img_009.jpg (loose), img_031.jpg (extra), img_018.jpg (wrong_class)

## Gợi ý đọc số
- `deleted` cao → ngưỡng thấp hoặc ánh xạ nhãn sai; `added` cao → máy bỏ sót (xem lớp nào);
- `accepted` thấp (< 50%) → sửa mỗi box lâu hơn vẽ mới: đổi model/ngưỡng hoặc vẽ tay;
- bắt bẫy < 80% → duyệt lại lô, siết checklist (quét ảnh TRƯỚC khi nhìn nhãn máy).
