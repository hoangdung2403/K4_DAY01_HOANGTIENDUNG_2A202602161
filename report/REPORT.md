# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy: 9/11/2026**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> ZIP do notebook tạo có tên `KX-DAY01-report.zip`. Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`)
    (468, "cab", 1, 0.510915, "ImageNet-1K")
- Record này mô tả toàn ảnh như thế nào? 
    Record này mô tả toàn ảnh bằng các nhãn vật thể có trong ảnh mà model cho là phù hợp nhất.
- Ai định nghĩa class list mà checkpoint có thể dự đoán?
    ImageNet-1K là 1 benchmark tiêu chuẩn gồm có 1000 phân loại hình ảnh.
- Vì sao cần giữ cả ID, tên lớp và tên taxonomy? 
    ID thì dùng để cho máy hiểu và đối chiếu đúng sai;
    Tên lớp giúp con người có thể đọc và dễ theo dõi hơn;
    tên taxonomy thì dùng để xác định class chuẩn theo hệ thống của benchmark ImageNet-1K.
- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?
    Cần quy định rõ ngay từ đầu cách chọn ground truth, cần xác định rõ mục tiêu của ảnh cần hướng đến chứ không được dựa trên điểm score của model làm điểm chuẩn.
- Vì sao model score không phải ground truth?
    vì score là điểm mà model đánh giá và đưa ra còn ground truth nó là tiêu chuẩn vàng là điểm chuẩn được xác định bởi con người, trong bài này cab có điểm cao nhất là 0.510915 nhưng không có nghĩa nó đúng nó vẫn có thể là minibus.
## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):
    ("person", 0.912625, [385.33, 69.24, 498.92, 348.92], 113.58px, 279.68px)
- Diễn giải vị trí box bằng lời:
    Box person nằm ở bên phải, bao quanh và ôm sát gần như toàn bộ cơ thể người.
- So sánh số prediction ở hai threshold:
    threshold càng thấp thì độ bao phủ càng cao, phtas hiện được nhièu object hơn nhưng cũng tạo thêm nhiều prediction, threshold cao làm giảm số prediction nhưng có nguy cơ bỏ sót object
- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?
    threshold càng nhỏ thì sẽ phát hiện nhiều bõ hơn và reviewer phải xem nhiều hơn và ngược lại
- Đề xuất một quy tắc box chặt:
    bõ phải bao phủ ít nhất là toàn bộ phần nhìn thấy được, sát mép object nhất có thể, càng ít background càng tốt
- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?
    box bao quanh phần có thể quan sát được, không nên tự suy đaons phần che, nếu bị cắt mép box được phép tiếp tục mở rộng ra ngoài ảnh, nếu vào trường hợp không rõ, không thể tự đoán được thì có thể đưa sang cho escalation.
## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):
    ("itchen-001","person",0.899318,348,[])
- Polygon bổ sung chi tiết gì so với box?
    bõ chỉ cho biết vùng hình chữ nhật còn Polygon cho biết đuongwf biên của vật thể
- `instance_id` dùng để làm gì và không phải loại ID nào?
    phân biệt từng object cụ thể trong cùng một ảnh, kể cả có cùng 1 class_name
- Đề xuất một quy tắc biên mask:
    Mask phải bám sát phần biên có thể quan sát của object, không bao gồm background; các vùng bị che khuất không được tự ý suy đoán nếu guideline không quy định rõ.
- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?
    Nếu ranh giới object không đủ rõ để xác định mask một cách nhất quán, ta không tự suy đoán mà đánh dấu trường hợp không chắc chắn và chuyển reviewer/escalation quyết định.
## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | 1class/1 ảnh | ảnh co nhiều chủ thể khác nhau | gán label theo guideline, không dựa vào score model | Kiểm tra class có đúng quy tắc và taxonomy không |
| Phát hiện vật thể | class + bbox cho từng object | Box quá rộng/hẹp,che khuất | Vẽ box sát object và gán class | Kiểm tra class, số object, vị trí và độ chặt của box |
| Instance segmentation | class + polygon/mask cho từng instance | biên object mờ, object tiếp xúc/che khuất, mask ăn sang background khác |  Vẽ polygon theo phần quan sát được| Kiểm tra instance, class và độ chính xác của biên mask |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu:
    Chỉ xử lý dữ liệu trong phạm vi của bài toán, không tự ý thu thập, thêm dữ liệu nhất là dữ liệu vượt ngoài phạm trù của bài toán đã đặt ra, không được chia sẽ dữ liệu ra bên ngoài.
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho:
    người phụ trách hoặc quản trị viên trực tiếp phụ trách hệ thống, bài toán này.
## 6. Danh sách bằng chứng

- [X] `classification_predictions.json`
- [X] `detection_predictions.json`
- [X] `segmentation_predictions.json`
- [X] `IMAGE_ATTRIBUTION.md`
- [X] `visuals/classification_top5.png`
- [X] `visuals/detection_predictions.png`
- [X] `visuals/segmentation_prediction.png`
- [X] Ô validation cuối notebook báo `PASS`.
- [X] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
