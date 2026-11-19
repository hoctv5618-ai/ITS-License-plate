# TÀI LIỆU GIẢI THÍCH THUẬT TOÁN (DÙNG CHO BÁO CÁO TIỂU LUẬN CHƯƠNG 2 & CHƯƠNG 4)

Tài liệu này giải thích bằng ngôn ngữ đời thường kèm ví dụ số nhỏ cho từng thuật toán sử dụng trong dự án, giúp sinh viên dễ dàng đưa vào báo cáo hoặc trả lời câu hỏi phản biện trước lớp.

---

## GIAI ĐOẠN 0: CHUẨN BỊ DỮ LIỆU, LỌC TRÙNG & CHIA TẬP HUẤN LUYỆN

### 1. Đọc/Ghi ảnh an toàn với đường dẫn tiếng Việt (`np.fromfile` + `cv2.imdecode`)
- **Ý tưởng:** Hàm `cv2.imread()` mặc định của OpenCV được viết bằng C++ cũ nên trên Windows chỉ đọc được đường dẫn bảng mã ASCII (không dấu). Khi thư mục có dấu tiếng Việt như `D:\hệ thống giao thông`, `cv2.imread()` sẽ trả về `None` dù file ảnh hoàn toàn bình thường. Thay vì để OpenCV tự mở file, ta dùng thư viện `numpy` (`np.fromfile`) đọc toàn bộ file thành mảng các byte thô (`uint8`), rồi đưa mảng byte đó cho `cv2.imdecode()` giải mã thành ma trận điểm ảnh BGR.
- **Ví dụ nhỏ:**
  - File ảnh có kích thước `15.000` byte trên ổ đĩa.
  - `np.fromfile("D:/hệ thống giao thông/anh1.jpg", dtype=np.uint8)` đọc ra mảng 1 chiều gồm `15.000` con số từ `0` đến `255` (ví dụ: `[255, 216, 255, 224, ...]`).
  - `cv2.imdecode()` bung mảng nén JPEG này thành ma trận ảnh 3 chiều kích thước `(480, 640, 3)`.
- **Vì sao chọn:** Cách này giải quyết triệt để lỗi đường dẫn tiếng Việt trên Windows mà không cần cài thêm thư viện ngoài hay buộc người dùng phải đổi tên thư mục gốc.

---

### 2. Mã băm MD5 (Message-Digest Algorithm 5) — Tìm ảnh trùng tuyệt đối
- **Ý tưởng:** MD5 giống như "dấu vân tay kỹ thuật số" của một file. Thuật toán đọc toàn bộ từng byte của file ảnh và tính ra một chuỗi 32 ký tự hệ thập lục phân (128 bit). Chỉ cần 2 file giống hệt nhau từng byte (dù đặt tên khác nhau ở 2 thư mục khác nhau), mã MD5 của chúng sẽ giống nhau 100%. Ngược lại, chỉ cần thay đổi 1 chấm màu duy nhất trong ảnh, mã MD5 sẽ biến đổi hoàn toàn.
- **Ví dụ số nhỏ:**
  - Ảnh `kaggle_00012.jpg` có nội dung các byte là `[10, 20, 30]` -> qua hàm băm MD5 tạo ra chuỗi: `a1b2c3d4e5f6...`
  - Ảnh `roboflow_00450.jpg` là bản copy của ảnh trên, cũng có các byte `[10, 20, 30]` -> MD5 cũng ra đúng `a1b2c3d4e5f6...` -> Kết luận: **Trùng chính xác (`exact_md5`)**.
  - Nhưng nếu `roboflow_00450.jpg` bị nén lại làm byte cuối đổi thành `[10, 20, 31]`, MD5 sẽ nhảy sang một chuỗi khác hẳn: `9f8e7d6c5b4a...` (lúc này MD5 không nhận ra nữa, cần đến `pHash` ở mục dưới).
- **Vì sao chọn:** MD5 có sẵn trong thư viện chuẩn `hashlib` của Python, tốc độ tính toán cực nhanh và chính xác tuyệt đối khi tìm các file bị copy lặp lại giữa hai bộ dữ liệu.

---

### 3. Mã băm cảm nhận (`Perceptual Hash - pHash`) & Đếm bit bằng phép `XOR` — Tìm ảnh gần trùng
- **Ý tưởng:** Trong thực tế, cùng một bức ảnh chụp xe máy nhưng khi tải lên Roboflow hoặc Kaggle có thể bị đổi kích thước (resize), nén giảm chất lượng JPEG hoặc chỉnh sáng nhẹ. Khi đó các byte trong file đã khác đi nên MD5 bó tay.
  Thuật toán **pHash** bắt chước mắt người:
  1. Đổi ảnh về ảnh xám và thu nhỏ xuống kích thước rất bé (32x32).
  2. Biến đổi tần số DCT (giống chuẩn nén JPEG) và giữ lại vùng tần số thấp `8x8` (64 ô) đại diện cho bố cục sáng/tối tổng thể của bức ảnh (bỏ qua nhiễu hạt chi tiết).
  3. Tính giá trị trung bình của 64 ô đó: ô nào lớn hơn trung bình thì ghi bit `1`, nhỏ hơn hoặc bằng thì ghi bit `0`. Kết quả thu được một số nguyên **64 bit** (gồm 64 chữ số `0` hoặc `1`).
  Để đo độ giống nhau giữa hai ảnh, ta đếm xem trong 64 vị trí có bao nhiêu bit khác nhau (gọi là **khoảng cách Hamming**). Trong code, ta chuyển `pHash` thành số nguyên rồi dùng phép toán logic **XOR (`^`)**: hai bit giống nhau (`0^0` hoặc `1^1`) cho kết quả `0`, hai bit khác nhau (`1^0` hoặc `0^1`) cho kết quả `1`. Sau đó dùng `.bit_count()` để đếm tổng số bit `1`.
- **Ví dụ số nhỏ (minh họa với mã băm rút gọn 8 bit):**
  - Giả sử vùng lưới của **Ảnh A** có độ sáng: `[120, 200, 180, 50, 40, 210, 90, 110]`, trung bình là `125`.
    - Các vị trí `> 125` thành `1`, còn lại thành `0` -> Mã băm Ảnh A: `01100100` (số nguyên = `100`).
  - **Ảnh B** là Ảnh A bị nén mờ nhẹ, độ sáng thành: `[118, 195, 175, 55, 130, 205, 85, 105]`. Chỉ có vị trí thứ 5 nhảy nhẹ qua mức trung bình -> Mã băm Ảnh B: `01101100` (số nguyên = `108`).
  - Thực hiện phép `XOR`:
    ```text
    Ảnh A : 0 1 1 0 0 1 0 0  (100)
    Ảnh B : 0 1 1 0 1 1 0 0  (108)
    -----------------------   phép XOR (^)
    Kết quả: 0 0 0 0 1 0 0 0  -> Có đúng 1 bit bằng 1 (Khoảng cách Hamming = 1)
    ```
  - Vì khoảng cách `1 <= 6` (ngưỡng `PHASH_THRESHOLD = 6`), thuật toán kết luận Ảnh A và Ảnh B là **cặp ảnh gần trùng (`near_phash`)**.
- **Vì sao chọn:** Chuyển `pHash` sang số nguyên 64-bit và dùng phép `XOR` giúp CPU so sánh hàng chục triệu cặp ảnh trong vài giây thay vì phải gọi hàm đối chiếu chuỗi ký tự chậm chạp.

---

### 4. Thuật toán Hợp - Tìm (`Union-Find`) — Gom nhóm ảnh trùng theo tính chất bắc cầu
- **Ý tưởng:** Khi quét các cặp trùng, ta có thể gặp tình huống bắc cầu:
  - Ảnh `A` gần trùng với ảnh `B` (khoảng cách = 4).
  - Ảnh `B` gần trùng với ảnh `C` (khoảng cách = 4).
  - Dù `A` và `C` không trực tiếp nằm trong cùng một dòng báo cáo (hoặc khoảng cách `A-C` là 7), về mặt ngữ nghĩa cả `A, B, C` đều chụp cùng một chiếc xe nên **bắt buộc phải gom chung vào một nhóm**.
  Thuật toán **Union-Find** dùng một từ điển `parent` (cha) cực kỳ đơn giản:
  - Ban đầu, mỗi ảnh tự làm "nhóm trưởng" (cha) của chính nó: `parent[x] = x`.
  - Hàm `find(x)`: đi ngược theo `parent` cho tới khi gặp "nhóm trưởng" gốc (phần tử có `parent[gốc] == gốc`).
  - Hàm `union(a, b)`: tìm nhóm trưởng của `a` và nhóm trưởng của `b`, rồi cho nhóm trưởng bên này nhận nhóm trưởng bên kia làm cha.
- **Ví dụ nhỏ:**
  - Có 4 ảnh: `A, B, C, D`. Ban đầu: `parent = {A: A, B: B, C: C, D: D}`.
  - Phát hiện cặp trùng `(A, B)`: gọi `union(A, B)` -> cho `B` nhận `A` làm cha: `parent[B] = A`.
  - Phát hiện cặp trùng `(B, C)`: gọi `union(B, C)` -> `find(B)` ra gốc là `A`, `find(C)` ra gốc là `C` -> cho `C` nhận `A` làm cha: `parent[C] = A`.
  - Kết quả cuối cùng:
    - `find(A) = A`, `find(B) = A`, `find(C) = A` -> Cả 3 ảnh `{A, B, C}` tự động quy về chung nhóm gốc `A`.
    - `find(D) = D` -> Ảnh `D` đứng riêng một nhóm `{D}`.
- **Vì sao chọn:** Cài đặt chỉ mất khoảng 15 dòng lệnh Python thuần (một `dict` và hai hàm ngắn), không cần thư viện đồ thị phức tạp mà gom nhóm bắc cầu chính xác 100%.

---

### 5. Rò rỉ dữ liệu (`Data Leakage`) & Chia tập theo cụm 80/10/10
- **Ý tưởng:** Trong học máy, tập `train` (80%) giống như **bài tập về nhà**, tập `val` (10%) là **đề thi thử**, còn tập `test` (10%) là **đề thi cuối kỳ**.
  Nếu hai bức ảnh giống nhau (cùng chụp một chiếc xe ở cùng một góc) mà một ảnh rơi vào tập `train`, ảnh kia rơi vào tập `test`, thì mô hình YOLO không cần học cách nhận diện biển số tổng quát mà chỉ cần "học thuộc lòng" bức ảnh đó trong lúc ôn bài là làm trúng ngay câu hỏi trong đề thi cuối kỳ. Hiện tượng đó gọi là **Rò rỉ dữ liệu (Data Leakage)**, làm điểm đánh giá trên báo cáo cao giả tạo nhưng khi đem ra chạy thực tế lại sai.
- **Ví dụ số nhỏ:**
  - Giả sử ta có 10 ảnh, trong đó 3 ảnh `{A1, A2, A3}` chụp cùng một xe (đã được Union-Find gom vào Nhóm 1), 7 ảnh còn lại mỗi ảnh một nhóm riêng.
  - Nếu chia ngẫu nhiên từng ảnh rời rạc: rất dễ xảy ra `train` chứa `{A1, A2}` còn `test` chứa `{A3}` -> **Bị rò rỉ dữ liệu!**
  - Khi chia theo nhóm (Group Split): cả cụm `{A1, A2, A3}` được bốc đi cùng nhau. Nếu cụm này vào `train` thì cả 3 ảnh đều nằm ở `train` (`test` hoàn toàn không có ảnh nào của chiếc xe đó).
- **Vì sao chọn:** Kết hợp chia theo cụm (Group Split) riêng cho từng nguồn (`kaggle` và `roboflow` đều chia 80/10/10 theo số lượng ảnh) vừa đảm bảo mỗi tập đều có đại diện của cả 2 nguồn dữ liệu, vừa cam kết kiểm tra chéo tên file và mã MD5 đạt **PASS 100% (0 ảnh trùng giữa các tập)**.

---

### 6. Ba điểm lưu ý chuyên sâu khi phân tích dữ liệu (Đưa vào Chương 4 — Thực nghiệm & Thảo luận)

#### a) Vì sao khoảng cách Hamming của `pHash` luôn là số chẵn (`0, 2, 4, 6, 8...`)?
- **Ý tưởng:** Hàm `imagehash.phash` lấy **trung vị (median)** của 64 giá trị tần số DCT làm ranh giới: nửa lớn hơn trung vị gán bit `1`, nửa nhỏ hơn gán bit `0`. Vì `64` là số chẵn, bất kỳ mã `pHash` nào cũng luôn có **đúng 32 bit `1` và 32 bit `0`**.
  Khi so sánh hai chuỗi nhị phân có cùng số lượng bit `1`: cứ mỗi khi Ảnh A đổi một bit từ `1 -> 0` để giống Ảnh B ở vị trí này, thì bắt buộc ở một vị trí khác phải có một bit đổi ngược lại từ `0 -> 1` để tổng số bit `1` vẫn giữ nguyên. Do đó, các vị trí khác nhau luôn đi theo từng cặp (2 bit một lần).
- **Ví dụ số nhỏ (với dãy 4 bit có đúng 2 bit `1` và 2 bit `0`):**
  - Ảnh A: `1 1 0 0` (có hai số `1`).
  - Ảnh B: `1 0 1 0` (cũng có hai số `1`).
  - So sánh từng vị trí: vị trí số 2 khác nhau (`1` vs `0`) và vị trí số 3 khác nhau (`0` vs `1`) -> Khoảng cách = **2** (không bao giờ tạo ra được hai dãy 4-bit cùng có hai số `1` mà chỉ lệch nhau đúng 1 vị trí).

#### b) Vì sao chọn "gom thừa" (`pHash <= 6`) thay vì "gom thiếu"?
- **Ý tưởng:** Khi đặt ngưỡng `pHash = 6`, một số ảnh chụp các xe khác nhau nhưng đứng ở cùng một trạm barie cố định bị nối bắc cầu vào chung một nhóm lớn (ví dụ nhóm 179 ảnh). Đây gọi là **gom thừa**.
  Ngược lại, nếu hạ ngưỡng quá thấp (ví dụ `pHash <= 2`), những ảnh chụp cùng một chiếc xe nhưng hơi lệch góc hoặc chói đèn sẽ không được nhận ra và bị tách rời (**gom thiếu**).
- **Ví dụ số nhỏ:**
  - **Nếu gom thiếu:** Cùng chiếc xe `51A-853.25` có 2 ảnh, 1 ảnh lọt vào `train`, 1 ảnh lọt vào `test` -> Gây **rò rỉ dữ liệu (Data Leakage)**, làm kết quả đánh giá gian lận.
  - **Nếu gom thừa:** 5 chiếc xe khác nhau ở cùng một góc camera được gom chung thành 1 cụm và cùng đưa vào tập `train`. Không có ảnh nào bị xóa mất (tổng số ảnh vẫn giữ nguyên 8,752 ảnh), mô hình vẫn học bình thường và tập `test` hoàn toàn trong sạch.
- **Vì sao chọn:** Trong chuẩn bị dữ liệu học máy, **gom thừa an toàn hơn gom thiếu**, kết hợp cách xếp nhóm lớn vào trước (như xếp đá tảng vào thùng trước rồi mới rải các nhóm nhỏ 1–2 ảnh vào tập đang thiếu nhiều nhất) giúp tỉ lệ thực tế vẫn đạt đúng `80% / 10% / 10%` tuyệt đối.

#### c) Hạn chế của bộ dữ liệu cần ghi rõ trong báo cáo tiểu luận
- **Thống kê thực tế từ `README.roboflow.txt` và `tools/image_sizes.py`:**
  - File `README.roboflow.txt` xác nhận bộ dữ liệu xuất ra **không có bước tiền xử lý resize/stretch** (`0.00%` ảnh có kích thước `640x640`).
  - Trong nguồn **roboflow** (`8,254` ảnh), có tới **`89.6%` (`7,395` ảnh)** có cùng đúng một kích thước `472x303` pixel (chiều cao biển trung vị trên ảnh gốc chỉ đạt `35.0 px`, `34.96%` biển nhỏ hơn `30 px`).
  - Trong nguồn **kaggle** (`498` ảnh), có tới **`88.2%` (`439` ảnh)** có cùng kích thước Full HD `1920x1080` pixel (chiều cao biển trung vị trên ảnh gốc là `242.0 px`).
- **Hai nhận xét quan trọng cần đưa vào Chương 4 (Thực nghiệm & Thảo luận):**
  1. **Tính đa dạng góc máy hạn chế:** Việc gần `90%` ảnh của mỗi nguồn tập trung vào đúng một độ phân giải cố định cho thấy dữ liệu đến từ **rất ít thiết lập camera giám sát cố định** (chủ yếu là camera trạm kiểm soát barie / bãi đỗ xe). Vì vậy, điểm số (`mAP`, `Precision`, `Recall`) đo trên tập `test` chỉ phản ánh hiệu năng của mô hình trên cùng loại camera đó, **chưa chắc giữ nguyên độ chính xác khi gặp camera lạ** (khác góc lắp đặt, tiêu cự hoặc điều kiện ánh sáng).
  2. **Ý nghĩa thực sự của việc tăng `imgsz` (`640` vs `960`) khi huấn luyện YOLO:** Với ảnh gốc độ phân giải thấp (`472x303` của `roboflow`), việc phóng to ảnh lên `imgsz = 640` hay `960` khi đưa vào mạng YOLO **không hề sinh thêm chi tiết quang học mới** (ảnh không nét hơn ảnh gốc). Tác dụng của `imgsz = 960` là:
     - Với ảnh `kaggle` (`1920x1080`): giảm mức độ bị thu nhỏ (chỉ thu nhỏ `2` lần thay vì `3` lần ở `imgsz = 640`), giúp giữ lại chi tiết thật của các biển số ở xa.
     - Với ảnh `roboflow` (`472x303`): chỉ làm thay đổi kích thước biểu diễn của vật thể trên lưới đặc trưng (feature map) mà mạng CNN nhìn thấy (tỉ lệ biển `< 20px` trên khung mạng giảm từ `12.80%` ở `640` xuống `2.58%` ở `960`), đổi lại thời gian tính toán mỗi ảnh tăng lên.

---

## GIAI ĐOẠN 1: CẤU HÌNH & TIỀN XỬ LÝ ẢNH BIỂN SỐ (`config.py`, `preprocess.py`)

### 1. Chuyển ảnh xám (`Grayscale`)
- **Ý tưởng:** Ảnh màu BGR gồm 3 kênh màu (Xanh dương, Xanh lá, Đỏ), mỗi điểm ảnh tốn 3 con số. Tuy nhiên, khi đọc chữ trên biển số, ta chỉ quan tâm chỗ nào tối (nét chữ) và chỗ nào sáng (nền biển), không cần biết xe màu gì. Hàm `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)` gộp 3 kênh lại thành 1 kênh duy nhất theo công thức độ nhạy mắt người:
  `Gray = 0.299 * R + 0.587 * G + 0.114 * B` (mắt người nhạy nhất với màu xanh lá nên hệ số `G` lớn nhất).
- **Ví dụ số nhỏ:**
  - Một điểm ảnh trên nền biển trắng hơi ngả vàng có `B = 200, G = 240, R = 250`.
  - Độ sáng xám: `0.299 * 250 + 0.587 * 240 + 0.114 * 200 = 74.75 + 140.88 + 22.8 = 238.43` -> làm tròn thành **`238`** (sáng, gần mức trắng 255).
- **Vì sao chọn:** Giảm 3 lần khối lượng dữ liệu tính toán cho các bước sau và là đầu vào bắt buộc của thuật toán cân bằng sáng CLAHE cũng như ngưỡng hóa Otsu.

---

### 2. Phóng to giữ nguyên tỉ lệ khung hình (`Aspect-Ratio Preserving Resize`) & Nội suy bậc ba (`Bicubic Interpolation`)
- **Ý tưởng:**
  - **Vì sao phải giữ nguyên tỉ lệ (`w/h`) và chỉ phóng to khi `h < 100px`?** Biển số Việt Nam có 2 dáng hình học rất khác nhau (trên biểu đồ histogram tập `train`, biển 2 dòng có đỉnh `w/h ≈ 1.3`, còn biển 1 dòng có đỉnh `w/h ≈ 3.3`, điểm lõm phân cách ở `w/h = 2.1`). Nếu ép cứng mọi biển về một khung cố định (ví dụ `300x100`), biển vuông sẽ bị kéo giãn ngang làm chữ bè dẹt, còn biển dài bị ép hẹp làm các số dính sát nhau. Mặt khác, thống kê thực tế trên 200 ảnh `test` cho thấy chiều cao biển trung vị chỉ đạt **`47.0 px`** và có tới **`88.04%`** biển thấp hơn `100 px`. Vì vậy, chỉ những biển có `h < 100 px` mới được nhân tỉ lệ `scale = 100 / h` cho cả chiều rộng lẫn chiều cao.
  - **Nội suy bậc ba (`cv2.INTER_CUBIC`) hoạt động thế nào?** Khi phóng to ảnh, máy tính phải "điền thêm" các điểm ảnh mới nằm giữa các điểm ảnh cũ. Thay vì chỉ copy điểm gần nhất (làm chữ bị răng cưa vỡ hạt) hay lấy trung bình 4 điểm quanh nó (`Bilinear` - làm chữ hơi mờ), nội suy `Bicubic` xét **16 điểm ảnh lân cận (vùng 4x4)** và lắp một đường cong đa thức bậc 3 đi qua chúng. Nhờ vậy, rìa nét chữ có độ chuyển mượt và giữ được độ dốc sắc nét hơn.
- **Ví dụ số nhỏ:**
  - Một biển số xe máy cắt ra có kích thước `w = 65 px, h = 50 px` (tỉ lệ `w/h = 1.3`).
  - Vì `h = 50 < 100`, hệ số phóng to là `scale = 100 / 50 = 2.0`.
  - Kích thước mới là `new_w = 65 * 2.0 = 130 px`, `new_h = 100 px` -> Tỉ lệ mới vẫn là `130 / 100 = 1.3` (chữ to gấp đôi nhưng không hề bị méo).

---

### 3. Cân bằng biểu đồ xám cục bộ có giới hạn tương phản (`CLAHE`)
- **Ý tưởng:** Khi chụp ngoài đường, biển số rất hay bị **nửa sáng nửa tối** (một bên bị bóng cản xe che khuất, một bên bị nắng hoặc đèn pha rọi chói). Nếu tăng sáng toàn bộ ảnh như nhau, vùng tối vừa rõ lên thì vùng sáng đã bị cháy trắng xóa.
  Thuật toán **CLAHE** (`Contrast Limited Adaptive Histogram Equalization`) giải quyết bằng 2 bước:
  1. **Chia lưới cục bộ (`tileGridSize = 8x8`):** Chia bức ảnh biển số thành 64 ô chữ nhật nhỏ và kéo giãn độ tương phản riêng cho từng ô. Nhờ đó, ô bị tối được nâng sáng lên, còn ô bị chói được dịu xuống.
  2. **Giới hạn độ khuếch đại (`clipLimit = 2.0`):** Trên nền trắng đồng nhất của biển số, nếu kéo tương phản quá mức thì các hạt bụi/nhiễu li ti sẽ bị phóng đại thành vết loang đen. `clipLimit` cắt bớt phần đỉnh quá cao của biểu đồ độ sáng trước khi cân bằng, giữ cho nền biển vẫn phẳng mịn.
- **Ví dụ số nhỏ:**
  - Giả sử ở góc biển bị bóng râm, chữ đen có độ sáng `30` và nền biển xám tối có độ sáng `60` (chênh lệch chỉ `30` mức nên nhìn rất mờ).
  - CLAHE xét riêng ô góc đó và kéo dãn dải `[30..60]` ra thành `[15..210]` -> Chữ đen thành `15`, nền thành `210` (chênh lệch vọt lên `195` mức, nhìn rõ nét chữ ngay lập tức mà không làm cháy sáng các ô khác).
- **Vì sao chọn:** CLAHE xử lý cực tốt hiện tượng bóng đổ và lóa đèn cục bộ trên biển số xe mà vẫn giữ nguyên thông tin mức xám mềm mại cho mạng học sâu của EasyOCR.

---

### 4. Làm mờ Gaussian (`Gaussian Blur`) + Ngưỡng hóa tự động Otsu (`Otsu Thresholding`)
- **Ý tưởng:**
  - **Gaussian Blur (`5x5`):** Dùng một ma trận trọng số hình chuông (tâm lớn nhất, giảm dần ra xung quanh) trượt qua ảnh để làm mờ nhẹ các chấm nhiễu hạt, vết bùn nhỏ hoặc đinh ốc trước khi tách nhị phân.
  - **Thuật toán Otsu:** Thay vì người lập trình phải đoán một con số ngưỡng cố định (như `127`) để quy định `< 127` là chữ đen (`0`) và `>= 127` là nền trắng (`255`), Otsu tự động quét thử mọi ngưỡng từ `0` đến `255` trên biểu đồ độ sáng của bức ảnh đó, rồi chọn ngưỡng `T*` sao cho **độ tách biệt (phương sai giữa hai nhóm điểm tối và điểm sáng) đạt cực đại**.
- **Ví dụ số nhỏ:**
  - Một biển số chụp buổi chiều có các điểm ảnh thuộc nét chữ tập trung quanh mức sáng `40..70`, còn nền biển tập trung quanh mức `130..170`.
  - Nếu dùng ngưỡng cứng `180`: cả chữ lẫn nền đều `< 180` nên toàn bộ biển biến thành màu đen (`0`)!
  - Thuật toán Otsu tự động tính toán và đặt ngưỡng ngay thung lũng giữa 2 cụm là `T* = 100` -> Nhóm chữ (`40..70`) biến thành `0` (đen), nhóm nền (`130..170`) biến thành `255` (trắng).
- **Vì sao để `use_threshold=False` làm mặc định?**
  - Như thấy rõ ở cột thứ 5 trong ảnh [preprocess_demo.jpg](file:///d:/hệ thống giao thông/plate_app/reports/preprocess_demo.jpg), khi biển số có vệt bóng đổ chéo rất gắt hoặc khung xe kim loại lóa sáng, một ngưỡng `T*` duy nhất cho toàn ảnh của Otsu có thể làm mất nét chữ ở vùng tối. Trong khi đó, EasyOCR sử dụng mạng CNN bên trong vốn được huấn luyện trên ảnh đa mức xám (256 mức sáng) nên đọc ảnh sau bước **CLAHE** thường ổn định hơn ảnh nhị phân trắng-đen cứng. Ở Giai đoạn 3, hàm `so_sanh_tien_xu_ly` sẽ kiểm chứng bằng số liệu thực tế giữa 3 chế độ này.

---

## GIAI ĐOẠN 2: PHÁT HIỆN BIỂN SỐ (YOLO) & ĐÁNH GIÁ MÔ HÌNH (`train.py`, `detect.py`, `evaluate.py`)

### 1. Cách YOLO phát hiện biển số & Ảnh hưởng của kích thước ảnh đầu vào (`imgsz`)
- **Ý tưởng:** Khác với các thuật toán cũ phải dùng cửa sổ trượt quét hàng nghìn lần qua bức ảnh, **YOLO** (*You Only Look Once*) đưa bức ảnh qua mạng CNN đúng 1 lần. Mạng chia ảnh thành các lưới ô vuông ở 3 tỉ lệ (bước nhảy `stride = 8, 16, 32` pixel). Mỗi ô lưới dự đoán tọa độ tâm `(cx, cy)`, chiều rộng `w`, chiều cao `h` và độ tự tin `conf`. Cuối cùng, thuật toán **Non-Maximum Suppression (NMS)** loại bỏ các khung bao trùng lặp cùng chỉ vào một biển số, chỉ giữ lại khung có độ tự tin cao nhất.
- **Ví dụ số nhỏ về ảnh hưởng của `imgsz` (`640` vs `960`):**
  - Nhánh phát hiện vật nhỏ nhất của YOLOv8 có bước nhảy `stride = 8` (cứ mỗi khối `8x8` pixel trên ảnh đầu vào gom thành 1 ô đặc trưng).
  - Giả sử ảnh `kaggle` gốc (`1920x1080`) có một chiếc xe máy ở xa với chiều cao biển số là `24 px`:
    - Nếu thu ảnh về `imgsz = 640` (nhỏ đi 3 lần): chiều cao biển chỉ còn `24 / 3 = 8 px` -> vừa khít 1 ô lưới duy nhất (`8x8`), rất dễ bị lẫn vào nền đường.
    - Nếu thu ảnh về `imgsz = 960` (nhỏ đi 2 lần): chiều cao biển còn `24 / 2 = 12 px` -> trải qua gần 2 ô lưới, giúp mạng dễ nhận ra hình dáng biển số hơn.

---

### 2. Chỉ số Giao trên Hợp (`IoU - Intersection over Union`)
- **Ý tưởng:** Làm sao máy tính biết khung dự đoán màu đỏ của YOLO có "trúng" khung nhãn thật màu xanh lá hay không? Ta dùng chỉ số **IoU**: lấy **diện tích phần giao nhau (chồng lên nhau)** chia cho **diện tích phần hợp (tổng diện tích bao phủ của cả hai khung)**.
  - Nếu 2 khung hoàn toàn rời nhau: `IoU = 0.0`.
  - Nếu 2 khung trùng khít 100%: `IoU = 1.0`.
  - Quy ước chuẩn quốc tế (PASCAL VOC / COCO): `IoU >= 0.50` (chồng khít từ một nửa diện tích trở lên) được tính là **phát hiện đúng**.
- **Ví dụ số nhỏ:**
  - Khung biển số thật `A` là hình chữ nhật từ `x = 0..100`, `y = 0..100` -> Diện tích `Area(A) = 100 * 100 = 10,000`.
  - Khung dự đoán `B` bị lệch sang phải một chút: `x = 20..120`, `y = 0..100` -> Diện tích `Area(B) = 100 * 100 = 10,000`.
  - Phần giao nhau (từ `x = 20..100`, `y = 0..100`) có rộng `80`, cao `100` -> `Giao = 80 * 100 = 8,000`.
  - Phần hợp = `Area(A) + Area(B) - Giao = 10,000 + 10,000 - 8,000 = 12,000`.
  - `IoU = 8,000 / 12,000 = 0.667` (`>= 0.50` -> Tính là **Phát hiện đúng - True Positive**).

---

### 3. Thuật toán Ghép cặp tham lam (`Greedy Matching`) trong `evaluate_utils.py`
- **Ý tưởng:** Một bức ảnh có thể có nhiều biển số thật và nhiều khung dự đoán. Ta ghép cặp theo nguyên tắc **1 - 1**:
  - Duyệt từng khung biển số thật: tìm trong số các khung dự đoán **chưa được ghép** xem khung nào có `IoU` lớn nhất với nó.
  - Nếu `IoU lớn nhất >= 0.50`: ghép cặp thành công (**True Positive - TP**) và đánh dấu khung dự đoán đó là "đã dùng" (không được ghép cho biển thật khác nữa).
  - Khung biển số thật nào không tìm được dự đoán thỏa mãn -> **Bỏ sót (False Negative - FN)**.
  - Khung dự đoán nào còn dư sau khi duyệt hết biển thật -> **Báo nhầm (False Positive - FP)**.
- **Ví dụ số nhỏ:**
  - Trong ảnh chỉ có **1 biển thật `G1`**, nhưng mô hình vẽ ra **2 khung dự đoán `P1` (`IoU = 0.82`) và `P2` (`IoU = 0.60`)** chồng lên cùng biển đó.
  - `G1` chọn `P1` (vì `0.82 > 0.60`) -> `G1` được tính là **1 phát hiện đúng (`TP = 1`)**, `P1` được đánh dấu đã dùng.
  - Khung `P2` dù có `IoU = 0.60` nhưng không còn biển thật nào để ghép -> `P2` bị tính là **1 ca phát hiện nhầm dư thừa (`FP = 1`)**.

---

### 4. Độ chính xác (`Precision`), Độ phủ (`Recall`) và `mAP` (`mAP50`, `mAP50-95`)
- **Ý tưởng & Công thức đời thường:**
  - **`Precision` (Độ chuẩn xác khi báo):** Trong tất cả những khung mà mô hình khoanh và bảo là biển số, có bao nhiêu phần trăm thực sự là biển số?
    `Precision = TP / (TP + FP)` (Càng ít báo nhầm `FP` thì `Precision` càng cao).
  - **`Recall` (Độ nhạy / Không bỏ sót):** Trong tất cả các biển số thực tế có mặt trên đường, mô hình tìm ra được bao nhiêu phần trăm?
    `Recall = TP / (TP + FN)` (Càng ít bỏ sót `FN` thì `Recall` càng cao).
  - **`mAP50` (Mean Average Precision tại `IoU = 0.50`):** Khi hạ ngưỡng tự tin `conf`, `Recall` tăng nhưng `Precision` giảm. Đường cong nối các cặp `(Recall, Precision)` tạo thành đường `PR_curve`. **Diện tích nằm dưới đường cong PR** đó chính là `AP` (với bài toán 1 lớp biển số thì `mAP50 = AP50`).
  - **`mAP50-95`:** Tính `mAP` ở 10 mức khắt khe tăng dần (`IoU = 0.50, 0.55, 0.60, ..., 0.95`) rồi lấy trung bình cộng. Mô hình muốn đạt `mAP50-95` cao thì không chỉ tìm đúng biển mà còn phải **khoanh thật sát viền biển số**.
- **Ví dụ số nhỏ:**
  - Tập kiểm thử có `100` biển số thật. Mô hình khoanh ra `95` khung, trong đó có `90` khung trúng biển thật (`TP = 90`), `5` khung khoanh nhầm vào chữ quảng cáo (`FP = 5`), và bỏ sót `10` biển thật ở xa (`FN = 10`).
  - `Precision = 90 / (90 + 5) = 90 / 95 = 94.74%`.
  - `Recall = 90 / (90 + 10) = 90 / 100 = 90.00%`.

---

### 5. Lấy mẫu phân tầng (`Stratified Sampling`) khi tạo bộ kiểm thử OCR
- **Ý tưởng:** Tập `test` có `825` ảnh `roboflow` nhưng chỉ có `50` ảnh `kaggle`. Nếu bốc ngẫu nhiên 150 biển số hoàn toàn tự do, gần như 95% mẫu bốc ra sẽ thuộc về `roboflow`, khiến ta không đánh giá được EasyOCR hoạt động thế nào trên ảnh `kaggle` hay trên biển nhỏ `<30px`.
  **Lấy mẫu phân tầng** chia toàn bộ các biển số ứng viên vào **12 ngăn (ô) riêng biệt** theo 3 tiêu chí: `Nguồn (2) x Loại biển vuông/dài (2) x Nhóm chiều cao gốc <30 / 30-60 / >60px (3) = 12 ô`. Sau đó lấy đều từ mỗi ô khoảng `12-13` mẫu (ô nào ít hơn chỉ tiêu thì lấy hết 100% và chia đều phần thiếu cho các ô còn dư).
- **Ví dụ số nhỏ:**
  - Cần chọn `150` biển từ `12` ô -> chỉ tiêu ban đầu mỗi ô là `150 // 12 = 12` biển (dư 6).
  - Giả sử ô `(kaggle, long, <30px)` ngoài thực tế chỉ có `2` biển ứng viên -> Ta lấy cả `2` biển đó, còn thiếu `10` biển sẽ được chia đều sang các ô còn nhiều mẫu để tổng cuối cùng vẫn đủ đúng **150 biển** và đại diện đầy đủ mọi tình huống khó/dễ.

---

### 6. Chia bộ nhãn OCR thành `dev` / `final` và vì sao không chỉnh tham số trên tập dùng để báo cáo
- **Ý tưởng:** Trong 150 ảnh biển số kiểm thử OCR, ta chia tiếp mỗi ô phân tầng thành 2 phần: **`dev` (~2/3, 101 biển)** và **`final` (~1/3, 49 biển)**.
  - Tập **`dev`** dùng để thử nghiệm, chọn chế độ tiền xử lý (`raw` vs `CLAHE` vs `Otsu`) và tinh chỉnh quy tắc sửa ký tự nhầm.
  - Tập **`final`** được giữ kín và **chỉ chạy đúng 1 lần duy nhất ở cuối cùng** để báo cáo kết quả.
- **Ví dụ nhỏ:**
  - Giả sử trong tập kiểm thử có 1 biển bị bùn che chữ `8` nhìn giống chữ `E`. Nếu người lập trình nhìn vào chính tập báo cáo rồi viết thêm luật riêng `"nếu gặp chữ E ở cuối thì đổi thành 8"`, điểm số trên tập đó sẽ tăng lên nhưng mang ra thực tế gặp biển khác có chữ `E` sẽ sửa sai.
  - Nhờ tách riêng `dev` (để tìm lỗi và chỉnh luật) và `final` (chỉ chấm điểm 1 lần), con số độ chính xác trong báo cáo hoàn toàn khách quan.

---

## GIAI ĐOẠN 3: NHẬN DẠNG KÝ TỰ (EASYOCR) & KIỂM TRA ĐỊNH DẠNG (`ocr.py`, `validate.py`, `evaluate_ocr.py`)

### 1. Cơ chế đọc chữ của EasyOCR (`CRAFT` + `CRNN` + `CTC`)
- **Ý tưởng:** Thư viện **EasyOCR** đọc chữ trên ảnh biển số qua 2 bước nối tiếp nhau:
  1. **Tìm dòng chữ (`CRAFT`):** Mạng CNN đầu tiên quét ảnh để khoanh ra các hộp chữ nhật bao quanh từng cụm ký tự (ví dụ khoanh dòng trên `"59-X1"` và dòng dưới `"123.45"`).
  2. **Đọc chuỗi ký tự (`ResNet + BiLSTM + CTC`):** Mỗi hộp chữ được cắt ra, đưa qua mạng CNN (`ResNet`) để rút trích nét chữ theo chiều ngang từ trái sang phải, rồi qua mạng chuỗi (`BiLSTM`) và bộ giải mã `CTC` để ghép thành chuỗi văn bản kèm điểm tự tin `conf` (`0.0 -> 1.0`).
- **Ví dụ số nhỏ:**
  - Hộp chữ `"30A-123.45"` được chia dọc thành 20 lát cắt từ trái sang phải.
  - Mạng dự đoán từng lát: `[3, 3, -, 0, -, A, -, 1, 2, 3, ., 4, 5, 5]`.
  - Bộ giải mã `CTC` gộp các chữ giống nhau liền kề và bỏ dấu nghỉ `-` -> thu được chuỗi `"30A-123.45"` với độ tự tin `conf = 0.91`.
- **Vì sao dùng `allowlist`:** Giới hạn bảng chữ cái chỉ gồm `0-9`, `A-Z`, `Đ`, `-`, `.` giúp EasyOCR không bao giờ đoán nhầm vết bẩn thành các dấu câu lạ (`?`, `!`, `@`, `/`).

---

### 2. Gom hộp chữ thành dòng theo tọa độ tâm `y` (`group_into_lines`)
- **Ý tưởng:** Biển 2 dòng (xe máy, ô tô vuông) có thể được EasyOCR trả về thành 2 hoặc 3 hộp chữ rời rạc theo thứ tự bất kỳ. Thay vì đoán cứng theo tỉ lệ khung ảnh `w/h`, ta gom dòng trực tiếp bằng hình học:
  - Tính tọa độ tâm dọc `cy` và chiều cao `bh` của mỗi hộp chữ.
  - Hai hộp chữ được xếp vào **cùng một dòng** nếu khoảng cách dọc giữa hai tâm nhỏ hơn một nửa chiều cao hộp (`|cy_1 - cy_2| < 0.5 * bh`).
  - Trong cùng một dòng, sắp các hộp từ trái sang phải (`cx` tăng dần); sau đó ghép các dòng từ trên xuống dưới (`cy` tăng dần) cách nhau bằng 1 dấu cách.
- **Ví dụ số nhỏ:**
  - Hộp A (`"59-X1"`) có `y` từ `10..50` -> tâm `cy = 30`, chiều cao `bh = 40` (nửa chiều cao là `20`).
  - Hộp B (nếu bị tách rời ở cùng hàng trên) có tâm `cy = 34` -> độ lệch `|34 - 30| = 4 < 20` -> **Gom chung vào Dòng 1**.
  - Hộp C (`"123.45"`) ở hàng dưới có `y` từ `60..100` -> tâm `cy = 80` -> độ lệch `|80 - 30| = 50 >= 20` -> **Tách thành Dòng 2**.
  - Kết quả nối dòng: `"59-X1 123.45"` (`so_dong = 2`).

---

### 3. Sửa ký tự nhầm theo mặt nạ vị trí từng dạng biển (`fix_characters` trong `validate.py`)
- **Ý tưởng:** Mạng OCR rất hay nhầm các cặp ký tự có hình dáng giống nhau:
  - Bảng ánh xạ gốc sang chữ số (`BASE_TO_DIGIT`): `O -> 0, D -> 0, I -> 1, Z -> 2, S -> 5, G -> 6, B -> 8`.
  - Bảng ánh xạ bổ sung sang chữ số (`EXTRA_TO_DIGIT`, bật/tắt qua `RULE_EXTRA_DIGIT_MAP`): `Q -> 0, U -> 0, J -> 3, L -> 4, T -> 7`.
  - Bảng ánh xạ sang chữ cái (`TO_ALPHA`): `0 -> O, 1 -> I, 2 -> Z, 4 -> A, 5 -> S, 6 -> G, 8 -> B`.
  Tuy nhiên, dựa trên định dạng biển phổ biến quan sát được, chưa đối chiếu văn bản quy định, mỗi vị trí trong các mẫu hỗ trợ có khuôn dạng Chữ (`L`) hoặc Số (`D`):
  - Ô tô 5 số (`30A-123.45`): 8 ký tự theo mẫu `DDLDDDDD` (2 số tỉnh + 1 chữ seri + 5 số thứ tự).
  - Ô tô 4 số cũ (`30A-1234`): 7 ký tự theo mẫu `DDLDDDD`.
  - Xe máy 5 số (`59-X1 123.45` hoặc `43-AA 567.89`): 9 ký tự theo mẫu `DDLDDDDDD` hoặc `DDLLDDDDD`.
  - Xe máy 4 số cũ (`29-L1 1234`): 8 ký tự có dấu tách cụm `2-2-4` theo mẫu `DDLDDDDD`.
- **Hạn chế:** Các mẫu kiểm tra trên dựa trên định dạng biển phổ biến quan sát được, chưa đối chiếu văn bản quy định. Biển 4 số có 2 chữ cái, biển quân đội/ngoại giao/công vụ chưa được hỗ trợ.
- **Ví dụ số nhỏ:**
  - OCR đọc ra chuỗi bị nhầm: `"3OA-123.4S"` -> Bỏ dấu câu còn 8 ký tự `"3OA1234S"`, đối chiếu với mặt nạ `DDLDDDDD`:
    - Vị trí 2 cần Số (`D`), đang là chữ `'O'` -> đổi thành `'0'`.
    - Vị trí 3 cần Chữ (`L`), đang là chữ `'A'` -> giữ nguyên `'A'`.
    - Vị trí 8 cần Số (`D`), đang là chữ `'S'` -> đổi thành `'5'`.
  - Chuỗi sau sửa là `"30A-123.45"` khớp hoàn toàn Regex `^\d{2}[A-Z]-\d{3}\.\d{2}$` và có mã tỉnh `30` hợp lệ (`11..99`) -> Chấp nhận kết quả sửa!
  - Ngược lại, nếu chuỗi rác `"ABCDEFGH"` không thể sửa khớp bất kỳ mẫu nào -> **Giữ nguyên `"ABCDEFGH"` và đánh dấu `is_valid = False`** (tuyệt đối không ép sửa bừa).

---

### 4. Khoảng cách chỉnh sửa `Levenshtein` và Tỉ lệ lỗi ký tự `CER` (`Character Error Rate`)
- **Ý tưởng:**
  - Nếu chỉ chấm điểm "Đúng tuyệt đối cả biển" (`Exact Match`), một biển 8 ký tự đọc đúng 7 ký tự (`30A-123.45` đọc thành `30A-128.45`) bị tính `0 điểm` ngang với một biển đọc sai hết toàn bộ.
  - **Khoảng cách Levenshtein** đếm **số thao tác chỉnh sửa tối thiểu** (Thêm 1 ký tự, Xóa 1 ký tự, hoặc Thay thế 1 ký tự) để biến chuỗi dự đoán thành chuỗi nhãn thật.
  - **Tỉ lệ lỗi ký tự (`CER`)** = `Tổng khoảng cách Levenshtein / Tổng số ký tự của nhãn thật`. (`CER` càng gần `0%` càng tốt).
- **Ví dụ số nhỏ:**
  - Nhãn thật (bỏ dấu câu): `GT = "30A12345"` (độ dài `8` ký tự).
  - Dự đoán của OCR: `PR = "30A12845"` (chỉ nhầm đúng số `3` thành số `8`).
  - Chỉ cần **1 phép thay thế** (`'8' -> '3'`) là hai chuỗi giống hệt nhau -> `Khoảng cách Levenshtein = 1`.
  - Tỉ lệ lỗi ký tự: `CER = 1 / 8 = 12.5%` (tức độ chính xác mức ký tự đạt `87.5%`).

---

## GIAI ĐOẠN 4 & 5: CƠ SỞ DỮ LIỆU SQLITE, BỎ PHIẾU VIDEO & GIAO DIỆN (`database.py`, `pipeline.py`, `app.py`)

### 1. Thuật toán Bỏ phiếu số đông giữa các khung hình Video (`Majority Voting`)
- **Ý tưởng:** Khi một chiếc xe chạy qua camera trong 2 giây (60 khung hình), ta không cần xử lý cả 60 khung mà chỉ lấy mẫu mỗi `FRAME_SKIP = 3` khung một lần. Ở một vài khung hình xe còn ở xa hoặc bị bóng cây lướt qua, OCR có thể đọc lệch 1 ký tự; nhưng ở đa số các khung hình khác, OCR đọc đúng.
  Thuật toán **Bỏ phiếu số đông** hoạt động như sau:
  1. **Gom cụm theo không gian & chuỗi:** Một biển số mới đọc được sẽ nhập vào cụm của xe trước đó nếu chuỗi ký tự giống nhau hoặc **chỉ khác tối đa 1 ký tự (`Levenshtein <= 1`)** và vị trí khung hình/thời gian gần nhau.
  2. **Bỏ phiếu chọn kết quả cuối cùng:** Khi video kết thúc, trong mỗi cụm xe, chuỗi biển số nào hợp lệ và **xuất hiện nhiều lần nhất** sẽ thắng cuộc (nếu hòa số phiếu thì chọn chuỗi có độ tự tin `ocr_conf` trung bình cao hơn). Đồng thời lưu lại bức ảnh ở khung hình có điểm `det_conf * ocr_conf` cao nhất làm `best_frame`.
- **Ví dụ số nhỏ:**
  - Một chiếc xe đi qua 5 khung hình được lấy mẫu, kết quả đọc lần lượt là:
    - Khung 0: `"30A-123.45"` (`conf = 0.84`)
    - Khung 3: `"30A-128.45"` (`conf = 0.71` — bị bóng lóa làm số `3` thành `8`, cách 1 ký tự nên vẫn gom vào cùng cụm)
    - Khung 6: `"30A-123.45"` (`conf = 0.92` — rõ nét nhất -> lưu làm `best_frame`)
    - Khung 9: `"30A-123.45"` (`conf = 0.89`)
    - Khung 12: `"30A-123.45"` (`conf = 0.86`)
  - Kiểm phiếu trong cụm 5 lần thấy: `"30A-123.45"` có **4 phiếu**, `"30A-128.45"` có **1 phiếu** -> Kết quả cuối cùng trả về đúng **`"30A-123.45"`** (`seen_count = 5`), loại bỏ hoàn toàn nhiễu ở Khung 3!

---

### 2. Lưu trữ lịch sử bằng `SQLite` và Truy vấn tham số hóa (`?`) chống SQL Injection
- **Ý tưởng:**
  - **SQLite (`sqlite3`)** là hệ quản trị cơ sở dữ liệu quan hệ gọn nhẹ có sẵn trong Python chuẩn, lưu toàn bộ bảng `detections` vào một file duy nhất (`data/plates_history.db`) mà không cần cài đặt máy chủ SQL phức tạp. File ảnh gốc và ảnh biển cắt được lưu ra thư mục `data/results/`, còn trong bảng SQLite chỉ lưu đường dẫn tới file ảnh để cơ sở dữ liệu luôn nhẹ và truy vấn nhanh.
  - **Vì sao dùng dấu hỏi chấm (`?`) thay vì cộng chuỗi `f"SELECT ... '{user_input}'"`?** Nếu ghép chuỗi trực tiếp từ ô tìm kiếm của người dùng vào câu lệnh SQL, khi người dùng gõ ký tự đặc biệt như `' OR '1'='1`, câu lệnh SQL sẽ bị biến dạng (**lỗi bảo mật SQL Injection**). Khi dùng tham số `?`, SQLite tách biệt hoàn toàn phần khung lệnh SQL và phần dữ liệu thuần túy, vừa an toàn tuyệt đối vừa không bị lỗi khi tên file có dấu nháy.





