# Nghiên cứu dự báo nhu cầu POS

Python là nơi duy nhất tính dự báo và backtest. Node.js lấy dữ liệu POS, lưu
kết quả và áp dụng rule cảnh báo, gợi ý nhập hàng; chủ cửa hàng vẫn phải duyệt
PO nháp. LightGBM hiện được huấn luyện trên dữ liệu giả lập, không phải bằng
chứng về độ chính xác tại cửa hàng thật.

## Thiết lập trên Windows

Yêu cầu Python 3.12 và terminal PowerShell trong VS Code. Cài Python chính thức,
sau đó mở terminal mới và tạo môi trường ảo một lần từ thư mục này:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tạo bộ dữ liệu nghiên cứu và chạy mô hình:

```powershell
.\.venv\Scripts\python.exe synthetic.py --out data\seed-42.json --seed 42 --skus 300 --days 365
.\.venv\Scripts\python.exe research.py --dataset data\seed-42.json --out artifacts\run-42 --seed 42
```

Đường dẫn dữ liệu và thư mục kết quả phải chưa tồn tại; lệnh sẽ không ghi đè.
Thay seed và tên đường dẫn để lặp lại thí nghiệm. Thêm `--no-impute` vào lệnh
nghiên cứu để thử không bù ngày hết hàng, dùng một thư mục kết quả khác.

Generator tạo nhật ký theo ngày từ tồn đầu, nhập, trả bán được, xuất hủy và bán.
`latent_demand` là đáp án giả lập chỉ dùng đánh giá; không dùng làm đặc trưng
huấn luyện hoặc đưa vào API. Bộ này không tạo dữ liệu giao dịch cho PostgreSQL.

## Quy trình hiện tại

- Mục tiêu dự báo là số đơn vị POS bán trong kỳ, trước khi trừ hàng trả. Đây là
  thay đổi so với dự báo Node.js cũ dùng số bán ròng; các rule tồn kho khác vẫn
  dùng chỉ số bán ròng như trước.
- LightGBM huấn luyện ba tổng không âm cho ngày 1–7, 8–14 và 15–28. Các mô hình
  so sánh là trung bình 28 ngày, SES và Croston-SBA.
- Chia thời gian 60% train, 20% validation, 20% test; giữ riêng mỗi SKU thứ năm
  khỏi train. Mỗi mốc test dùng lịch sử trước mốc; nhãn train phải kết thúc trong
  train. Các cửa sổ train bị hết hàng trong kỳ nhãn bị loại.
- `report.json` báo MAE, WAPE, bias trên SKU đã thấy và chưa thấy, theo kịch bản
  và theo kỳ; báo riêng nhu cầu ẩn giả lập và các cửa sổ bán quan sát không hết
  hàng. WAPE/bias là `null` nếu tổng thực tế bằng 0.
- Tham số mô hình cố định, chưa tối ưu trên validation. Cần chạy nhiều seed và
  phân tích kết quả. Báo cáo cũng có mô phỏng chính sách nhập đơn giản trên nhu
  cầu ẩn, với lead time 5 ngày, chu kỳ 7 ngày, tồn an toàn 4 đơn vị; không có PO
  đầu kỳ hoặc hàng trả. Nó không thay thế mô phỏng đầy đủ luồng nhập thực tế.
- Dữ liệu giả lập có thể quá đơn giản; điểm số không thể khẳng định mô hình tốt
  trên dữ liệu thật. Mức độ tin cậy và khoảng dự báo chưa được hiệu chuẩn.

## API nội bộ

Mô hình chỉ được tải từ thư mục artifact cục bộ do người vận hành chỉ định;
service kiểm tra checksum và schema đặc trưng. Trước khi chạy, cấu hình
`FORECAST_RESEARCH_TOKEN` trong `abc-backend\.env`. Token dùng chữ, số, dấu
gạch dưới hoặc gạch ngang; không đưa token lên Git.

Mỗi lần khởi động máy, mở terminal PowerShell trong VS Code, chuyển vào
`abc-ml` và chạy:

```powershell
.\start-service.ps1
```

Giữ terminal mở khi sử dụng AI; `Ctrl+C` để dừng. Script dùng model
`artifacts\run-42` và token trong backend `.env`, không in token ra màn hình.
Nếu cổng 8001 đang được sử dụng, dừng service cũ trước khi chạy.

`GET /health` trả trạng thái sẵn sàng. `POST /v1/forecasts:predict` yêu cầu header
`X-Forecast-Token`; truyền `request_id`, `as_of` và `variants`, mỗi SKU gồm
`variant_id` và `history` liên tiếp tới `as_of`. Mỗi ngày có `date`,
`units_sold`, `stockout` (true, false hoặc null). `min_series_days` mặc định 28,
cho phép tăng theo cấu hình. Chuỗi ngắn, có ngày tồn kho không rõ hoặc dưới 7 ngày
đóng cửa còn hàng trong 28 ngày gần nhất trả `INSUFFICIENT_DATA`; ngày thiếu/trùng hoặc không kết thúc
tại `as_of` trả `INVALID_HISTORY`. Không gửi trường đáp án ẩn. Tối đa 200 SKU
mỗi request; backend chia batch.

Khi đủ dữ liệu, Python so sánh tất cả phương pháp bằng tối đa bốn kỳ rolling-origin
7 ngày gần nhất, chỉ dùng kỳ mà mọi ngày được xác nhận không hết hàng. Dự báo của
mỗi mốc chỉ dùng lịch sử trước mốc; LightGBM dùng artifact cố định huấn luyện
trên dữ liệu giả lập. Lịch sử trước ngày tồn kho không rõ gần nhất không được
đưa vào các phương pháp; tồn cuối ngày không chứng minh có hàng cả ngày. Chọn trung bình 28 ngày nếu có dưới hai kỳ hợp lệ hoặc
không tính được WAPE; chỉ chuyển phương pháp khi WAPE thấp hơn baseline hơn 10%.
Độ lệch chuẩn lấy từ các ngày không hết hàng trong 28 ngày gần nhất; mức tin cậy
là phân loại theo backtest, không phải khoảng xác suất được hiệu chuẩn.

Để chạy dự báo trong ứng dụng, cấu hình backend với
`FORECAST_RESEARCH_URL=http://127.0.0.1:8001` và `FORECAST_RESEARCH_TOKEN`
bằng cùng token của service, sau đó khởi động lại backend. Service cần chạy trên
cùng máy; backend chỉ cho phép địa chỉ loopback này và không ghi token vào log.
Chạy phân tích POS để cập nhật dự báo. Nếu service lỗi, các rule cảnh báo độc
lập vẫn chạy; kết quả dự báo cũ được đánh dấu và không thể tạo PO nháp từ gợi ý
cũ. Python không chạy thì hệ thống không tự thay bằng dự báo Node.js. Sau khi cập
nhật mã, khởi động lại cả Python và backend; backend yêu cầu phản hồi có
`stock_history_policy: known-v1` để không dùng nhầm service Python phiên bản cũ.
