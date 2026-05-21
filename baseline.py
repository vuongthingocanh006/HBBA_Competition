import pandas as pd
import numpy as np

print("1. Đọc dữ liệu gốc train.csv và sample_submission.csv")
df = pd.read_csv('final_data.csv')
sub = pd.read_csv('sample_submission.csv')

# Ép kiểu ngày tháng
df['Date'] = pd.to_datetime(df['Date'])

print("2. Gộp dữ liệu giao dịch theo Ngày và SKU...")
daily_df = df.groupby(['Date', 'ItemCode'], as_index=False)['Quantity'].sum()

# Khử các giá trị âm (nếu tổng ngày bị âm do trả hàng) về 0
daily_df['Quantity'] = daily_df['Quantity'].clip(lower=0)

print("3. Trích xuất chuỗi 56 ngày cuối cùng trong lịch sử tập Train...")
max_date = daily_df['Date'].max() # Ngày 2025-09-05

# Định vị mốc thời gian lấy dữ liệu quá khứ
# Chu kỳ 1 (Tương ứng F1 -> F28): 28 ngày cuối cùng
history_cycle_1 = daily_df[(daily_df['Date'] > max_date - pd.Timedelta(days=28)) & (daily_df['Date'] <= max_date)]

# Chu kỳ 2 (Tương ứng F29 -> F56): 28 ngày trước đó nữa
history_cycle_2 = daily_df[(daily_df['Date'] > max_date - pd.Timedelta(days=56)) & (daily_df['Date'] <= max_date - pd.Timedelta(days=28))]

print("4. Chuyển đổi dữ liệu lịch sử thành định dạng hàng ngang (Pivot) từ F1 đến F28...")

# Sắp xếp thời gian tăng dần để ánh xạ đúng thứ tự ngày F1 -> F28
history_cycle_1 = history_cycle_1.sort_values(by=['ItemCode', 'Date'])
history_cycle_2 = history_cycle_2.sort_values(by=['ItemCode', 'Date'])

# Tạo số cột F tương ứng cho từng dòng
history_cycle_1['day_index'] = history_cycle_1.groupby('ItemCode').cumcount() + 1
history_cycle_1['F_col'] = 'F' + history_cycle_1['day_index'].astype(str)

history_cycle_2['day_index'] = history_cycle_2.groupby('ItemCode').cumcount() + 1
history_cycle_2['F_col'] = 'F' + history_cycle_2['day_index'].astype(str)

# Xoay bảng (Pivot table) sang dạng chiều ngang
pivot_val = history_cycle_1.pivot(index='ItemCode', columns='F_col', values='Quantity').reset_index()
pivot_eval = history_cycle_2.pivot(index='ItemCode', columns='F_col', values='Quantity').reset_index()

print("5. Ánh xạ kết quả vào file mẫu nộp bài (Sample Submission)...")

# Tạo cột nối tạm thời trong file submission
sub['ItemCode'] = sub['id'].str.replace('_validation|_evaluation', '', regex=True)
sub['is_val'] = sub['id'].str.contains('_validation')

# Tách file sub thành 2 phần để ghép cho chuẩn
sub_val = sub[sub['is_val'] == True].drop(columns=[f'F{i}' for i in range(1, 29)])
sub_eval = sub[sub['is_val'] == False].drop(columns=[f'F{i}' for i in range(1, 29)])

# Trộn dữ liệu quá khứ tương ứng vào
sub_val = pd.merge(sub_val, pivot_val, on='ItemCode', how='left')
sub_eval = pd.merge(sub_eval, pivot_eval, on='ItemCode', how='left')

# Ghép 2 nửa lại thành file submission hoàn chỉnh
final_sub = pd.concat([sub_val, sub_eval], axis=0)

# Điền các giá trị thiếu bằng 0 (nếu có SKU nào không bán gì trong 56 ngày cuối)
final_sub = final_sub.fillna(0)

# Sắp xếp lại đúng thứ tự dòng ban đầu của ban tổ chức và loại bỏ các cột nháp
final_sub = final_sub.set_index('id').loc[sub['id']].reset_index()
cols_to_keep = ['id'] + [f'F{i}' for i in range(1, 29)]
final_sub = final_sub[cols_to_keep]

# Đảm bảo các cột kết quả dự báo là số nguyên (vì số lượng bán là số nguyên)
for i in range(1, 29):
    final_sub[f'F{i}'] = final_sub[f'F{i}'].astype(int)

# 6. Xuất file kết quả
output_file = 'naive_seasonal_submission.csv'
final_sub.to_csv(output_file, index=False)
print(f"🎉 Xuất file thành công: '{output_file}'!")