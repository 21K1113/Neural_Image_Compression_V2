import pandas as pd

# CSVファイルの読み込み
input_file = 'log.csv'  # 元のCSVファイル名
output_file = 'data_log.csv'  # 結果を保存する新しいCSVファイル名

# CSVのデータをDataFrameに読み込む
df = pd.read_csv(input_file)

def fp_data(row):
    if row['METHOD'] == 2:
        fp = (row["G0 BIT"]/8 * pow(row['IMAGE SIZE']//pow(2, row["G0 RES"]), 2) * row["G0 CH"] +
              row["G1 BIT"]/8 * pow(row['IMAGE SIZE']//pow(2, row["G0 RES"]+1), 2) * row["G1 CH"])
    elif row['METHOD'] == 3 or row['METHOD'] == 4:
        fp = (row["G0 BIT"] / 8 * pow(row['IMAGE SIZE'] // pow(2, row["G0 RES"]), 3) * row["G0 CH"] +
              row["G1 BIT"] / 8 * pow(row['IMAGE SIZE'] // pow(2, row["G0 RES"] + 1), 3) * row["G1 CH"])
    return fp

def decoder_data(row):
    h = 64
    b = 4
    if row['METHOD'] == 2:
        decoder = ((row["G0 CH"] * 4 + row["G1 CH"] + 6 * 2 + 1 + 1) * h + (h + 1) * h + (h + 1) * 3) * b
    elif row['METHOD'] == 3:
        decoder = ((row["G0 CH"] * 8 + row["G1 CH"] + 6 * 3 + 1 + 1) * h + (h + 1) * h + (h + 1) * 3) * b
    elif row['METHOD'] == 4:
        decoder = ((row["G0 CH"] * 4 + row["G1 CH"] + 6 * 3 + 1 + 1) * h + (h + 1) * h + (h + 1) * 3) * b
    return decoder


df["FP DATA"] = df.apply(fp_data, axis=1)

# 'END DATE TIME' の隣に新しい列を挿入
end_date_time_index = df.columns.get_loc('END DATE TIME') + 1
df.insert(end_date_time_index, "FP DATA", df.pop("FP DATA"))

df["DECODER DATA"] = df.apply(decoder_data, axis=1)

# 'END DATE TIME' の隣に新しい列を挿入
end_date_time_index = df.columns.get_loc('FP DATA') + 1
df.insert(end_date_time_index, "DECODER DATA", df.pop("DECODER DATA"))

df["SUM DATA"] = df["FP DATA"] + df["DECODER DATA"]

# 'END DATE TIME' の隣に新しい列を挿入
end_date_time_index = df.columns.get_loc('DECODER DATA') + 1
df.insert(end_date_time_index, "SUM DATA", df.pop("SUM DATA"))

# 結果を新しいCSVファイルに保存（元のファイルを上書きしない）
df.to_csv(output_file, index=False)

print(f'処理が完了しました。結果は {output_file} に保存されました。')
