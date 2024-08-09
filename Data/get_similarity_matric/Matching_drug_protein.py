import json

import pandas as pd

# 读取两个CSV文件
df1 = pd.read_csv("unique_drugbank.csv")
df2 = pd.read_csv("DrugBank_p.txt", sep=' ', header=None, names=['drug_name', 'protein_name','smile','sequence','lable'])



for i, df1_drug_name in enumerate(df1['id']):
    matched_rows = df2.loc[df2['drug_name'] == df1_drug_name, ['protein_name', 'sequence','lable']]
    # 检查是否有匹配的行
    if not matched_rows.empty:
        for j, (_, row) in enumerate(matched_rows.iterrows()):
            # 新增列名
            col_name = f"target{j+1}"
            # 将蛋白质名称、序列和标签信息存储在列表中
            info_list = [row['protein_name'], row['sequence'], row['lable']]
            info_str = json.dumps(info_list)
            # 将信息存储在新列中
            # df1.loc[df1['id'] == df1_drug_name, col_name] = info_list
            df1.loc[df1['id'] == df1_drug_name, col_name] = info_str


df1.to_csv('drug_target.csv',index=False)



