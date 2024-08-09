import pandas as pd
import torch


class DrugTargetGraph:
    def __init__(self, drug_file='drug_target_number.csv', target_file='target_number.csv'):
        self.df_drug = pd.read_csv(drug_file)
        self.df_target = pd.read_csv(target_file)
        self.edge_index = []
        self.lable = []
    def generate_edge_index(self):
        for index, row in self.df_drug.iterrows():
            drug_id = row['number']
            for i in range(1, 74):
                target_value = row[f'target{i}']
                target_value = str(target_value)
                if target_value == 'nan':
                    if i == 1:
                        self.edge_index.append([drug_id, drug_id])
                        self.lable.append(0)
                    continue
                target_idname = str(row[f'target{i}']).split(',')[0].strip("[").strip("'").strip('"')
                drug_target_id = float((row[f'target{i}']).split(',')[2].strip("]"))
                target_id = self.df_target.loc[self.df_target['target'] == target_idname, 'number'].values[0]
                self.edge_index.append([drug_id, target_id])
                self.lable.append(drug_target_id)
                self.edge_index.append([target_id, drug_id])
                self.lable.append(drug_target_id)

        return self.edge_index,self.lable
def save_to_file(edge_index,lable):
    # 将edge_index和lable转换为pandas DataFrame
    edge_index_df = pd.DataFrame(edge_index, columns=['source', 'target'])
    lable_df = pd.DataFrame(lable, columns=['lable'])

    # 保存到CSV文件
    edge_index_df.to_csv('edge_index.csv', index=False)
    lable_df.to_csv('lable.csv', index=False)
# # 使用示例
graph = DrugTargetGraph()
edge_index,lable = graph.generate_edge_index()
edge_index = torch.tensor(edge_index)
lable = torch.tensor(lable)
save_to_file(edge_index,lable)

