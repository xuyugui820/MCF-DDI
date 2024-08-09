# # from scipy.spatial.distance import cosine
# # import numpy as np
# # import pandas as pd
# # # 假设embeddings是一个n x d的矩阵，其中n是节点的数量，d是嵌入的维度
# # embeddings = pd.read_csv('embedding.csv',nrows=1705,usecols=range(0,129))
# # embeddings_array = embeddings.values
# # # 归一化嵌入向量
# # normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
# #
# # # 计算余弦相似性矩阵
# # cosine_similarity_matrix = 1 - np.array([[cosine(normalized_embeddings[i], normalized_embeddings[j])
# #                                           for j in range(len(embeddings))]
# #                                          for i in range(len(embeddings))])
# # cosine_similarity_matrix = pd.DataFrame(cosine_similarity_matrix)
# # cosine_similarity_matrix.to_csv('cosine_similarity_matrix.csv',index=True)
#
# from scipy.spatial.distance import cosine
# import numpy as np
# import pandas as pd
#
# # 读取包含嵌入向量的 CSV 文件
# data = pd.read_csv('embedding.csv')
# # 提取 ID 列和嵌入向量列
# ids = data.iloc[:, 0]  # 第一列是 ID 列
# embeddings = data.iloc[:, 1:]  # 剩下的列是嵌入向量列
#
# # 将嵌入向量转换为 NumPy 数组
# embeddings_array = embeddings.values
# # 归一化嵌入向量
# normalized_embeddings = embeddings_array / np.linalg.norm(embeddings_array, axis=1)[:, np.newaxis]
# # 计算余弦相似性矩阵
# cosine_similarity_matrix = 1 - np.array([[cosine(normalized_embeddings[i], normalized_embeddings[j])
#                                           for j in range(len(normalized_embeddings))]
#                                          for i in range(len(normalized_embeddings))])
# # 创建带有 ID 列和余弦相似性值的 DataFrame
# similarity_matrix = pd.DataFrame(cosine_similarity_matrix, index=ids, columns=ids)
#
# # 保存余弦相似性矩阵到 CSV 文件
# similarity_matrix.to_csv('cosine_similarity_matrix.csv', index=True)
import pandas as pd

#在GPU上计算余弦相似度
import torch
import pandas as pd
import numpy as np

# 读取包含嵌入向量的 CSV 文件
data = pd.read_csv('embedding.csv')
# 提取 ID 列和嵌入向量列
ids = data.iloc[0:1705, 0]  # 第一列是 ID 列
embeddings = data.iloc[0:1705, 1:]  # 只取第一行到第1706行，以及从第二列开始的所有列

# 将嵌入向量转换为 PyTorch 张量
embeddings_tensor = torch.tensor(embeddings.values, dtype=torch.float32)

# 将张量移到 GPU 上
if torch.cuda.is_available():
    device = torch.device("cuda")
    embeddings_tensor = embeddings_tensor.to(device)

# 归一化嵌入向量
normalized_embeddings = embeddings_tensor / torch.norm(embeddings_tensor, dim=1, keepdim=True)

# 计算余弦相似性矩阵
cosine_similarity_matrix = torch.mm(normalized_embeddings, normalized_embeddings.t())

# 创建带有 ID 列和余弦相似性值的 DataFrame
similarity_matrix = pd.DataFrame(cosine_similarity_matrix.cpu().numpy(), index=ids, columns=ids)

# 保存余弦相似性矩阵到 CSV 文件
similarity_matrix.to_csv('cosine_similarity_matrix.csv', index=True)





#将drugbank id 加入相似性矩阵
pd_cosine_similarity_matrix = pd.read_csv('cosine_similarity_matrix.csv')
drug_target_number = pd.read_csv('drug_target_number.csv')
drugbank_id = drug_target_number['id']
pd_cosine_similarity_matrix.insert(1, 'drugbank_id', drugbank_id)
pd_cosine_similarity_matrix.to_csv('cosine_similarity_did_matrix.csv', index=False)