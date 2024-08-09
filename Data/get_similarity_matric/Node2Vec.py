import pandas as pd
from torch_geometric.data import Data
from torch_geometric.nn import Node2Vec
import torch

from get_edge_index import DrugTargetGraph



device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

edge_index = pd.read_csv('edge_index.csv')
edge_index = torch.tensor(edge_index.values, dtype=torch.long).t().contiguous()
lable = pd.read_csv('lable.csv')
data = Data(edge_index = edge_index,edge_attr=lable)
# lable = torch.tensor(lable['edge_label_column_name'], dtype=torch.float)  # 假设边标签数据存储在 'edge_label_column_name' 列中
model = Node2Vec(data.edge_index, embedding_dim=128, walk_length=10,
                              context_size=5, walks_per_node=8, num_negative_samples=1,
                              p=0.25, q=4, sparse=True).to(device)
loader = model.loader(batch_size=128, shuffle=True, num_workers=4)
pos_rw, neg_rw = next(iter(loader))
optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)



def train():
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)





for epoch in range(1, 1001):
    loss = train()
    # acc = test()
    print(f'Epoch:{epoch:02d}, Loss: {loss:.4f}')

node_embeddings = model()
embed = pd.DataFrame(node_embeddings.cpu().detach().numpy())
embed.to_csv('embed.csv')

emb = pd.read_csv('embed.csv')
emb.rename(columns={emb.columns[0]:'id'},inplace=True)
emb.to_csv('embedding.csv',index=False)
print(emb.shape)