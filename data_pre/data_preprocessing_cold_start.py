import dgl
from numpy.lib.function_base import append
import torch
from itertools import chain
from collections import defaultdict

from rdkit.Chem import AllChem
from torch_geometric.data import Data
from rdkit import Chem, DataStructs
import pandas as pd
import numpy as np

from tqdm import tqdm
import pickle
import os
from dataset import read_pickle, split_train_valid, DrugDataset
from data_pre import CustomData


def one_of_k_encoding(k, possible_values):
    if k not in possible_values:
        raise ValueError(f"{k} is not a valid value in {possible_values}")
    return [k == e for e in possible_values]


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom, atom_symbols, explicit_H=True, use_chirality=False):
    results = one_of_k_encoding_unk(atom.GetSymbol(), atom_symbols + ['Unknown']) + \
              one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + \
              one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
              [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
              one_of_k_encoding_unk(atom.GetHybridization(), [
                  Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                  Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                    SP3D, Chem.rdchem.HybridizationType.SP3D2
              ]) + [atom.GetIsAromatic()]
    if explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                  [0, 1, 2, 3, 4])
    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
                atom.GetProp('_CIPCode'),
                ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False
                                 ] + [atom.HasProp('_ChiralityPossible')]

    results = np.array(results).astype(np.float32)

    return torch.from_numpy(results)


# def generate_drug_data(mol_graph, atom_symbols):
#
#     edge_list = torch.LongTensor([(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol_graph.GetBonds()])
#     undirected_edge_list = torch.cat([edge_list, edge_list[:, [1, 0]]], dim=0) if len(edge_list) else edge_list
#
#     features = [(atom.GetIdx(), atom_features(atom, atom_symbols)) for atom in mol_graph.GetAtoms()]
#     features.sort() # to make sure that the feature matrix is aligned according to the idx of the atom
#     _, features = zip(*features)
#     features = torch.stack(features)
#
#     return Data(x=features, edge_index=undirected_edge_list.T)
def generate_drug_data(mol_graph, atom_symbols, smiles_rdkit_list, id, smile):
    edge_list = torch.LongTensor(
        [(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), *edge_features(b)) for b in mol_graph.GetBonds()])

    edge_list, edge_feats = (edge_list[:, :2], edge_list[:, 2:].float()) if len(edge_list) else (
        torch.LongTensor([]), torch.FloatTensor([]))  # 将点到点的边序号和对应键分别存储在两个列表中。
    edge_list = torch.cat([edge_list, edge_list[:, [1, 0]]], dim=0) if len(edge_list) else edge_list  # 添加逆向边（变成两倍长度）
    edge_feats = torch.cat([edge_feats] * 2, dim=0) if len(edge_feats) else edge_feats  # 两倍长度。

    features = [(atom.GetIdx(), atom_features(atom, atom_symbols)) for atom in mol_graph.GetAtoms()]
    features.sort()
    _, features = zip(*features)
    features = torch.stack(features)  # 将特征堆叠成一个张量

    line_graph_edge_index = torch.LongTensor([])  # 构建（edge_list，edge_list）的矩阵，如果两个点相连则为true
    if edge_list.nelement() != 0:
        conn = (edge_list[:, 1].unsqueeze(1) == edge_list[:, 0].unsqueeze(0)) & (
                edge_list[:, 0].unsqueeze(1) != edge_list[:, 1].unsqueeze(0))
        line_graph_edge_index = conn.nonzero(as_tuple=False).T  # 找到为true的元素，返回这些位置的坐标（第一个为（2，104）代表（1，0）和（2，0）两个点有边相连）

    new_edge_index = edge_list.T
    fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2) for mol in
           smiles_rdkit_list]  # smiles对象 mol 和半径 2，表示生成 Morgan 指纹时使用的环的半径
    mol_graph_fps = AllChem.GetMorganFingerprintAsBitVect(mol_graph, 2)  # 生成分子图的摩根指纹，使用环的半径
    similarity_matrix = np.zeros((1, len(smiles_rdkit_list)))
    for i in range(len(smiles_rdkit_list)):
        similarity = DataStructs.FingerprintSimilarity(fps[i], mol_graph_fps)
        similarity_matrix[0][i] = similarity
    similarity_matrix = torch.tensor(similarity_matrix)
    drug_target_similar_pd = pd.read_csv('Data/get_similarity_matric/cosine_similarity_did_matrix.csv')
    drug_target_similar = drug_target_similar_pd.loc[drug_target_similar_pd['drugbank_id'] == id, '0':'1704']
    drug_target_similar_array = drug_target_similar.values
    drug_target_similar_tensor = torch.from_numpy(drug_target_similar_array)
    data = CustomData(x=features, edge_index=new_edge_index, line_graph_edge_index=line_graph_edge_index,
                      edge_attr=edge_feats, sim=similarity_matrix, drug_target_sim=drug_target_similar_tensor, id=id,
                      smile=smile)

    #    data_dgl = {'num_atom': features.shape[0], 'atom_type': features.long(), 'bond_type': edge_feats.long(), 'graph' : g}
    return data  # 返回每个原子特征x（33，70）、边对edge_index（2，74）,有边相连矩阵中坐标line_graph_edge_index（2，104）、边特征edge_attr=[74,6]、相似矩阵（1，544）,id='DB09053'


def edge_features(bond):
    bond_type = bond.GetBondType()
    return torch.tensor([
        bond_type == Chem.rdchem.BondType.SINGLE,
        bond_type == Chem.rdchem.BondType.DOUBLE,
        bond_type == Chem.rdchem.BondType.TRIPLE,
        bond_type == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),  # 判断化学键是否处于共轭体系中
        bond.IsInRing()]).long()  # 是否位于环中


def generate_drug_data_dgl(mol_graph, atom_symbols):
    edge_list = torch.LongTensor(
        [(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), *edge_features(b)) for b in mol_graph.GetBonds()])
    edge_list, edge_feats = (edge_list[:, :2], edge_list[:, 2:].float()) if len(edge_list) else (
        torch.LongTensor([]), torch.FloatTensor([]))
    edge_list = torch.cat([edge_list, edge_list[:, [1, 0]]], dim=0) if len(edge_list) else edge_list  # 获得逆向边
    edge_feats = torch.cat([edge_feats] * 2, dim=0) if len(edge_feats) else edge_feats  # 特征与边对应

    features = [(atom.GetIdx(), atom_features(atom, atom_symbols)) for atom in mol_graph.GetAtoms()]  #
    features.sort()
    _, features = zip(*features)
    features = torch.stack(features)
    node_feature = features.long()  #
    edge_feature = edge_feats.long()

    g = dgl.DGLGraph()  # 创建了一个空的图对象 g，该对象可以用来存储和表示图结构
    g.add_nodes(features.shape[0])  # 获得这个smiles分子有多少个原子
    g.ndata['feat'] = node_feature
    for src, dst in edge_list:
        g.add_edges(src.item(), dst.item())
    g.edata['feat'] = edge_feature
    data_dgl = g
    return data_dgl


def load_drug_mol_dat(args):
    data = pd.read_csv(args.dataset_filename)
    drug_id_mol_tup = []
    symbols = list()
    drug_smile_dict = {}
    smiles_rdkit_list = []  # RDKit地址

    for id1, id2, smiles1, smiles2, relation in zip(data[args.c_id1], data[args.c_id2], data[args.c_s1],
                                                    data[args.c_s2], data[args.c_y]):
        drug_smile_dict[id1] = smiles1
        drug_smile_dict[id2] = smiles2

    for id, smiles in drug_smile_dict.items():
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is not None:
            drug_id_mol_tup.append((id, mol))
            symbols.extend(atom.GetSymbol() for atom in mol.GetAtoms())
    for m in drug_id_mol_tup:
        smiles_rdkit_list.append(m[-1])
    symbols = list(set(symbols))  # 列表转换成集合（消除重复）再将集合转换成列表
    ##返回每个原子特征（32，45）、边对（2，70）、有边相连矩阵中坐标（2，104）、边特征（70，6），相似矩阵（1，544）
    drug_data_pyg = {id: generate_drug_data(mol, symbols, smiles_rdkit_list, id, drug_smile_dict[id]) for id, mol in
                     tqdm(drug_id_mol_tup, desc='Processing drugs_pyg')}
    # 创建了一个图对象，并向其中添加了节点、边和节点特征。用于图数据的处理和分析
    drug_data_dgl = {id: generate_drug_data_dgl(mol, symbols) for id, mol in
                     tqdm(drug_id_mol_tup, desc='Processing drugs_dgl')}
    save_data(drug_data_pyg, 'drug_data_coldstart_pyg.pkl', args)
    save_data(drug_data_dgl, 'drug_data_coldstart_dgl.pkl', args)
    return drug_data_pyg, drug_data_dgl


def generate_pair_triples(args):
    pos_triples = []
    drug_ids = []

    with open(f'{args.dirname}/{args.save_dir.lower()}/drug_data_coldstart_pyg.pkl', 'rb') as f:
        drug_ids = list(pickle.load(f).keys())
        drug_ids = set(drug_ids)

    data = pd.read_csv(args.dataset_filename)
    # p_smile1_list = []
    # p_smile2_list = []
    # p_finger1_list = []
    # p_finger2_list = []
    # 读取fingger文件
    # unique_drugbank_finger = pd.read_csv('Data/fingerprint.csv')
    # unique_drugbank_finger['id'] = unique_drugbank_finger['id'].astype(str)
    unique_drugbank_finger = pd.read_csv('Data/fingerprint.csv')
    unique_drugbank_finger['id'] = unique_drugbank_finger['id'].astype(str)
    for id1, id2, relation, p_smile1, p_smile2 in zip(data[args.c_id1], data[args.c_id2], data[args.c_y],
                                                      data[args.c_s1], data[args.c_s2]):
        if ((id1 not in drug_ids) or (id2 not in drug_ids)): continue
        # Drugbank dataset 1-based indexed, substract by one
        # if args.dataset in ('drugbank', ):
        #     relation -= 1
        # row1 = unique_drugbank_finger[unique_drugbank_finger['id'] == id1]
        # row2 = unique_drugbank_finger[unique_drugbank_finger['id'] == id2]
        # finger1 = row1.iloc[:, 2:].values.tolist()
        # finger2 = row2.iloc[:, 2:].values.tolist()
        pos_triples.append([id1, id2, str(relation)])  ## Turn to string for compatibility with id1 and id2 type
        # p_smile1_list.append(p_smile1)
        # p_smile2_list.append(p_smile2)
    if len(pos_triples) == 0:
        raise ValueError('Erroneous dataset! All tuples are invalid.')

    data_statistics = load_data_statistics(pos_triples)
    random_num_gen.shuffle(pos_triples)  # Shuffled in-place.

    old_old_triples = []
    new_old_triples = []
    new_new_triples = []
    old_drug_ids = []
    new_drug_ids = []
    secured_rels = []
    remaining_pos_trip = []

    for triple in pos_triples:
        if triple[2] in secured_rels:
            remaining_pos_trip.append(triple)
        else:
            secured_rels.append(triple[2])
            old_old_triples.append(triple)
            old_drug_ids.extend(triple[:2])

    num_new_drug = np.ceil(args.new_drug_ratio * len(drug_ids)).astype(int)
    total_num_drug = len(drug_ids)
    old_drug_ids = set(old_drug_ids)
    pos_triples = remaining_pos_trip
    remaining_drug_ids = list(drug_ids - old_drug_ids)

    random_num_gen.shuffle(remaining_drug_ids)

    new_drug_ids = set(remaining_drug_ids[:num_new_drug])
    old_drug_ids |= set(remaining_drug_ids[num_new_drug:])

    assert (new_drug_ids & old_drug_ids) == set()
    assert (new_drug_ids | old_drug_ids) == drug_ids

    for item in pos_triples:
        if (item[0] in new_drug_ids) and (item[1] in new_drug_ids):
            new_new_triples.append(item)
        elif (item[0] in old_drug_ids) and (item[1] in old_drug_ids):
            old_old_triples.append(item)
        else:
            new_old_triples.append(item)

    new_drug_ids = np.asarray(list(new_drug_ids))
    old_drug_ids = np.asarray(list(old_drug_ids))
    valid_drug_ids = (new_drug_ids, old_drug_ids)

    # p_nn_smile1 = []
    # p_nn_smile2 = []
    # n_nn_smile = []
    # p_nn_finger1 = []
    # p_nn_finger2 = []
    # n_nn_finger = []
    for pos_tups, desc in [
        (new_new_triples, 's1'),
        (old_old_triples, 'train'),
        (new_old_triples, 's2')
    ]:
        pos_tups = np.array(pos_tups)
        all_samples = []
        for pos_item in tqdm(pos_tups, desc=f'Generating Negative sample for {desc}'):
            temp_neg = []
            h, t, r = pos_item[:3]

            if args.dataset == 'drugbank':
                neg_heads, neg_tails = _normal_batch(h, t, r, args.neg_ent, data_statistics, valid_drug_ids, desc)
                temp_neg = [str(neg_h) + '$h' for neg_h in neg_heads] + \
                           [str(neg_t) + '$t' for neg_t in neg_tails]
            else:
                raise NotImplementedError()
                existing_drug_ids = np.asarray(list(set(
                    np.concatenate(
                        [data_statistics["ALL_TRUE_T_WITH_HR"][(h, r)], data_statistics["ALL_TRUE_H_WITH_TR"][(h, r)]],
                        axis=0)
                )))
                temp_neg = _corrupt_ent(existing_drug_ids, args.neg_ent, valid_drug_ids)

            # all_samples.append('_'.join(map(str, temp_neg[:args.neg_ent])))
            all_samples.append(list(map(str, [h, t])) + list(map(str, temp_neg[:args.neg_ent])))

        all_id = all_samples
        smiles_h_list = []
        smiles_t_list = []
        smiles_n_list = []
        finger_h_list = []
        finger_t_list = []
        finger_n_list = []
        neg_samples = []
        for h_id, t_id, n_id in all_id:
            neg_samples.append(n_id)
            # 去掉额外的 $t 部分
            n_id = n_id.split('$')[0]

            row_h = unique_drugbank_finger[unique_drugbank_finger['id'] == h_id]
            row_t = unique_drugbank_finger[unique_drugbank_finger['id'] == t_id]
            row_n = unique_drugbank_finger[unique_drugbank_finger['id'] == n_id]
            if not row_h.empty:
                smiles_h = row_h.iloc[0]['smiles']
                smiles_t = row_t.iloc[0]['smiles']
                smiles_n = row_n.iloc[0]['smiles']
                finger_h = row_h.iloc[:, 2:].values.tolist()
                finger_t = row_t.iloc[:, 2:].values.tolist()
                finger_n = row_n.iloc[:, 2:].values.tolist()
                # row['fingerprint'] = finger.values.tolist()
                smiles_h_list.append(smiles_h)
                smiles_t_list.append(smiles_t)
                smiles_n_list.append(smiles_n)
                finger_h_list.append(finger_h)
                finger_t_list.append(finger_t)
                finger_n_list.append(finger_n)

            else:
                neg_smiles.append(None)  # 如果找不到对应的 ID，添加 None 到列表中
                print(n_id + str("have not smile"))
        df = pd.DataFrame({'Drug1_ID': pos_tups[:, 0],
                           'Drug2_ID': pos_tups[:, 1],
                           'p_smile1': smiles_h_list,
                           'p_smile2': smiles_t_list,
                           'Y': pos_tups[:, 2],
                           'Neg samples': neg_samples,
                           'Neg_smile': smiles_n_list,
                           'p_finger1': finger_h_list,
                           'p_finger2': finger_t_list,
                           'Neg_finger': finger_n_list})
        filename = f'{args.dirname}/{args.save_dir}/pair_pos_neg_triples-fold{args.seed}-{desc}.csv'
        df.to_csv(filename, index=False)
        print(f'\nData saved as {filename}!')
    save_data(data_statistics, 'data_statistics.pkl', args)


def load_data_statistics(all_tuples):
    print('Loading data statistics ...')
    statistics = dict()
    statistics["ALL_TRUE_H_WITH_TR"] = defaultdict(list)
    statistics["ALL_TRUE_T_WITH_HR"] = defaultdict(list)
    statistics["FREQ_REL"] = defaultdict(int)
    statistics["ALL_H_WITH_R"] = defaultdict(dict)
    statistics["ALL_T_WITH_R"] = defaultdict(dict)
    statistics["ALL_TAIL_PER_HEAD"] = {}
    statistics["ALL_HEAD_PER_TAIL"] = {}

    for h, t, r in tqdm(all_tuples, desc='Getting data statistics'):
        statistics["ALL_TRUE_H_WITH_TR"][(t, r)].append(h)
        statistics["ALL_TRUE_T_WITH_HR"][(h, r)].append(t)
        statistics["FREQ_REL"][r] += 1.0
        statistics["ALL_H_WITH_R"][r][h] = 1
        statistics["ALL_T_WITH_R"][r][t] = 1

    for t, r in statistics["ALL_TRUE_H_WITH_TR"]:
        statistics["ALL_TRUE_H_WITH_TR"][(t, r)] = np.array(list(set(statistics["ALL_TRUE_H_WITH_TR"][(t, r)])))
    for h, r in statistics["ALL_TRUE_T_WITH_HR"]:
        statistics["ALL_TRUE_T_WITH_HR"][(h, r)] = np.array(list(set(statistics["ALL_TRUE_T_WITH_HR"][(h, r)])))

    for r in statistics["FREQ_REL"]:
        statistics["ALL_H_WITH_R"][r] = np.array(list(statistics["ALL_H_WITH_R"][r].keys()))
        statistics["ALL_T_WITH_R"][r] = np.array(list(statistics["ALL_T_WITH_R"][r].keys()))
        statistics["ALL_HEAD_PER_TAIL"][r] = statistics["FREQ_REL"][r] / len(statistics["ALL_T_WITH_R"][r])
        statistics["ALL_TAIL_PER_HEAD"][r] = statistics["FREQ_REL"][r] / len(statistics["ALL_H_WITH_R"][r])

    print('getting data statistics done!')

    return statistics


def save_data(data, filename, args):
    dirname = f'{args.dirname}/{args.save_dir}'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    filename = dirname + '/' + filename
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f'\nData saved as {filename}!')


def _corrupt_ent(positive_existing_ents, max_num, drug_ids):
    corrupted_ents = []
    while len(corrupted_ents) < max_num:
        candidates = random_num_gen.choice(drug_ids, (max_num - len(corrupted_ents)) * 2, replace=False)
        invalid_drug_ids = np.concatenate([positive_existing_ents, corrupted_ents], axis=0)
        mask = np.isin(candidates, invalid_drug_ids, assume_unique=True, invert=True)
        corrupted_ents.extend(candidates[mask])

    corrupted_ents = np.array(corrupted_ents)[:max_num]
    return corrupted_ents


def save_data_coldstart(root):
    drug_graph = read_pickle(os.path.join(root, 'drug_data_coldstart_pyg.pkl'))  # 以字典的形式呈现{'ID'：特征}
    drug_graph_dgl = read_pickle(os.path.join(root, 'drug_data_coldstart_dgl.pkl'))

    train_df = pd.read_csv(os.path.join(root, 'pair_pos_neg_triples-fold0-train.csv'))
    s1_df = pd.read_csv(os.path.join(root, 'pair_pos_neg_triples-fold0-s1.csv'))
    s2_df = pd.read_csv(os.path.join(root, 'pair_pos_neg_triples-fold0-s2.csv'))
    train_set = DrugDataset(train_df, drug_graph, drug_graph_dgl)
    s1_df_set = DrugDataset(s1_df, drug_graph, drug_graph_dgl)
    s2_df_set = DrugDataset(s2_df, drug_graph, drug_graph_dgl)
    # 保存 train_set 和 val_set 到 pkl 文件
    with open(os.path.join(root, f'pair_pos_neg_triples-fold0-train.pkl'), 'wb') as f:
        pickle.dump(train_set, f)

    with open(os.path.join(root, f'pair_pos_neg_triples-fold0-s1.pkl'), 'wb') as f:
        pickle.dump(s1_df_set, f)

    with open(os.path.join(root, f'pair_pos_neg_triples-fold0-s2.pkl'), 'wb') as f:
        pickle.dump(s2_df_set, f)


def _normal_batch(h, t, r, neg_size, data_statistics, drug_ids, desc):
    neg_size_h = 0
    neg_size_t = 0

    index = (0, 0) if desc == 's1' else (1, 1)
    if desc == 's2':
        index = (0, 1) if h in drug_ids[0] else (1, 0)

    prob = data_statistics["ALL_TAIL_PER_HEAD"][r] / (data_statistics["ALL_TAIL_PER_HEAD"][r] +
                                                      data_statistics["ALL_HEAD_PER_TAIL"][r])
    for i in range(neg_size):
        if random_num_gen.random() < prob:
            neg_size_h += 1
        else:
            neg_size_t += 1

    return (_corrupt_ent(data_statistics["ALL_TRUE_H_WITH_TR"][t, r], neg_size_h, drug_ids[index[0]]),
            _corrupt_ent(data_statistics["ALL_TRUE_T_WITH_HR"][h, r], neg_size_t, drug_ids[index[1]]))


import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-d', '--dataset', type=str, required=False, default='drugbank',
                    choices=['deepddi', 'drugbank', 'twosides'],
                    help='Dataset to preproces. Choose from (deepddi, drugbank, twosides)')
parser.add_argument('-n', '--neg_ent', type=int, default=1, help='Number of negative samples')
parser.add_argument('-s', '--seed', type=int, default=0,
                    help='Seed for the random number generator and used as fold number as well')
parser.add_argument('-n_d_r', '--new_drug_ratio', type=float, default=0.2)
parser.add_argument('-o', '--operation', type=str, required=False, default='all',
                    choices=['all', 'generate_triples', 'drug_data'], help='Operation to perform')
parser.add_argument('-s_d', '--save_dir', type=str, default='drugbank_coldstart')
dataset_columns_map = {
    'drugbank': ('d1', 'd2', 'smile1', 'smile2', 'type'),
    'twosides': ('Drug1_ID', 'Drug2_ID', 'Drug1', 'Drug2', 'New Y'),
    'zhangddi': ('drugbank_id_1', 'drugbank_id_2', 'smiles_2', 'smiles_1', 'label')
}

dataset_file_name_map = {
    'drugbank': ('drugbank.csv', '\t'),
    'twosides': ('twosides_ge_500.csv', ',')
}
args = parser.parse_args()
args.dataset = args.dataset.lower()

args.c_id1, args.c_id2, args.c_s1, args.c_s2, args.c_y = dataset_columns_map[args.dataset]
args.dataset_filename, args.delimiter = dataset_file_name_map[args.dataset]
# args.delimiter = dataset_file_name_map[args.dataset]
args.dataset_filename = 'Data/drugbank.csv'
args.dirname = 'Data'

random_num_gen = np.random.RandomState(args.seed)
# if args.operation in ('all', 'drug_data'):
#      load_drug_mol_dat(args)
#
if args.operation in ('all', 'generate_triples'):
    generate_pair_triples(args)

root_coldstart = f'{args.dirname}/drugbank_coldstart'
save_data_coldstart(root_coldstart)
