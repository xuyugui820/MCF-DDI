import pickle
import pandas as pd
import torch.optim as optim
import numpy as np
import torch.nn as nn
import os
# from dataset import load_ddi_dataset, DrugDataLoader
from dataset import DrugDataLoader
from logger.train_logger import TrainLogger
# from .logger import train_logger
from data_pre.data_pre import CustomData
import argparse
from metrics import *
from utils import *
from tqdm import tqdm
import warnings
from model import gnn_model
warnings.filterwarnings("ignore")
from model import FingerprintNet
#
# import random
# random.seed(10)  # 选择一个固定的随机种子
# np.random.seed(10)
# torch.manual_seed(10)  # CPU 上的随机种子
import random
import torch
random.seed(312)
np.random.seed(312)
torch.manual_seed(312)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(312)

def val_SRR(SRR, criterion, dataloader, device, epoch):
    SRR.eval()
    running_loss = AverageMeter()

    pred_list = []
    label_list = []

    for data in tqdm(dataloader,desc='val_epoch_{}'.format(epoch),leave=True):
        # head_pairs, tail_pairs, head_pairs_dgl, tail_pairs_dgl, rel, label = [d.to(device) for d in data]
        head_pairs, tail_pairs, head_pairs_dgl, tail_pairs_dgl, head_finger,tail_finger,rel, label=[d.to(device) for d in data]
        sim_h = head_pairs.sim
        sim_t = tail_pairs.sim
        sim_dt_h = head_pairs.drug_target_sim
        sim_dt_t = tail_pairs.drug_target_sim
        # 使用分子指纹时，改变通道数
        dim = head_finger.shape[0]
        head_finger = np.reshape(head_finger.cpu(), (dim, 1, 881)).to(device)
        tail_finger = np.reshape(tail_finger.cpu(), (dim, 1, 881)).to(device)

        batch_h_e = head_pairs_dgl.edata['feat']
        batch_t_e = tail_pairs_dgl.edata['feat']
        with torch.no_grad():
            pred = SRR.forward(head_pairs, tail_pairs, head_pairs_dgl, tail_pairs_dgl, batch_h_e, batch_t_e, head_finger, tail_finger, rel, sim_h, sim_t,sim_dt_h,sim_dt_t)
            loss = criterion(pred, label)
            pred_cls = torch.sigmoid(pred)
            pred_list.append(pred_cls.view(-1).detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())
            running_loss.update(loss.item(), label.size(0))

    pred_probs = np.concatenate(pred_list, axis=0)
    label = np.concatenate(label_list, axis=0)

    acc, auroc, f1_score, precision, recall, ap = do_compute_metrics(pred_probs, label)

    epoch_loss = running_loss.get_average()
    running_loss.reset()

    SRR.train()

    return epoch_loss, acc, auroc, f1_score, precision, recall, ap

def val_finger(FingerModel, criterion, dataloader, device, epoch):
    FingerModel.eval()
    running_loss = AverageMeter()

    pred_list = []
    label_list = []

    for data in tqdm(dataloader,desc='val_epoch_{}'.format(epoch),leave=True):
        # head_pairs, tail_pairs, head_pairs_dgl, tail_pairs_dgl, rel, label = [d.to(device) for d in data]
        # head_pairs, tail_pairs, head_pairs_dgl, tail_pairs_dgl, head_finger,tail_finger,rel, label=[d.to(device) for d in data]
        # sim_h = head_pairs.sim
        # sim_t = tail_pairs.sim
        #dim = data[4].shape[0]
        # head_finger = np.reshape(head_finger.cpu(), (dim, 1, 881)).to(device)
        # tail_finger = np.reshape(tail_finger.cpu(), (dim, 1, 881)).to(device)
        # batch_h_e = head_pairs_dgl.edata['feat']
        # batch_t_e = tail_pairs_dgl.edata['feat']

        #======================================================================
        head_pairs, tail_pairs, head_pairs_dgl, tail_pairs_dgl, head_finger, tail_finger, rel, label = [d.to(device) for
                                                                                                        d in data]
        # head_finger_cpu = head_finger.cpu()
        # tail_finger_cpu = tail_finger.cpu()
        # head_finger_cpu = create_mlp(881, 881, head_finger_cpu)
        # tail_finger_cpu = create_mlp(881, 881, head_finger_cpu)
        dim = head_finger.shape[0]
        head_finger = np.reshape(head_finger.cpu(), (dim, 1, 881)).to(device)
        tail_finger = np.reshape(tail_finger.cpu(), (dim, 1, 881)).to(device)
        # print(rel.cpu().unique());exit()
        sim_h = head_pairs.sim
        sim_t = tail_pairs.sim

        # head_finger_cpu = head_finger_cpu.detach().cpu()
        # tail_finger_cpu = tail_finger_cpu.detach().cpu()
        # head_finger_cpu = np.reshape(head_finger_cpu, (dim, 1, 881)).to(device)
        # tail_finger_cpu = np.reshape(tail_finger_cpu, (dim, 1, 881)).to(device)
        batch_h_e = head_pairs_dgl.edata['feat']
        batch_t_e = tail_pairs_dgl.edata['feat']

        with torch.no_grad():
            pred = FingerModel.forward(head_pairs, tail_pairs, head_pairs_dgl, tail_pairs_dgl, batch_h_e, batch_t_e, head_finger, tail_finger, rel, sim_h, sim_t)
            loss = criterion(pred, label)
            pred_cls = torch.sigmoid(pred)
            #pred_cls = pred
            pred_list.append(pred_cls.view(-1).detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())
            running_loss.update(loss.item(), label.size(0))
            has_nan = np.isnan(pred_cls).any()
            # if has_nan:
            #     print("pred_cls（NaN）")
            #     print(pred_cls)
    pred_probs = np.concatenate(pred_list, axis=0)

    label = np.concatenate(label_list, axis=0)

    acc, auroc, f1_score, precision, recall, ap = do_compute_metrics(pred_probs, label)

    epoch_loss = running_loss.get_average()
    running_loss.reset()

    FingerModel.train()

    return epoch_loss, acc, auroc, f1_score, precision, recall, ap


def create_mlp(input_dim, output_dim, data_tensor):
    # 定义 MLP
    mlp = nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.PReLU(),
        nn.Linear(output_dim, output_dim),
        nn.BatchNorm1d(output_dim),
        nn.PReLU(),
        nn.Linear(output_dim, output_dim),
        nn.BatchNorm1d(output_dim),
    )

    # 在 MLP 中运行输入数据
    output = mlp(data_tensor)

    # 将输出张量转换为 torch.FloatTensor 类型
    output = output.float()

    return output


def main():
    parser = argparse.ArgumentParser()

    # Add argument
    #parser.add_argument('--n_iter', type=int, default=10, help='number of GNN')
    parser.add_argument('--L', type=int, default=3, help='number of Graph Transformer')
    parser.add_argument('--fold', type=int, default=0, help='[0, 1, 2]')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    #parser.add_argument('--weight_decay', type=float, default=5e-4, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--save_model', action='store_true', help='whether save model or not')
    #parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--cold_or_hot',type=str,default='hot',choices=['hot','cold'])
    parser.add_argument('--nn_or_no',default='nn',type=str,choices=['nn','no'])


    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--n_iter', type=int, default=10, help='number of GNN')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='number of epochs')
    args = parser.parse_args()

    params = dict(
        model='MCF-DDI',
        data_root='Data',
        save_dir='save',
        dataset='drugbank_coldstart' if args.cold_or_hot == 'cold' else 'drugbank',
        epochs=args.epochs,
        fold=args.fold,
        save_model=args.save_model,
        lr=args.lr,
        batch_size=args.batch_size,
        n_iter=args.n_iter,
        weight_decay=args.weight_decay,
        L = args.L
    )

    logger = TrainLogger(params)
    logger.info(__file__)

    save_model = params.get('save_model')
    batch_size = params.get('batch_size')
    dataset = params.get('dataset')
    data_root = params.get('data_root')
    #data_set = params.get('dataset')
    fold = params.get('fold')
    epochs = params.get('epochs')
    n_iter = params.get('n_iter')
    lr = params.get('lr')
    L = params.get('L')
    weight_decay = params.get('weight_decay')
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    # device = "cpu"
    MODEL_NAME = 'MCF-DDI'
    net_params = dict(
        L=3,
        n_heads=8,
        hidden_dim=128,
        out_dim=128,
        edge_feat=True,
        residual=True,
        readout="mean",
        in_feat_dropout=0.0,
        dropout=0.0,
        layer_norm=False,
        batch_norm=True,
        self_loop=False,
        lap_pos_enc=True,
        pos_enc_dim=6,
        full_graph=False,
        batch_size=1,
        # num_atom_type=node_dim,
        # num_bond_type=edge_dim,
        num_atom_type=70,
        num_bond_type=6,
        device=device,
        n_iter=n_iter
    )

    if args.cold_or_hot == 'cold':
        data_path = os.path.join(data_root, dataset)
        # train_loader, val_loader = load_ddi_dataset(root=data_path, batch_size=batch_size, fold=fold,cold_or_hot=args.cold_or_hot,nn_or_no=args.nn_or_no)
        with open(os.path.join(data_path, f'pair_pos_neg_triples-fold0-train.pkl'), 'rb') as f:
            train_set = pickle.load(f)
        # #new-new
        # with open(os.path.join(data_path, f'pair_pos_neg_triples-fold0-s1.pkl'), 'rb') as f:
        #     val_set = pickle.load(f)
        #     print("begin new-new experiment")
        #new-old
        with open(os.path.join(data_path, f'pair_pos_neg_triples-fold0-s2.pkl'), 'rb') as f:
            val_set = pickle.load(f)
            print("begin new-old experiment")
        train_loader = DrugDataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=16,
                                      drop_last=True)
        val_loader = DrugDataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=16)
        print("Number of samples in the train set: ", len(train_set))
        print("Number of samples in the cal set: ", len(val_set))
        data = next(iter(train_loader))
        node_dim = data[0].x.size(-1)
        edge_dim = data[0].edge_attr.size(-1)
        net_params['num_atom_type'] = node_dim
        net_params['num_bond_type'] = edge_dim
        print("start with cold start fold_{}".format(fold) + " transformer have {} layers".format(L),
              "sub extract have {} layers".format(n_iter))
        Model = gnn_model(MODEL_NAME, net_params).to(device)

        # #linear
        optimizer = optim.Adam(Model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.99 ** (epoch))
        #===================================================================
        #knn
        # optimizer = optim.AdamW(Model.parameters(), lr=1e-3, weight_decay=1e-5)
        # # # 定义学习率调度器
        # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
        #==========================================================

        criterion = nn.BCEWithLogitsLoss().to(device)
        # criterion = nn.BCELoss().to(device)


        running_loss = AverageMeter()
        running_acc = AverageMeter()

        Model.train()
        for epoch in range(epochs):
            # 返回每个原子特征x（33，70）、边对edge_index（2，74）,有边相连矩阵中坐标line_graph_edge_index（2，104）、边特征edge_attr=[74,6]、相似矩阵（1，544）,id='DB09053'
            for data in tqdm(train_loader, desc='train_loader_epoch_{}'.format(epoch), leave=True):
                head_pairs, tail_pairs, head_pairs_dgl, tail_pairs_dgl, head_finger, tail_finger, rel, label = [
                    d.to(device) for d in data]

                # 使用分子指纹时，改变通道数
                dim = head_finger.shape[0]
                head_finger = np.reshape(head_finger.cpu(), (dim, 1, 881)).to(device)
                tail_finger = np.reshape(tail_finger.cpu(), (dim, 1, 881)).to(device)

                # print(rel.cpu().unique());exit()
                sim_h = head_pairs.sim
                sim_t = tail_pairs.sim
                sim_dt_h = head_pairs.drug_target_sim
                sim_dt_t = tail_pairs.drug_target_sim
                batch_h_e = head_pairs_dgl.edata['feat']
                batch_t_e = tail_pairs_dgl.edata['feat']
                # pred = SRR.forward(head_pairs, tail_pairs, head_pairs_dgl, tail_pairs_dgl,batch_h_e, batch_t_e,head_finger,tail_finger, rel, sim_h, sim_t)
                pred = Model(head_pairs, tail_pairs, head_pairs_dgl, tail_pairs_dgl, batch_h_e, batch_t_e, head_finger,
                             tail_finger, rel, sim_h, sim_t,sim_dt_h,sim_dt_t)
                loss = criterion(pred, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # pred_cls = (pred > 0.5).detach().cpu().numpy()
                pred_cls = (torch.sigmoid(pred) > 0.5).detach().cpu().numpy()
                acc = accuracy(label.detach().cpu().numpy(), pred_cls)
                running_acc.update(acc)
                running_loss.update(loss.item(), label.size(0))

            epoch_loss = running_loss.get_average()
            epoch_acc = running_acc.get_average()
            running_loss.reset()
            running_acc.reset()

            val_loss, val_acc, val_auroc, val_f1_score, val_precision, val_recall, val_ap = val_SRR(Model, criterion,
                                                                                                    val_loader,
                                                                                                    device, epoch)
            val_msg = "epoch-%d, train_loss-%.4f, train_acc-%.4f, val_loss-%.4f, val_acc-%.4f, val_auroc-%.4f, val_f1_score-%.4f, val_prec-%.4f, val_rec-%.4f, val_ap-%.4f" % (
                epoch, epoch_loss, epoch_acc, val_loss, val_acc, val_auroc, val_f1_score, val_precision, val_recall,
                val_ap)

            # test_loss, test_acc, test_auroc, test_f1_score, test_precision, test_recall, test_ap =  val_SRR(SRR, criterion,
            #                                                                                            test_loader, device,
            #                                                                                            epoch)
            # test_msg = "epoch-%d, train_loss-%.4f, train_acc-%.4f, test_loss-%.4f, test_acc-%.4f, test_auroc-%.4f, test_f1_score-%.4f, test_prec-%.4f, test_rec-%.4f, test_ap-%.4f" % (
            #     epoch, epoch_loss, epoch_acc, test_loss, test_acc, test_auroc, test_f1_score, test_precision, test_recall,
            #     test_ap)
            # val_loss, val_acc, val_auroc, val_f1_score, val_precision, val_recall, val_ap = val_finger(Model, criterion, val_loader,
            #                                                                                     device, epoch)
            # val_msg = "epoch-%d, train_loss-%.4f, train_acc-%.4f, val_loss-%.4f, val_acc-%.4f, val_auroc-%.4f, val_f1_score-%.4f, val_prec-%.4f, val_rec-%.4f, val_ap-%.4f" % (
            #     epoch, epoch_loss, epoch_acc, val_loss, val_acc, val_auroc, val_f1_score, val_precision, val_recall, val_ap)

            print("========================================================")
            logger.info(val_msg)
            print("========================================================")
            scheduler.step()
            if save_model:
                msg = "epoch-%d, train_loss-%.4f, train_acc-%.4f, val_loss-%.4f, val_acc-%.4f" % (
                epoch, epoch_loss, epoch_acc, val_loss, val_acc)
                save_model_dict(Model, logger.get_model_dir(), msg)
            # 测试集
        # test_loss, test_acc, test_auroc, test_f1_score, test_precision, test_recall, test_ap = val_finger(FingerModel, criterion,test_loader, device,epoch)
        # test_msg = "test_loss-%.4f, test_acc-%.4f, test_auroc-%.4f, test_f1_score-%.4f, test_prec-%.4f, test_rec-%.4f, test_ap-%.4f" % (
        # test_loss, test_acc, test_auroc, test_f1_score, test_precision, test_recall,test_ap)
        # print("========================================================")
        # logger.info(test_msg)
        # print("========================================================")
    else:

        for i in range(5):
            i+=1
            data_path = os.path.join(data_root, dataset)
            # train_loader, val_loader = load_ddi_dataset(root=data_path, batch_size=batch_size, fold=fold,cold_or_hot=args.cold_or_hot,nn_or_no=args.nn_or_no)
            with open(os.path.join(data_path, f'train_set_fold_{i}.pkl'), 'rb') as f:
                train_set = pickle.load(f)
            with open(os.path.join(data_path, f'val_set_fold_{i}.pkl'), 'rb') as f:
                val_set = pickle.load(f)
            train_loader = DrugDataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8,
                                          drop_last=True)
            val_loader = DrugDataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8)
            data = next(iter(train_loader))
            node_dim = data[0].x.size(-1)
            edge_dim = data[0].edge_attr.size(-1)
            net_params['num_atom_type'] = node_dim
            net_params['num_bond_type'] = edge_dim
            print("Number of samples in the validation set: ", len(train_set))
            print("Number of samples in the test set: ", len(val_set))
            print("=================================================================================")
            print("start with hot start fold_{}".format(i) + " transformer have {} layers".format(L),
                  "sub extract have {} layers".format(n_iter))
            Model = gnn_model(MODEL_NAME, net_params).to(device)

            optimizer = optim.Adam(Model.parameters(), lr=lr, weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.99 ** (epoch))
            # optimizer = optim.Adam(SRR.parameters(), lr=lr, weight_decay=weight_decay)

            criterion = nn.BCEWithLogitsLoss().to(device)

            # KAN==============
            # optimizer = optim.AdamW(Model.parameters(), lr=lr, weight_decay=weight_decay)
            # # # 定义学习率调度器
            # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)
            #==============================
            running_loss = AverageMeter()
            running_acc = AverageMeter()

            Model.train()
            for epoch in range(epochs):
                # 返回每个原子特征x（33，70）、边对edge_index（2，74）,有边相连矩阵中坐标line_graph_edge_index（2，104）、边特征edge_attr=[74,6]、相似矩阵（1，544）,id='DB09053'
                for data in tqdm(train_loader,desc='train_loader_epoch_{}'.format(epoch),leave=True):

                    head_pairs, tail_pairs, head_pairs_dgl, tail_pairs_dgl, head_finger,tail_finger,rel, label = [d.to(device) for d in data]

                    #使用分子指纹时，改变通道数
                    dim = head_finger.shape[0]
                    head_finger = np.reshape(head_finger.cpu(), (dim, 1, 881)).to(device)
                    tail_finger = np.reshape(tail_finger.cpu(), (dim, 1, 881)).to(device)


                    # print(rel.cpu().unique());exit()
                    sim_h = head_pairs.sim
                    sim_t = tail_pairs.sim
                    sim_dt_h = head_pairs.drug_target_sim
                    sim_dt_t = tail_pairs.drug_target_sim
                    batch_h_e = head_pairs_dgl.edata['feat']
                    batch_t_e = tail_pairs_dgl.edata['feat']
                    # pred = SRR.forward(head_pairs, tail_pairs, head_pairs_dgl, tail_pairs_dgl,batch_h_e, batch_t_e,head_finger,tail_finger, rel, sim_h, sim_t)
                    pred = Model(head_pairs, tail_pairs, head_pairs_dgl, tail_pairs_dgl, batch_h_e, batch_t_e,
                                 head_finger,
                                 tail_finger, rel, sim_h, sim_t, sim_dt_h, sim_dt_t)
                    loss = criterion(pred, label)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    #pred_cls = (pred > 0.5).detach().cpu().numpy()
                    pred_cls = (torch.sigmoid(pred) > 0.5).detach().cpu().numpy()
                    acc = accuracy(label.detach().cpu().numpy(), pred_cls)
                    running_acc.update(acc)
                    running_loss.update(loss.item(), label.size(0))

                epoch_loss = running_loss.get_average()
                epoch_acc = running_acc.get_average()
                running_loss.reset()
                running_acc.reset()

                val_loss, val_acc, val_auroc, val_f1_score, val_precision, val_recall, val_ap = val_SRR(Model, criterion, val_loader,
                                                                                                    device, epoch)
                val_msg = "the-%d-fold,epoch-%d, train_loss-%.4f, train_acc-%.4f, val_loss-%.4f, val_acc-%.4f, val_auroc-%.4f, val_f1_score-%.4f, val_prec-%.4f, val_rec-%.4f, val_ap-%.4f" % (
                i,epoch, epoch_loss, epoch_acc, val_loss, val_acc, val_auroc, val_f1_score, val_precision, val_recall, val_ap)
                # test_loss, test_acc, test_auroc, test_f1_score, test_precision, test_recall, test_ap =  val_SRR(SRR, criterion,
                #                                                                                            test_loader, device,
                #                                                                                            epoch)
                # test_msg = "epoch-%d, train_loss-%.4f, train_acc-%.4f, test_loss-%.4f, test_acc-%.4f, test_auroc-%.4f, test_f1_score-%.4f, test_prec-%.4f, test_rec-%.4f, test_ap-%.4f" % (
                #     epoch, epoch_loss, epoch_acc, test_loss, test_acc, test_auroc, test_f1_score, test_precision, test_recall,
                #     test_ap)
                # val_loss, val_acc, val_auroc, val_f1_score, val_precision, val_recall, val_ap = val_finger(Model, criterion, val_loader,
                #                                                                                     device, epoch)
                # val_msg = "epoch-%d, train_loss-%.4f, train_acc-%.4f, val_loss-%.4f, val_acc-%.4f, val_auroc-%.4f, val_f1_score-%.4f, val_prec-%.4f, val_rec-%.4f, val_ap-%.4f" % (
                #     epoch, epoch_loss, epoch_acc, val_loss, val_acc, val_auroc, val_f1_score, val_precision, val_recall, val_ap)



                print("========================================================")
                logger.info(val_msg)
                print("========================================================")
                scheduler.step()
                # if save_model:
                #     msg = "epoch-%d, train_loss-%.4f, train_acc-%.4f, val_loss-%.4f, val_acc-%.4f" % (
                #     epoch, epoch_loss, epoch_acc, val_loss, val_acc)
                #     save_model_dict(SRR, logger.get_model_dir(), msg)
                    #测试集
            # test_loss, test_acc, test_auroc, test_f1_score, test_precision, test_recall, test_ap = val_finger(FingerModel, criterion,test_loader, device,epoch)
            # test_msg = "test_loss-%.4f, test_acc-%.4f, test_auroc-%.4f, test_f1_score-%.4f, test_prec-%.4f, test_rec-%.4f, test_ap-%.4f" % (
            # test_loss, test_acc, test_auroc, test_f1_score, test_precision, test_recall,test_ap)
            # print("========================================================")
            # logger.info(test_msg)
            # print("========================================================")

if __name__ == "__main__":
    main()




