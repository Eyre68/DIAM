import timeit
import argparse
import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as fn
from data_preprocess import *
from hetero_view.graph_builder import create_dgl_heterograph
from pos_contrast import build_relation_graphs, find_positive_pairs, feature_cos
import pandas as pd
import os
from model1 import DIAM
from metric import *
import warnings

warnings.filterwarnings('ignore')
device = torch.device('cuda')

import random
torch.manual_seed(123)
random.seed(123)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--k_fold', type=int, default=10, help='k-fold cross validation')
    parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight_decay')
    parser.add_argument('--random_seed', type=int, default=1234, help='random seed')
    parser.add_argument('--neighbor', type=int, default=5, help='neighbor')
    parser.add_argument('--negative_rate', type=float, default=1.0, help='negative_rate')
    parser.add_argument('--dataset', default='C-dataset', help='dataset')
    parser.add_argument('--dropout', default='0.2', type=float, help='dropout of HGT')
    parser.add_argument('--n', default='2', type=int, help='n power of the adjacency matrix')
    parser.add_argument('--p_drug', default='30', type=int, help='threshold of drug node degree')
    parser.add_argument('--p_disease', default='15', type=int, help='threshold of disease node degree')
    parser.add_argument('--t', default='0.5', type=float, help='threshold of feature similarity')
    parser.add_argument('--pos_num', default='5', type=int, help='threshold of positive samples selection')
    parser.add_argument('--hgt_layer', default='1', type=int, help='heterogeneous graph transformer layer')
    parser.add_argument('--hgt_head', default='2', type=int, help='heterogeneous graph transformer head')
    parser.add_argument('--embedding_dim', default='512', type=int, help='GAE embedding dimension')
    parser.add_argument('--hgt_in_dim', default='64', type=int, help='heterogeneous graph transformer input dimension')
    parser.add_argument('--hgt_head_dim', default='32', type=int, help='heterogeneous graph transformer head dimension')
    parser.add_argument('--hgt_out_dim', default='128', type=int, help='heterogeneous graph transformer output dimension')
    parser.add_argument('--tau_cross', default='0.8', type=float, help='tau in cross-view contrastive loss')
    parser.add_argument('--tau_hgt', default='0.8', type=float, help='tau_mp in intra-view contrastive loss')
    parser.add_argument('--tau_hgcn', default='0.8', type=float, help='tau_sc in intra-view contrastive loss')
    parser.add_argument('--lam_cross', default='0.5', type=float, help='lam in cross-view contrastive loss')
    parser.add_argument('--lam_intra', default='0.5', type=float, help='lam in intra-view contrastive loss')
    parser.add_argument('--loss_rate', default='0.5', type=float, help='loss rate of unsupervised learning and training')
    parser.add_argument('--n_neg', default='20', type=int, help='number of negative candidates')
    parser.add_argument('--pool', default='mean', help='method of pooling in negative sampling')
    parser.add_argument('--alpha', type=float, default=0.3, help='weight for direct path scores')
    parser.add_argument('--beta', type=float, default=0.5, help='weight for indirect path scores')
    parser.add_argument('--gamma', type=float, default=0.5, help='non-linear enhancement factor')
    parser.add_argument('--neg_threshold', type=float, default=90.0)
    parser.add_argument('--early_stop_patience', type=int, default=100, help='patience for early stopping')

    args = parser.parse_args()
    args.data_dir = 'data/' + args.dataset + '/'

    data = get_data(args)
    args.drug_number = data['drug_number']
    args.disease_number = data['disease_number']
    args.protein_number = data['protein_number']
    drug_feature_dim = data['drug_gae'].shape[1]
    disease_feature_dim = data['disease_gae'].shape[1]
    protein_feature_dim = data['protein_gae'].shape[1]

    data = data_processing(data, args)
    data = k_fold(data, args)

    drdr_graph, didi_graph, prpr_graph, data = dgl_similarity_graph(data, args)

    device = torch.device('cpu')

    drdr_graph = drdr_graph.to(device)
    didi_graph = didi_graph.to(device)
    drug_feature = torch.FloatTensor(data['drug_gae']).to(device)
    disease_feature = torch.FloatTensor(data['disease_gae']).to(device)
    protein_feature = torch.FloatTensor(data['protein_gae']).to(device)

    all_sample = torch.tensor(data['all_drdi']).long()

    start = timeit.default_timer()

    cross_entropy = nn.CrossEntropyLoss()

    Metric = ('Epoch\t\tTime\t\tLoss_con\t\tLoss_cls\t\tAUC\t\tAUPR\t\tAccuracy\t\tPrecision\t\tRecall\t\tF1-score\t\tMcc')
    AUCs, AUPRs = [], []

    precisions,recalls,accuracys,f1_scores=[],[],[],[]

    print('Dataset:', args.dataset)

    for i in range(args.k_fold):

        print('fold:', i)
        print(Metric)

        model = DIAM(args, drug_feature_dim, disease_feature_dim, protein_feature_dim, use_mechanism_weights=True)
        model = model.to(device)

        train_associations = data['X_train_p'][i]
        topo_scores = model.compute_topo_path_scores(data, train_associations)
        data['topo_path_scores'] = topo_scores

        data = model.compute_target_path_scores(data)
        data = model.compute_mechanism_weights(data)

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='max', factor=0.8, patience=100, verbose=True, min_lr=1e-3)

        text_file = "./data/{}/fold/{}/data_train.csv".format(args.dataset,i)
        train_samples = np.array(pd.read_csv(text_file).iloc[:, 1:])
        train_X=train_samples[:,0:2]
        train_X_pos = [row[:-1] for row in train_samples if row[-1] == 1]
        train_X_neg=[row[:-1] for row in train_samples if row[-1] == 0]
        train_Y=train_samples[:,-1]

        text_file1 = "./data/{}/fold/{}/data_test.csv".format(args.dataset,i)
        test_samples = np.array(pd.read_csv(text_file1).iloc[:, 1:])
        test_X = test_samples[:, 0:2]
        test_X_pos = [row[:-1] for row in test_samples if row[-1] == 1]
        test_X_neg = [row[:-1] for row in test_samples if row[-1] == 0]
        test_Y = test_samples[:, -1]

        data['X_train'][i]=train_X
        data['X_train_p'][i]= np.array([item.tolist() for item in train_X_pos])
        data['X_train_n'][i]= np.array([item.tolist() for item in train_X_neg])
        data['Y_train'][i]=train_Y

        data['X_test'][i]=test_X
        data['X_test_p'][i]= np.array([item.tolist() for item in test_X_pos])
        data['X_test_n'][i]= np.array([item.tolist() for item in test_X_neg])
        data['Y_test'][i]=test_Y

        best_auc, best_aupr, best_accuracy, best_precision, best_recall, best_f1, best_mcc = 0, 0, 0, 0, 0, 0, 0
        X_train = torch.LongTensor(data['X_train'][i]).to(device)
        X_train_p = torch.LongTensor(data['X_train_p'][i]).to(device)
        X_train_n = torch.LongTensor(data['X_train_n'][i]).to(device)
        Y_train = torch.LongTensor(data['Y_train'][i]).to(device)
        X_test = torch.LongTensor(data['X_test'][i]).to(device)
        X_test_p = torch.LongTensor(data['X_test_p'][i]).to(device)
        X_test_n = torch.LongTensor(data['X_test_n'][i]).to(device)
        Y_test = data['Y_test'][i].flatten()

        drdipr_graph, meta_paths, edge_types, data = dgl_heterograph(data, data['X_train_p'][i], args)

        feature_graph_drug = feature_cos(data, 'drug', args.t)
        feature_graph_disease = feature_cos(data, 'disease', args.t)
        g_m_drug, g_m_disease = create_dgl_heterograph(drdipr_graph, edge_types, args, data)

        rdr, drd = build_relation_graphs(data['X_train_p'][i], data, args)
        pos_r, pos_d = find_positive_pairs(rdr, drd, feature_graph_drug, feature_graph_disease, args)

        drdipr_graph = drdipr_graph.to(device)
        g_m_drug = g_m_drug.to(device)
        g_m_disease = g_m_disease.to(device)

        neg_candidates_train_r = []
        for k in range(data['drug_number']):
            neg_k = np.where(data['Y_train'][i] == 0)[0]
            neg_k = data['X_train'][i][neg_k][:, 1]
            neg_candidates_train_r.append(neg_k)

        neg_candidates_train_d = []
        for k in range(data['disease_number']):
            neg_k = np.where(data['Y_train'][i] == 0)[0]
            neg_k = data['X_train'][i][neg_k][:, 0]
            neg_candidates_train_d.append(neg_k)

        neg_candidates_test_r = []
        for k in range(data['drug_number']):
            neg_k = np.where(data['Y_test'][i] == 0)[0]
            neg_k = data['X_test'][i][neg_k][:, 1]
            neg_candidates_test_r.append(neg_k)

        neg_candidates_test_d = []
        for k in range(data['disease_number']):
            neg_k = np.where(data['Y_test'][i] == 0)[0]
            neg_k = data['X_test'][i][neg_k][:, 0]
            neg_candidates_test_d.append(neg_k)

        pos_r = torch.IntTensor(pos_r.toarray()).to(device)
        pos_d = torch.IntTensor(pos_d.toarray()).to(device)

        for epoch in range(args.epochs):
            model.train()
            l_con, train_output, _, _, beta_values = model(g_m_drug, g_m_disease, pos_r, pos_d,
                                                           drug_feature, disease_feature, protein_feature, X_train_p,
                                                           X_train_n,
                                                           neg_candidates_train_r, neg_candidates_train_d,
                                                           data['X_train_n'][i], data, return_beta=True)

            l_train = cross_entropy(train_output, torch.flatten(Y_train))
            l_all = args.loss_rate * l_con + (1 - args.loss_rate) * l_train

            optimizer.zero_grad()
            l_all.backward()
            optimizer.step()

            l_con = l_con.detach().cpu().numpy()
            l_train = l_train.detach().cpu().numpy()

            with torch.no_grad():
                model.eval()
                _, test_score, _, _ = model(g_m_drug, g_m_disease,
                                            pos_r, pos_d, drug_feature, disease_feature, protein_feature, X_test_p,
                                            X_test_n, neg_candidates_test_r, neg_candidates_test_d,data['X_test_n'][i],data)


            test_prob = fn.softmax(test_score, dim=-1)
            test_score = torch.argmax(test_score, dim=-1)

            predict_result=test_prob
            predict_score=np.zeros((len(test_score),1))

            test_prob = test_prob[:, 1]
            test_prob = test_prob.cpu().numpy()
            test_score = test_score.cpu().numpy()

            AUC, AUPR, accuracy, precision, recall, f1, mcc = get_metric(Y_test, test_score, test_prob)

            scheduler.step(AUC)

            end = timeit.default_timer()
            time = end - start
            show = [epoch + 1, round(time, 2), l_con, l_train, round(AUC, 5), round(AUPR, 5), round(accuracy, 5),
                       round(precision, 5), round(recall, 5), round(f1, 5), round(mcc, 5)]
            print('\t\t'.join(map(str, show)))
            if AUC > best_auc:
                best_epoch = epoch + 1
                best_auc = AUC
                best_aupr, best_accuracy, best_precision, best_recall, best_f1, best_mcc = AUPR, accuracy, precision, recall, f1, mcc
                print('AUC improved at epoch ', best_epoch, ';\tbest_auc:', best_auc)
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= args.early_stop_patience:
                    print(
                        f'Early stopping at epoch {epoch + 1} as AUC did not improve for {args.early_stop_patience} epochs.')
                    break
        AUCs.append(best_auc)
        AUPRs.append(best_aupr)
        accuracys.append(best_accuracy)
        precisions.append(best_precision)
        recalls.append(best_recall)
        f1_scores.append(best_f1)


    print("***************************************************************")
    print('AUC:', AUCs)
    AUC_mean = np.mean(AUCs)
    AUC_std = np.std(AUCs)
    print('Mean AUC:', AUC_mean, '(', AUC_std, ')')

    print("***************************************************************")
    print('AUPR:', AUPRs)
    AUPR_mean = np.mean(AUPRs)
    AUPR_std = np.std(AUPRs)
    print('Mean AUPR:', AUPR_mean, '(', AUPR_std, ')')

    print("***************************************************************")
    print('accuracy:', accuracys)
    accuracy_mean = np.mean(accuracys)
    accuracy_std = np.std(accuracys)
    print('Mean accuracy:', accuracy_mean, '(', accuracy_std, ')')

    print("***************************************************************")
    print('precision:', precisions)
    precision_mean = np.mean(precisions)
    precision_std = np.std(precisions)
    print('Mean precision:', precision_mean, '(', precision_std, ')')

    print("***************************************************************")
    print('recall:', recalls)
    recall_mean = np.mean(recalls)
    recall_std = np.std(recalls)
    print('Mean recall:', recall_mean, '(', recall_std, ')')

    print("***************************************************************")
    print('f1:', f1_scores)
    f1_mean = np.mean(f1_scores)
    f1_std = np.std(f1_scores)
    print('Mean f1:', f1_mean, '(', f1_std, ')')


    results = {
        'AUC': AUCs,
        'AUPR': AUPRs,
        'accuracy': accuracys,
        'precision': precisions,
        'recall': recalls,
        'f1': f1_scores
    }
    df = pd.DataFrame(results)
    df.loc['Mean'] = df.mean()
    df.loc['Var'] = df.var()
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"results.xlsx")
    print("Saving results to", output_file)
    print("DataFrame content:\n", df)

    try:
        df.to_excel(output_file, index_label='')
        print("File saved successfully")
    except Exception as e:
        print(f"Error saving file: {e}")
