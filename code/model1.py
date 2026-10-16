import torch
import torch.nn as nn
import torch.nn.functional as F
from Contrast import Contrast
from hetero_view.hgtencoder import HGTEncoder
import numpy as np
import math
from torch.nn.parameter import Parameter

import random
torch.manual_seed(123)
random.seed(123)


def dynamic_knn(similarity_matrix, k_min=5, k_max=15):
    n = similarity_matrix.shape[0]
    dynamic_k_values = np.zeros(n, dtype=int)
    for i in range(n):
        similarities = similarity_matrix[i]
        mean_sim = np.mean(similarities)
        k_value = int(np.interp(mean_sim, (0, 1), (k_max, k_min)))
        k_value = max(k_min, min(k_value, k_max))

        dynamic_k_values[i] = k_value

    return dynamic_k_values

def build_hypergraph_matrix(similarity_matrix, dynamic_k_values):
    n = similarity_matrix.shape[0]
    hypergraph_matrix = np.zeros((n, n), dtype=int)

    for i in range(n):
        k_value = dynamic_k_values[i]

        top_k_indices = np.argsort(similarity_matrix[i])[-k_value-1:-1][::-1]

        hypergraph_matrix[i, i] = 1
        for neighbor in top_k_indices:
            hypergraph_matrix[neighbor, i] = 1

    return hypergraph_matrix


def Eu_dis(x):
    x=x
    x = np.mat(x)
    aa = np.sum(np.multiply(x, x), 1)
    ab = x * x.T
    dist_mat = aa + aa.T - 2 * ab
    dist_mat[dist_mat < 0] = 0
    dist_mat = np.sqrt(dist_mat)
    dist_mat = np.maximum(dist_mat, dist_mat.T)
    return dist_mat


def feature_concat(*F_list, normal_col=False):
    features = None
    for f in F_list:
        if f is not None and f != []:
            # deal with the dimension that more than two
            if len(f.shape) > 2:
                f = f.reshape(-1, f.shape[-1])
            # normal each column
            if normal_col:
                f_max = np.max(np.abs(f), axis=0)
                f = f / f_max
            # facing the first feature matrix appended to fused feature matrix
            if features is None:
                features = f
            else:
                features = np.hstack((features, f))
    if normal_col:
        features_max = np.max(np.abs(features), axis=0)
        features = features / features_max
    return features


def hyperedge_concat(*H_list):
    H = None
    for h in H_list:
        if h is not None and h != []:
            # for the first H appended to fused hypergraph incidence matrix
            if H is None:
                H = h
            else:
                if type(h) != list:
                    H = np.hstack((H, h))
                else:
                    tmp = []
                    for a, b in zip(H, h):
                        tmp.append(np.hstack((a, b)))
                    H = tmp
    return H


def generate_G_from_H(H, variable_weight=False):
    if type(H) != list:
        return _generate_G_from_H(H, variable_weight)
    else:
        G = []
        for sub_H in H:
            G.append(generate_G_from_H(sub_H, variable_weight))
        return G

def _generate_G_from_H(H, variable_weight=False):
    H = np.array(H)
    n_edge = H.shape[1]
    W = np.ones(n_edge)
    # the degree of the node
    DV = np.sum(H * W, axis=1)
    # the degree of the hyperedge
    DE = np.sum(H, axis=0)
    new_matrix=np.diag(np.power(DE, -1))
    new_matrix[np.isnan(new_matrix) | np.isinf(new_matrix)] = 0.0
    invDE = np.mat(new_matrix)
    new_matrix=np.diag(np.power(DV, -0.5))
    new_matrix[np.isnan(new_matrix) | np.isinf(new_matrix)] = 0.0
    DV2 = np.mat(new_matrix)
    W = np.mat(np.diag(W))
    H = np.mat(H)
    HT = H.T

    if variable_weight:
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        return DV2_H, W, invDE_HT_DV2
    else:
        G = DV2 * H * W * invDE * HT * DV2
        return G

def build_hypergraph(sim, num_neighbor):
    if num_neighbor > sim.shape[0] or num_neighbor < 0:
        num_neighbor = sim.shape[0]
    neighbor = np.argpartition(-sim, kth=num_neighbor, axis=1)[:, :num_neighbor]
    row_index = np.arange(neighbor.shape[0]).repeat(neighbor.shape[1])
    col_index = neighbor.reshape(-1)
    zero=np.zeros((sim.shape[0],sim.shape[1]))
    zero[row_index, col_index]=sim[row_index, col_index]
    return zero
def build_muti_hypergraph(sim, num_neighbor):
    if num_neighbor > sim.shape[0] or num_neighbor < 0:
        num_neighbor = sim.shape[0]
    neighbor = np.argpartition(-sim, kth=num_neighbor, axis=1)[:, :num_neighbor]
    row_index = np.arange(neighbor.shape[0]).repeat(neighbor.shape[1])
    col_index = neighbor.reshape(-1)
    zero=np.zeros((sim.shape[0],sim.shape[1]))
    zero[row_index, col_index]=sim[row_index, col_index]
    return zero

def loadGR1(mm):
    dig=np.ones(mm.shape[0])
    I=np.diag(dig)
    H=mm+I
    G =_generate_G_from_H(H)
    return G

def loadGD1(dd):
    dig = np.ones(dd.shape[0])
    I = np.diag(dig)
    H = dd+I
    G =_generate_G_from_H(H)
    return G

def generate_association_matrix(similarity_matrix):
    n = similarity_matrix.shape[0]
    association_matrix = np.zeros_like(similarity_matrix, dtype=int)

    for i in range(n):
        sorted_indices = np.argsort(similarity_matrix[i])[::-1]
        association_matrix[i, sorted_indices[:7]] = 1

    return association_matrix

class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()
        self.dropout = nn.Dropout(0.4)
        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        self.act = nn.ReLU()
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        torch.nn.init.xavier_uniform_(self.weight)
    def forward(self, x: torch.Tensor, G: torch.Tensor):
        # x = torch.matmul(x, self.weight.cuda())
        x = torch.matmul(x, self.weight)
        x = G@x
        if self.bias is not None:
            # x = x + self.bias.cuda()
            x = x + self.bias
        return x
class HGNN_fc(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(HGNN_fc, self).__init__()
        self.fc = nn.Linear(in_ch, out_ch)

    def forward(self, x):
        return self.fc(x)


class HGNN_embedding(nn.Module):
    def __init__(self, in_ch, n_hid, dropout=0.5):
        super(HGNN_embedding, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_ch, n_hid)
        self.hgc2 = HGNN_conv(n_hid, n_hid)

    def forward(self, x, G):
        x = F.relu(self.hgc1(x, G))
        x = F.dropout(x, self.dropout)
        x = F.relu(self.hgc2(x, G))
        return x


class PCP(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, topo_scores, target_scores):

        if len(topo_scores.shape) == 2:
            topo_scores = topo_scores.flatten()
            target_scores = target_scores.flatten()

        scores = torch.stack([topo_scores, target_scores], dim=-1)

        beta = self.predictor(scores).squeeze(-1)

        return beta

class DIAM(nn.Module):
    def __init__(self, args, drug_feature_dim, disease_feature_dim, protein_feature_dim, use_mechanism_weights=True):
        super(DIAM, self).__init__()
        self.args = args
        self.hgt = HGTEncoder(self.args)
        self.contrast_r = Contrast(self.args)
        self.contrast_d = Contrast(self.args)
        self.n_neg = self.args.n_neg
        self.pool = self.args.pool
        self.drug_linear = nn.Linear(drug_feature_dim, args.hgt_in_dim)
        self.disease_linear = nn.Linear(disease_feature_dim, args.hgt_in_dim)
        self.protein_linear = nn.Linear(protein_feature_dim, args.hgt_in_dim)
        self.HGNN = HGNN_embedding(args.hgt_in_dim, args.hgt_out_dim)

        self.hyperedge_k = 15
        self.use_mechanism_weights = use_mechanism_weights
        self.hidden_dim = args.hgt_out_dim

        self.mlp = nn.Sequential(
            nn.Linear(self.args.hgt_out_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 2)
        )
        self.alpha = args.alpha
        self.dynamic_beta = PCP(hidden_dim=64)
        self.use_dynamic_beta = True

    def compute_topo_path_scores(self, data, train_associations):
        S_r = data['drug_sim']
        S_d = data['disease_sim']

        train_drug_indices = np.unique([x[0] for x in train_associations])
        train_disease_indices = np.unique([x[1] for x in train_associations])

        topo_scores = np.zeros((data['drug_number'], data['disease_number']))

        A_rd_train = np.zeros_like(data['A_rd'])
        for (drug_idx, disease_idx) in train_associations:
            A_rd_train[drug_idx, disease_idx] = 1

        for i in train_drug_indices:
            for j in train_disease_indices:
                drugs_for_disease_j = np.where(A_rd_train[:, j] == 1)[0]
                drugs_for_disease_j = drugs_for_disease_j[drugs_for_disease_j != i]
                sim_r = np.mean(S_r[i, drugs_for_disease_j]) if len(drugs_for_disease_j) > 0 else 0

                diseases_for_drug_i = np.where(A_rd_train[i, :] == 1)[0]
                diseases_for_drug_i = diseases_for_drug_i[diseases_for_drug_i != j]
                sim_d = np.mean(S_d[j, diseases_for_drug_i]) if len(diseases_for_drug_i) > 0 else 0

                topo_scores[i, j] = self.alpha * sim_r + (1 - self.alpha) * sim_d

        return topo_scores

    def compute_target_path_scores(self, data):
        A_rt = data['A_rt']  # shape: (drug_number, protein_number)
        A_dt = data['A_dt']  # shape: (disease_number, protein_number)

        target_drug_count = np.sum(A_rt, axis=0)
        target_disease_count = np.sum(A_dt, axis=0)
        target_importance = np.log(1 + target_drug_count + target_disease_count)

        if np.max(target_importance) > 0:
            target_importance = target_importance / np.max(target_importance)

        weighted_targets = A_rt * target_importance
        target_scores = weighted_targets @ A_dt.T

        data['target_path_scores'] = target_scores
        return data

    def compute_mechanism_weights(self, data, use_dynamic=True):
        topo_scores = data['topo_path_scores']
        target_scores = data['target_path_scores']

        topo_norm = (topo_scores - np.min(topo_scores)) / (np.max(topo_scores) - np.min(topo_scores) + 1e-8)
        target_norm = (target_scores - np.min(target_scores)) / (
                    np.max(target_scores) - np.min(target_scores) + 1e-8)

        if use_dynamic and self.use_dynamic_beta:
            topo_tensor = torch.tensor(topo_norm, dtype=torch.float32).to(next(self.parameters()).device)
            target_tensor = torch.tensor(target_norm, dtype=torch.float32).to(next(self.parameters()).device)

            with torch.no_grad():
                beta_dynamic = self.dynamic_beta(topo_tensor, target_tensor)  # [n_drugs*n_diseases]
                beta_dynamic = beta_dynamic.view(topo_scores.shape).cpu().numpy()

            mechanism_weights = np.power(topo_norm + 1e-8, 1 - beta_dynamic) * \
                                np.power(target_norm + 1e-8, beta_dynamic)

            data['beta_values'] = beta_dynamic
        else:
            mechanism_weights = np.power(topo_norm + 1e-8, 1 - self.beta) * \
                                np.power(target_norm + 1e-8, self.beta)
            data['beta_values'] = np.full_like(topo_norm, self.beta)

        data['mechanism_weights'] = mechanism_weights
        return data

    def forward(self, g_rm, g_dm, pos_r, pos_d, drug_feature, disease_feature, protein_feature,
                pos_sample, neg_sample, neg_candidates_r, neg_candidates_d, neg, data, return_beta=False):
        if isinstance(pos_sample, np.ndarray):
            pos_sample = torch.tensor(pos_sample, dtype=torch.long).to(drug_feature.device)
        if isinstance(neg, np.ndarray):
            neg = torch.tensor(neg, dtype=torch.long).to(drug_feature.device)
        if isinstance(neg_candidates_r, np.ndarray):
            neg_candidates_r = torch.tensor(neg_candidates_r, dtype=torch.long).to(drug_feature.device)
        if isinstance(neg_candidates_d, np.ndarray):
            neg_candidates_d = torch.tensor(neg_candidates_d, dtype=torch.long).to(drug_feature.device)

        drug_feature = F.elu(self.drug_linear(drug_feature))
        disease_feature = F.elu(self.disease_linear(disease_feature))
        protein_feature = F.elu(self.protein_linear(protein_feature))

        drug_drug = data['drug_sim']
        drug_association = generate_association_matrix(drug_drug)
        rr = build_hypergraph(drug_association, self.hyperedge_k)
        rr_graph = loadGR1(rr)
        rr_feature = drug_feature
        rr_embedding = self.HGNN(torch.tensor(rr_feature, dtype=torch.float32), torch.tensor(rr_graph, dtype=torch.float32))

        disease_disease = data['disease_sim']
        disease_association = generate_association_matrix(disease_disease)
        dd = build_hypergraph(disease_association, self.hyperedge_k)
        dd_graph = loadGR1(dd)
        dd_feature = disease_feature
        dd_embedding = self.HGNN(torch.tensor(dd_feature, dtype=torch.float32), torch.tensor(dd_graph, dtype=torch.float32))

        r_mp, d_mp = self.hgt(g_rm, g_dm, drug_feature, disease_feature, protein_feature)

        l_r = self.contrast_r(r_mp, rr_embedding, pos_r)
        l_d = self.contrast_d(d_mp, dd_embedding, pos_d)

        r_emb = r_mp
        d_emb = d_mp

        extended_pos_indices = pos_sample
        target_pos_size = len(pos_sample)
        target_neg_size = target_pos_size
        target_total_size = target_pos_size + target_neg_size

        if self.use_mechanism_weights and data.get('mechanism_weights') is not None:
            mechanism_weights = data['mechanism_weights']
            neg_threshold = np.percentile(mechanism_weights, self.args.neg_threshold)
            hard_neg = (mechanism_weights >= neg_threshold) & (data['A_rd'] == 0)
            hard_neg_indices = torch.nonzero(torch.tensor(hard_neg, dtype=torch.float32).to(drug_feature.device),
                                             as_tuple=False)
            if len(hard_neg_indices) > target_neg_size:
                hard_neg_indices = hard_neg_indices[torch.randperm(len(hard_neg_indices))[:target_neg_size]]
            elif len(hard_neg_indices) < target_neg_size:
                extra_neg_size = target_neg_size - len(hard_neg_indices)
                extra_neg = neg[torch.randperm(len(neg))[:extra_neg_size]] if len(neg) >= extra_neg_size else neg
                hard_neg_indices = torch.cat((hard_neg_indices, extra_neg), dim=0) if len(
                    hard_neg_indices) > 0 else extra_neg
            combined_neg_indices = hard_neg_indices
        else:
            if len(neg) > target_neg_size:
                combined_neg_indices = neg[torch.randperm(len(neg))[:target_neg_size]]
            else:
                combined_neg_indices = neg
            if isinstance(combined_neg_indices, np.ndarray):
                combined_neg_indices = torch.tensor(combined_neg_indices, dtype=torch.long).to(drug_feature.device)

        emb_neg_r = r_emb[combined_neg_indices[:, 0]]
        emb_neg_d = d_emb[combined_neg_indices[:, 1]]

        pos_topo_scores = torch.tensor([
            data['topo_path_scores'][pair[0].item(), pair[1].item()]
            for pair in extended_pos_indices
        ], dtype=torch.float32).to(drug_feature.device)

        pos_target_scores = torch.tensor([
            data['target_path_scores'][pair[0].item(), pair[1].item()]
            for pair in extended_pos_indices
        ], dtype=torch.float32).to(drug_feature.device)

        if self.use_dynamic_beta:
            beta_pos = self.dynamic_beta(pos_topo_scores, pos_target_scores)  # [batch]

            drdi_emb_pos = torch.mul(r_emb[extended_pos_indices[:, 0]],
                                     d_emb[extended_pos_indices[:, 1]])

            beta_weight = beta_pos.unsqueeze(-1)  # [batch, 1]
            drdi_emb_pos = drdi_emb_pos + 0.5 * beta_weight* drdi_emb_pos
        else:
            drdi_emb_pos = torch.mul(r_emb[extended_pos_indices[:, 0]],
                                     d_emb[extended_pos_indices[:, 1]])

        neg_topo_scores = torch.tensor([
            data['topo_path_scores'][pair[0].item(), pair[1].item()]
            for pair in combined_neg_indices
        ], dtype=torch.float32).to(drug_feature.device)

        neg_target_scores = torch.tensor([
            data['target_path_scores'][pair[0].item(), pair[1].item()]
            for pair in combined_neg_indices
        ], dtype=torch.float32).to(drug_feature.device)

        if self.use_dynamic_beta:
            beta_neg = self.dynamic_beta(neg_topo_scores, neg_target_scores)
            drdi_emb_neg = torch.mul(emb_neg_r, emb_neg_d)
            beta_weight_neg = beta_neg.unsqueeze(-1)
            drdi_emb_neg = drdi_emb_neg  + 0.5 * beta_weight_neg * drdi_emb_neg
        else:
            drdi_emb_neg = torch.mul(emb_neg_r, emb_neg_d)

        drdi_emb = torch.cat((drdi_emb_pos, drdi_emb_neg), dim=0)
        output = self.mlp(drdi_emb)

        if return_beta and self.use_dynamic_beta:
            return l_r + l_d, output, drdi_emb_pos, drdi_emb_neg, torch.cat([beta_pos, beta_neg])
        else:
            return l_r + l_d, output, drdi_emb_pos, drdi_emb_neg




