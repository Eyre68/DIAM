import dgl
import random
import torch
torch.manual_seed(123)
random.seed(123)


def build_hetero_graphs(g, target_category, edge_types, data=None):
    hetero_dic = {}
    edge_weights = {}
    for edge, (src_type, dst_type) in edge_types.items():
        src, dst = g[edge].edges()
        hetero_dic[(src_type, edge, dst_type)] = (src, dst)
        edge_weights[(src_type, edge, dst_type)] = torch.ones(len(src), dtype=torch.float32)

    new_g = dgl.heterograph(hetero_dic)
    new_g.nodes[target_category].data['h'] = g.ndata['h'][target_category]

    for etype, weights in edge_weights.items():
        new_g.edges[etype].data['weight'] = weights

    return new_g


def create_dgl_heterograph(drdipr_graph, edge_types, args, data):
    target_category = 'drug'
    g_m_drug = build_hetero_graphs(drdipr_graph, target_category, edge_types, data)

    target_category = 'disease'
    g_m_disease = build_hetero_graphs(drdipr_graph, target_category, edge_types, data)

    return g_m_drug, g_m_disease