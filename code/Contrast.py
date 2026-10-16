import torch
import torch.nn as nn

import random
torch.manual_seed(123)
random.seed(123)


class Contrast(nn.Module):
    def __init__(self, args):
        super(Contrast, self).__init__()
        self.hidden_dim = args.hgt_out_dim
        self.proj_cross = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        self.proj_intra_hgt = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        self.proj_intra_hgcn = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        self.tau_cross = args.tau_cross
        self.tau_hgt = args.tau_hgt
        self.tau_hgcn = args.tau_hgcn
        self.lam_cross = args.lam_cross
        self.lam_intra = args.lam_intra

        for model in self.proj_cross:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)
        for model in self.proj_intra_hgt:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)
        for model in self.proj_intra_hgcn:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)

    def sim(self, z1, z2, tau):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / tau)
        return sim_matrix

    def forward(self, z_hgt, z_hgcn, pos):
        z_hgt_cross = self.proj_cross(z_hgt)
        z_hgcn_cross = self.proj_cross(z_hgcn)
        z_hgt_intra = self.proj_intra_hgt(z_hgt)
        z_hgcn_intra = self.proj_intra_hgcn(z_hgcn)
        matrix_hgt2hgcn = self.sim(z_hgt_cross, z_hgcn_cross, self.tau_cross)
        matrix_hgcn2hgt = matrix_hgt2hgcn.t()

        matrix_hgt2hgcn = matrix_hgt2hgcn / (torch.sum(matrix_hgt2hgcn, dim=1).view(-1, 1) + 1e-8)
        l_hgt_cross = -torch.log(matrix_hgt2hgcn.mul(pos).sum(dim=-1)).mean()

        matrix_hgcn2hgt = matrix_hgcn2hgt / (torch.sum(matrix_hgcn2hgt, dim=1).view(-1, 1) + 1e-8)
        l_hgcn_cross = -torch.log(matrix_hgcn2hgt.mul(pos).sum(dim=-1)).mean()

        l_cross = self.lam_cross * l_hgt_cross + (1 - self.lam_cross) * l_hgcn_cross

        matrix_hgt2hgt = self.sim(z_hgt_intra, z_hgt_intra, self.tau_hgt)
        matrix_hgcn2hgcn = self.sim(z_hgcn_intra, z_hgcn_intra, self.tau_hgcn)

        matrix_hgt2hgt = matrix_hgt2hgt / (torch.sum(matrix_hgt2hgt, dim=1).view(-1, 1) + 1e-8)
        l_hgt_intra = -torch.log(matrix_hgt2hgt.mul(pos).sum(dim=-1)).mean()

        matrix_hgcn2hgcn = matrix_hgcn2hgcn / (torch.sum(matrix_hgcn2hgcn, dim=1).view(-1, 1) + 1e-8)
        l_hgcn_intra = -torch.log(matrix_hgcn2hgcn.mul(pos).sum(dim=-1)).mean()
        l_intra = self.lam_intra * l_hgt_intra + (1 - self.lam_intra) * l_hgcn_intra

        return l_cross + l_intra

