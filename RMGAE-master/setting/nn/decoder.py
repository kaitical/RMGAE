import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td


__all__ = [
    'BerpoDecoder',
    'MaskGAEBerpoDecoder',
]


class BernoulliDecoder(nn.Module):
    def __init__(self, num_nodes, num_edges, balance_loss=False):
        """Base class for Bernoulli decoder.

        Args:
            num_nodes: Number of nodes in a graph.
            num_edges: Number of edges in a graph.
            balance_loss: Whether to balance contribution from edges and non-edges.
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.num_possible_edges = num_nodes**2 - num_nodes
        self.num_nonedges = self.num_possible_edges - self.num_edges
        self.balance_loss = balance_loss

    def forward_batch(self, emb, idx):
        """Compute probabilities of given edges.

        Args:
            emb: embedding matrix, shape (num_nodes, emb_dim)
            idx: edge indices, shape (batch_size, 2)

        Returns:
            edge_probs: Bernoulli distribution for given edges, shape (batch_size)
        """
        raise NotImplementedError

    def forward_full(self, emb):
        """Compute probabilities for all edges.

        Args:
            emb: embedding matrix, shape (num_nodes, emb_dim)

        Returns:
            edge_probs: Bernoulli distribution for all edges, shape (num_nodes, num_nodes)
        """
        raise NotImplementedError

    def loss_batch(self, emb, ones_idx, zeros_idx):
        """Compute loss for given edges and non-edges."""
        raise NotImplementedError

    def loss_full(self, emb, adj):
        """Compute loss for all edges and non-edges."""
        raise NotImplementedError


class BerpoDecoder(BernoulliDecoder):
    def __init__(self, num_nodes, num_edges, balance_loss=False):
        super().__init__(num_nodes, num_edges, balance_loss)
        edge_proba = num_edges / (num_nodes**2 - num_nodes)
        self.eps = -np.log(1 - edge_proba)

    def forward_batch(self, emb, idx):
        """Compute probabilities of given edges.

        Args:
            emb: embedding matrix, shape (num_nodes, emb_dim)
            idx: edge indices, shape (batch_size, 2)

        Returns:
            edge_probs: Bernoulli distribution for given edges, shape (batch_size)
        """
        e1, e2 = idx.t()
        logits = torch.sum(emb[e1] * emb[e2], dim=1)
        logits += self.eps
        probs = 1 - torch.exp(-logits)
        return td.Bernoulli(probs=probs)

    def forward_full(self, emb):
        """Compute probabilities for all edges.

        Args:
            emb: embedding matrix, shape (num_nodes, emb_dim)

        Returns:
            edge_probs: Bernoulli distribution for all edges, shape (num_nodes, num_nodes)
        """
        logits = emb @ emb.t()
        logits += self.eps
        probs = 1 - torch.exp(-logits)
        return td.Bernoulli(probs=probs)

    def loss_batch(self, emb, ones_idx, zeros_idx):
        """Compute BerPo loss for a batch of edges and non-edges."""
        # Loss for edges

        e1, e2 = ones_idx[:, 0], ones_idx[:, 1]
        edge_dots = torch.sum(emb[e1] * emb[e2], dim=1)
        loss_edges = -torch.mean(torch.log(-torch.expm1(-self.eps - edge_dots)))

        # Loss for non-edges
        ne1, ne2 = zeros_idx[:, 0], zeros_idx[:, 1]
        loss_nonedges = torch.mean(torch.sum(emb[ne1] * emb[ne2], dim=1))
        if self.balance_loss:
            neg_scale = 1.0
        else:
            neg_scale = self.num_nonedges / self.num_edges
        return (loss_edges + neg_scale * loss_nonedges) / (1 + neg_scale)

    def loss_full(self, emb, adj):
        """Compute BerPo loss for all edges & non-edges in a graph."""
        e1, e2 = adj.nonzero()
        edge_dots = torch.sum(emb[e1] * emb[e2], dim=1)
        loss_edges = -torch.sum(torch.log(-torch.expm1(-self.eps - edge_dots)))

        # Correct for overcounting F_u * F_v for edges and nodes with themselves
        self_dots_sum = torch.sum(emb * emb)
        correction = self_dots_sum + torch.sum(edge_dots)
        sum_emb = torch.sum(emb, dim=0, keepdim=True).t()
        loss_nonedges = torch.sum(emb @ sum_emb) - correction

        if self.balance_loss:
            neg_scale = 1.0
        else:
            neg_scale = self.num_nonedges / self.num_edges
        return (loss_edges / self.num_edges + neg_scale * loss_nonedges / self.num_nonedges) / (1 + neg_scale)
class MaskGAEBerpoDecoder(BerpoDecoder):
    def __init__(self, num_nodes, num_edges, balance_loss=False,a=0.5):
        """MaskGAE + BerPo结合解码器

        Args:
            num_nodes: 图中的节点数
            num_edges: 图中的边数
            balance_loss: 是否平衡损失中边和非边的贡献
            mask_rate: 遮蔽率（用于 MaskGAE）
        """
        super().__init__(num_nodes, num_edges, balance_loss)
        self.a = a  # 设置权重参数
         # MaskGAE 中用于遮蔽的比率

    def forward_batch(self, emb, idx, mask_idx=None):
        """计算给定边的概率，同时支持遮蔽部分的边的计算

        Args:
            emb: 嵌入矩阵，形状为 (num_nodes, emb_dim)
            idx: 边的索引，形状为 (batch_size, 2)
            mask_idx: 遮蔽边的索引，形状为 (batch_size, 2)，用于 MaskGAE

        Returns:
            edge_probs: 给定边的伯努利分布概率，形状为 (batch_size)
        """
        e1, e2 = idx.t()
        logits = torch.sum(emb[e1] * emb[e2], dim=1)
        logits += self.eps
        probs = 1 - torch.exp(-logits)

        # 如果存在遮蔽边（mask_idx），计算遮蔽边的概率
        if mask_idx is not None:
            m1, m2 = mask_idx.t()
            masked_logits = torch.sum(emb[m1] * emb[m2], dim=1)
            masked_logits += self.eps
            masked_probs = 1 - torch.exp(-masked_logits)
            return td.Bernoulli(probs=probs), td.Bernoulli(probs=masked_probs)

        return td.Bernoulli(probs=probs)

    def forward_full(self, emb, mask_emb=None):
        """计算所有边的概率，支持遮蔽部分的边

        Args:
            emb: 嵌入矩阵，形状为 (num_nodes, emb_dim)
            mask_emb: 遮蔽后的嵌入矩阵（可选）

        Returns:
            edge_probs: 所有边的伯努利分布概率，形状为 (num_nodes, num_nodes)
        """
        logits = emb @ emb.t()
        logits += self.eps
        probs = 1 - torch.exp(-logits)

        # 如果存在遮蔽后的嵌入（mask_emb），计算遮蔽部分的概率
        if mask_emb is not None:
            masked_logits = mask_emb @ mask_emb.t()
            masked_logits += self.eps
            masked_probs = 1 - torch.exp(-masked_logits)
            return td.Bernoulli(probs=probs), td.Bernoulli(probs=masked_probs)

        return td.Bernoulli(probs=probs)

    def loss_batch(self, emb, ones_idx, zeros_idx, mask_idx=None):
        """计算给定边和非边的损失，同时加入遮蔽边的损失

        Args:
            emb: 嵌入矩阵，形状为 (num_nodes, emb_dim)
            ones_idx: 真实边索引
            zeros_idx: 负样本索引
            mask_idx: 遮蔽边索引（可选）

        Returns:
            total_loss: 总损失（包含原始损失和 MaskGAE 重建损失）
        """
        # 标准的 BerPo Loss
        loss_berpo = super().loss_batch(emb, ones_idx, zeros_idx)

        # MaskGAE 的重建损失
        if mask_idx is not None:
            mask_loss = self.maskgae_loss(emb, mask_idx)
            total_loss = self.a * loss_berpo + (1 - self.a) * mask_loss
        else:
            total_loss = loss_berpo

        return total_loss

    def loss_full(self, emb, adj, mask_idx=None):
        """计算包含 MaskGAE 重建损失的 BerPo 损失

        Args:
            emb: 嵌入矩阵，形状为 (num_nodes, emb_dim)
            adj: 邻接矩阵
            mask_idx: 遮蔽边的索引（可选）

        Returns:
            total_loss: 总损失
        """
        # 计算原始 BerPo loss
        loss_berpo = super().loss_full(emb, adj)

        # 计算 MaskGAE 相关的重建损失
        if mask_idx is not None:
            mask_loss = self.maskgae_loss(emb, mask_idx)
            total_loss = loss_berpo + mask_loss
        else:
            total_loss = loss_berpo

        return total_loss
    def maskgae_loss(self, emb, masked_ones_idx):
        """计算 MaskGAE 的重建损失（遮蔽边的重建损失）。

        Args:
            emb: 嵌入矩阵，形状为 (num_nodes, emb_dim)
            masked_ones_idx: 被遮蔽的真实边索引

        Returns:
            mask_loss: MaskGAE 损失
        """


        m1, m2 = masked_ones_idx[:, 0], masked_ones_idx[:, 1]
        masked_scores = torch.sum(emb[m1] * emb[m2], dim=1)
        mask_loss = F.binary_cross_entropy_with_logits(masked_scores, torch.ones_like(masked_scores))
        return mask_loss