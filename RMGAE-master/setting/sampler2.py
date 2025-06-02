import numpy as np
import scipy.sparse as sp
import torch
import torch.utils.data as data_utils


class EdgeSampler2(data_utils.Dataset):
    """Sample edges, non-edges, and masked edges from a graph.

    Args:
        A: adjacency matrix.
        num_pos: number of edges per batch.
        num_neg: number of non-edges per batch.
        masked_edges: edges that were masked in the adjacency matrix.
        num_masked_samples: number of masked edges to sample per batch.
    """

    def __init__(self, A, num_pos=1000, num_neg=1000, masked_edges=None, num_masked_samples=500):
        self.num_pos = num_pos
        self.num_neg = num_neg
        self.num_masked_samples = num_masked_samples
        self.A = A
        self.edges = np.transpose(A.nonzero())
        self.num_nodes = A.shape[0]
        self.num_edges = self.edges.shape[0]

        self.masked_edges = masked_edges if masked_edges is not None else np.array([])

    def __getitem__(self, key):
        np.random.seed(key)

        # 正样本：从原始边集中采样
        edges_idx = np.random.randint(0, self.num_edges, size=self.num_pos, dtype=np.int64)
        next_edges = self.edges[edges_idx, :]

        # 负样本：随机采样非边
        generated = False
        while not generated:
            candidate_ne = np.random.randint(0, self.num_nodes, size=(2 * self.num_neg, 2), dtype=np.int64)
            cne1, cne2 = candidate_ne[:, 0], candidate_ne[:, 1]
            to_keep = (1 - self.A[cne1, cne2]).astype(np.bool_).A1 * (cne1 != cne2)
            next_nonedges = candidate_ne[to_keep][:self.num_neg]
            generated = to_keep.sum() >= self.num_neg

        # 额外采样被遮盖的边
        if self.num_masked_samples > 0 and len(self.masked_edges) > 0:
            mask_idx = np.random.choice(len(self.masked_edges), self.num_masked_samples, replace=False)
            sampled_masked_edges = self.masked_edges[mask_idx]

        else:
            sampled_masked_edges = np.empty((0, 2), dtype=np.int64)

        return (
            torch.LongTensor(next_edges),
            torch.LongTensor(next_nonedges),
            torch.LongTensor(sampled_masked_edges)
        )

    def __len__(self):
        return 2 ** 32


def collate_fn(batch):
    edges, nonedges, masked_edges = batch[0]
    return (edges, nonedges, masked_edges)


def get_edge_sampler2(A, num_pos=1000, num_neg=1000, masked_edges=None, num_masked_samples=500, num_workers=2):
    data_source = EdgeSampler2(A, num_pos, num_neg, masked_edges, num_masked_samples)
    return data_utils.DataLoader(data_source, num_workers=num_workers, collate_fn=collate_fn)
