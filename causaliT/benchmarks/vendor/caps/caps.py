# Vendored from https://github.com/E2real/CaPS (MIT, see LICENSE in this folder)
# commit 325a4ffffab0f9cd14db66dc404b422bc976d916.
#
# Contents: the CaPS order search and parent-score functions, copied from
# train_order.py (Stein_hess, compute_top_order, previously nested inside
# train_test) and utils.py (full_DAG, get_parents_score,
# pre_pruning_with_parents_score, add_edge_with_parents_score, is_acyclic).
# The algorithmic code is unchanged. Edits, all non-algorithmic:
#   - nested functions lifted to module level; `device` is a parameter;
#   - the debug print of parents_score removed;
#   - the R CAM pruning (cdt.launch_R_script + pruning_R_files/) is NOT
#     vendored; causaliT/benchmarks/methods/caps.py uses dodiscover's Python
#     CAMPruning with the same cutoff (0.001) instead.

import networkx as nx
import numpy as np
import torch


def is_acyclic(adj_matrix):
    G = nx.DiGraph(adj_matrix)
    return nx.is_directed_acyclic_graph(G)


def full_DAG(top_order):
    d = len(top_order)
    A = np.zeros((d, d))
    for i, var in enumerate(top_order):
        A[var, top_order[i + 1:]] = 1
    return A


def get_parents_score(curr_H, full_H, i):
    full_H = torch.hstack([full_H[0:i], full_H[i + 1:]])
    parents_score = np.abs(curr_H - full_H)
    parents_score = torch.cat([parents_score[:i], torch.tensor([0.0]), parents_score[i:]])
    return parents_score


def add_edge_with_parents_score(dag, parents_score, lambda2):
    avg_parent_score = (dag * parents_score.T).mean()
    add_edge = (parents_score.T >= lambda2 * avg_parent_score).astype(np.int32)
    idx = np.transpose(np.nonzero(add_edge))
    val = np.array([parents_score.T[i[0]][i[1]] for i in idx])
    sorted_idx = np.argsort(-val)
    idx = idx[sorted_idx]
    for i in idx:
        if dag[i[0], i[1]] == 1:
            continue
        else:
            dag[i[0], i[1]] = 1
            if is_acyclic(dag):
                continue
            else:
                dag[i[0], i[1]] = 0
    return dag


def pre_pruning_with_parents_score(init_dag, parents_score, lambda1):
    d = init_dag.shape[0]
    threshold = (np.max(parents_score, axis=1) / lambda1).reshape(d, 1)
    threshold = np.tile(threshold, (1, d))
    mask = (parents_score >= threshold).T.astype(np.int32)
    return init_dag * mask


def Stein_hess(X, eta_G, eta_H, s=None, device="cpu"):
    """
    Estimates the diagonal of the Hessian of log p_X at the provided samples points
    X, using first and second-order Stein identities
    """
    n, d = X.shape
    X = X.to(device)
    X_diff = X.unsqueeze(1) - X
    if s is None:
        D = torch.norm(X_diff, dim=2, p=2)
        s = D.flatten().median()
    K = torch.exp(-torch.norm(X_diff, dim=2, p=2) ** 2 / (2 * s ** 2)) / s

    nablaK = -torch.einsum('kij,ik->kj', X_diff, K) / s ** 2
    G = torch.matmul(torch.inverse(K + eta_G * torch.eye(n).to(device)), nablaK)

    nabla2K = torch.einsum('kij,ik->kj', -1 / s ** 2 + X_diff ** 2 / s ** 4, K)
    return (-G ** 2 + torch.matmul(torch.inverse(K + eta_H * torch.eye(n).to(device)), nabla2K)).to('cpu')


def compute_top_order(X, eta_G, eta_H, dispersion="mean", device="cpu"):
    n, d = X.shape
    full_X = X
    order = []
    active_nodes = list(range(d))
    for i in range(d - 1):
        H = Stein_hess(X, eta_G, eta_H, device=device)
        if dispersion == "mean":  # Lemma 1 of CaPS
            l = int(H.mean(axis=0).argmax())
        else:
            raise Exception("Unknown dispersion criterion")

        order.append(active_nodes[l])
        active_nodes.pop(l)

        X = torch.hstack([X[:, 0:l], X[:, l + 1:]])
    order.append(active_nodes[0])
    order.reverse()

    # compute parents score
    full_H = Stein_hess(full_X, eta_G, eta_H, device=device).mean(axis=0)
    parents_score = np.zeros((d, d))
    for i in range(d):
        curr_X = torch.hstack([full_X[:, 0:i], full_X[:, i + 1:]])
        curr_H = Stein_hess(curr_X, eta_G, eta_H, device=device).mean(axis=0)
        parents_score[i] = get_parents_score(curr_H, full_H, i)

    return order, parents_score
