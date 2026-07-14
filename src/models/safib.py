# coding: utf-8
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, degree

from common.abstract_recommender import GeneralRecommender


class SAFIBBackbone(GeneralRecommender):
    """Encode ID, visual, and textual signals on the user--item graph."""

    def __init__(self, config, dataset):
        super(SAFIBBackbone, self).__init__(config, dataset)

        num_user = self.n_users
        num_item = self.n_items
        batch_size = config['train_batch_size']
        dim_x = config['embedding_size']
        self.n_layers = config['n_mm_layers']
        self.knn_k = config['knn_k']
        self.mm_image_weight = config['mm_image_weight']
        has_id = True
        self.num_freq_bands = config['num_freq_bands']
        self.ib_alpha = config['ib_alpha']
        self.ib_mu = config['ib_mu']
        self.ib_phi_plus = config['ib_phi_plus']
        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.aggr_mode = config['aggr_mode']
        self.num_layer = config['num_layers']
        self.dataset = dataset
        self.reg_weight = config['reg_weight']
        self.ib_weight = config['ib_weight']
        self.drop_rate = 0.1
        self.dim_latent = 64
        self.mm_adj = None
        self.strict_missing_graph = self._config_bool(config, 'strict_missing_graph', False)

        self.user_id_embedding = nn.Embedding(self.n_users, self.dim_latent)
        self.item_id_embedding = nn.Embedding(self.n_items, self.dim_latent)
        nn.init.xavier_uniform_(self.user_id_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)

        # Rebuild from the current masked feature copies. Persisting this graph
        # would couple different missing-mask runs through a shared cache.
        if self.v_feat is not None:
            indices, image_adj = self.get_knn_adj_mat(self.image_embedding.weight.detach())
            image_adj = self._isolate_missing_adj(image_adj, self._missing_indices_from_available(self.image_available))
            self.mm_adj = image_adj
        if self.t_feat is not None:
            indices, text_adj = self.get_knn_adj_mat(self.text_embedding.weight.detach())
            text_adj = self._isolate_missing_adj(text_adj, self._missing_indices_from_available(self.text_available))
            self.mm_adj = text_adj
        if self.v_feat is not None and self.t_feat is not None:
            self.mm_adj = self.mm_image_weight * image_adj + (1.0 - self.mm_image_weight) * text_adj
            del text_adj
            del image_adj
        # packing interaction in training into edge_index
        train_interactions = dataset.inter_matrix(form='coo').astype(np.float32)
        edge_index = self.pack_edge_index(train_interactions)
        self.edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(self.device)
        self.edge_index = torch.cat((self.edge_index, self.edge_index[[1, 0]]), dim=1)

        self.weight_u = nn.Parameter(nn.init.xavier_normal_(
            torch.tensor(np.random.randn(self.num_user, 3, 1), dtype=torch.float32, requires_grad=True)))
        self.weight_u.data = F.softmax(self.weight_u, dim=1)

        self.item_index = torch.zeros([self.num_item], dtype=torch.long)
        index = []
        for i in range(self.num_item):
            self.item_index[i] = i
            index.append(i)
        self.drop_percent = self.drop_rate
        self.single_percent = 1
        self.double_percent = 0

        drop_item = torch.tensor(
            np.random.choice(self.item_index, int(self.num_item * self.drop_percent), replace=False))
        drop_item_single = drop_item[:int(self.single_percent * len(drop_item))]

        self.dropv_node_idx_single = drop_item_single[:int(len(drop_item_single) * 1 / 3)]
        self.dropt_node_idx_single = drop_item_single[int(len(drop_item_single) * 2 / 3):]

        self.dropv_node_idx = self.dropv_node_idx_single
        self.dropt_node_idx = self.dropt_node_idx_single

        mask_cnt = torch.zeros(self.num_item, dtype=int).tolist()
        for edge in edge_index:
            mask_cnt[edge[1] - self.num_user] += 1
        mask_dropv = []
        mask_dropt = []
        for idx, num in enumerate(mask_cnt):
            temp_false = [False] * num
            temp_true = [True] * num
            mask_dropv.extend(temp_false) if idx in self.dropv_node_idx else mask_dropv.extend(temp_true)
            mask_dropt.extend(temp_false) if idx in self.dropt_node_idx else mask_dropt.extend(temp_true)

        edge_index = edge_index[np.lexsort(edge_index.T[1, None])]
        edge_index_dropv = edge_index[mask_dropv]
        edge_index_dropt = edge_index[mask_dropt]

        self.edge_index_dropv = torch.tensor(edge_index_dropv).t().contiguous().to(self.device)
        self.edge_index_dropt = torch.tensor(edge_index_dropt).t().contiguous().to(self.device)

        self.edge_index_dropv = torch.cat((self.edge_index_dropv, self.edge_index_dropv[[1, 0]]), dim=1)
        self.edge_index_dropt = torch.cat((self.edge_index_dropt, self.edge_index_dropt[[1, 0]]), dim=1)

        if self.v_feat is not None:
            self.v_gcn = GCN(self.dataset, batch_size, num_user, num_item, dim_x, self.aggr_mode,
                             num_layer=self.num_layer, has_id=has_id, dropout=self.drop_rate, dim_latent=self.dim_latent,
                             device=self.device, features=self.v_feat)  # 256)
        if self.t_feat is not None:
            self.t_gcn = GCN(self.dataset, batch_size, num_user, num_item, dim_x, self.aggr_mode,
                             num_layer=self.num_layer, has_id=has_id, dropout=self.drop_rate, dim_latent=self.dim_latent,
                             device=self.device, features=self.t_feat)

        self.id_gcn = GCN(self.dataset, batch_size, num_user, num_item, dim_x, self.aggr_mode,
                             num_layer=self.num_layer, has_id=has_id, dropout=self.drop_rate, dim_latent=None,
                             device=self.device, features=self.item_id_embedding.weight)

        self.FeqDecomposeOperator = FrequencyDecompositionModule(self.dim_latent, dim_x, dim_x, dim_x, self.num_freq_bands, self.ib_alpha, self.ib_mu, self.ib_phi_plus, hidden_dim=16)

    def _config_bool(self, config, key, default=False):
        try:
            value = config[key]
        except Exception:
            value = default
        if isinstance(value, str):
            return value.lower() in ['true', '1', 'yes', 'y']
        return bool(value)

    def _missing_indices_from_available(self, available):
        if available is None:
            return torch.empty(0, dtype=torch.long, device=self.device)
        available = available.to(self.device)
        return torch.where(~available)[0].long()

    def _isolate_missing_adj(self, adj, missing_items):
        adj = adj.coalesce()
        if (not self.strict_missing_graph) or missing_items is None or missing_items.numel() == 0:
            return adj
        missing_items = missing_items.to(adj.device).long()
        missing_mask = torch.zeros(adj.size(0), dtype=torch.bool, device=adj.device)
        missing_mask[missing_items] = True
        indices = adj.indices()
        values = adj.values()
        keep = (~missing_mask[indices[0]]) & (~missing_mask[indices[1]])
        kept_indices = indices[:, keep]
        kept_values = values[keep]
        loop_indices = torch.stack([missing_items, missing_items], dim=0)
        loop_values = torch.ones(missing_items.numel(), dtype=values.dtype, device=values.device)
        new_indices = torch.cat([kept_indices, loop_indices], dim=1)
        new_values = torch.cat([kept_values, loop_values], dim=0)
        return torch.sparse.FloatTensor(new_indices, new_values, adj.size()).coalesce()
    def get_knn_adj_mat(self, mm_embeddings):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True).clamp_min(1e-8))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        del sim
        # construct sparse adj
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device)
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        # norm
        return indices, self.compute_normalized_laplacian(indices, adj_size)

    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse.FloatTensor(indices, values, adj_size)

    def pack_edge_index(self, inter_mat):
        rows = inter_mat.row
        cols = inter_mat.col + self.n_users
        return np.column_stack((rows, cols))

    def cross_entropy_loss(self, u_emb, pos_i_emb, neg_i_emb):
        """
        Treat (user, pos_item) as positive samples and (user, neg_item) as negative samples,
        then concatenate them and optimize using a binary cross-entropy loss.
        """
        pos_scores = torch.sum(u_emb * pos_i_emb, dim=1)  # [batch_size]
        neg_scores = torch.sum(u_emb * neg_i_emb, dim=1)  # [batch_size]

        pos_labels = torch.ones_like(pos_scores)  # [batch_size],
        neg_labels = torch.zeros_like(neg_scores) # [batch_size],

        all_scores = torch.cat([pos_scores, neg_scores], dim=0)   # [2 * batch_size]
        all_labels = torch.cat([pos_labels, neg_labels], dim=0)   # [2 * batch_size]

        loss = F.binary_cross_entropy_with_logits(all_scores, all_labels)

        return loss

    def modality_frequency_contrastive_loss(
        self,
        user_band_embs,
        item_band_embs,
        temperature: float = 1.0,
    ):
        """Measure alignment among ID, visual, and textual blocks by band.

        Args:
            user_band_embs: List[num_bands], each (N_user, 3*D)
            item_band_embs: List[num_bands], each (N_item, 3*D)
            temperature:    float

        Returns:
            loss: scalar tensor
        """
        num_bands = len(user_band_embs)
        loss = 0.0

        for band_idx in range(num_bands):
            user_band = user_band_embs[band_idx]  # (N_user, 3D)
            item_band = item_band_embs[band_idx]  # (N_item, 3D)

            user_id_emb, user_vis_emb, user_txt_emb = torch.chunk(user_band, 3, dim=-1)
            item_id_emb, item_vis_emb, item_txt_emb = torch.chunk(item_band, 3, dim=-1)

            user_cos_id_vis = F.cosine_similarity(user_id_emb, user_vis_emb, dim=-1)
            user_cos_id_txt = F.cosine_similarity(user_id_emb, user_txt_emb, dim=-1)
            user_cos_vis_txt = F.cosine_similarity(user_vis_emb, user_txt_emb, dim=-1)

            item_cos_id_vis = F.cosine_similarity(item_id_emb, item_vis_emb, dim=-1)
            item_cos_id_txt = F.cosine_similarity(item_id_emb, item_txt_emb, dim=-1)
            item_cos_vis_txt = F.cosine_similarity(item_vis_emb, item_txt_emb, dim=-1)

            ones_u = torch.ones_like(user_cos_id_vis)
            ones_i = torch.ones_like(item_cos_id_vis)

            loss = loss + F.mse_loss(user_cos_id_vis, ones_u)
            loss = loss + F.mse_loss(user_cos_id_txt, ones_u)
            loss = loss + F.mse_loss(user_cos_vis_txt, ones_u)

            loss = loss + F.mse_loss(item_cos_id_vis, ones_i)
            loss = loss + F.mse_loss(item_cos_id_txt, ones_i)
            loss = loss + F.mse_loss(item_cos_vis_txt, ones_i)

            if band_idx >= num_bands // 2:
                user_logits = (user_band @ user_band.T) / temperature
                item_logits = (item_band @ item_band.T) / temperature
                loss = loss - F.log_softmax(user_logits, dim=-1).mean()
                loss = loss - F.log_softmax(item_logits, dim=-1).mean()

        # 6 = 3 modality pairs * (user + item)
        normalizer = max(1, num_bands * 6)
        return loss / normalizer

class GCN(torch.nn.Module):
    def __init__(self, datasets, batch_size, num_user, num_item, dim_id, aggr_mode, num_layer, has_id, dropout,
                 dim_latent=None, device=None, features=None):
        super(GCN, self).__init__()
        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.datasets = datasets
        self.dim_id = dim_id
        self.dim_feat = features.size(1)
        self.dim_latent = dim_latent
        self.aggr_mode = aggr_mode
        self.num_layer = num_layer
        self.has_id = has_id
        self.dropout = dropout
        self.device = device

        if self.dim_latent:
            self.preference = nn.Parameter(nn.init.xavier_normal_(torch.tensor(
                np.random.randn(num_user, self.dim_latent), dtype=torch.float32, requires_grad=True),
                gain=1).to(self.device))
            self.MLP = nn.Linear(self.dim_feat, 4 * self.dim_latent)
            self.MLP_1 = nn.Linear(4 * self.dim_latent, self.dim_latent)
            self.conv_embed_layer = Base_gcn(self.dim_latent, self.dim_latent, aggr=self.aggr_mode)
        else:
            self.preference = nn.Parameter(nn.init.xavier_normal_(torch.tensor(
                np.random.randn(num_user, self.dim_feat), dtype=torch.float32, requires_grad=True),
                gain=1).to(self.device))
            self.conv_embed_layer = Base_gcn(self.dim_latent, self.dim_latent, aggr=self.aggr_mode)

    def forward(self, edge_index_drop, edge_index, features):
        temp_features = self.MLP_1(F.leaky_relu(self.MLP(features))) if self.dim_latent else features
        x = torch.cat((self.preference, temp_features), dim=0).to(self.device)
        x = F.normalize(x).to(self.device)
        outs = [x]
        h = x
        for _ in range(self.num_layer):
            h = self.conv_embed_layer(h, edge_index)
            outs.append(h)
        x_hat = sum(outs)
        return x_hat, self.preference

class Base_gcn(MessagePassing):
    def __init__(self, in_channels, out_channels, normalize=True, bias=True, aggr='add', **kwargs):
        super(Base_gcn, self).__init__(aggr=aggr, **kwargs)
        self.aggr = aggr
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x, edge_index, size=None):
        if size is None:
            edge_index, _ = remove_self_loops(edge_index)
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, size):
        if self.aggr == 'add':
            row, col = edge_index
            deg = degree(row, size[0], dtype=x_j.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            return norm.view(-1, 1) * x_j
        return x_j

    def update(self, aggr_out):
        return aggr_out

    def __repr(self):
        return '{}({},{})'.format(self.__class__.__name__, self.in_channels, self.out_channels)

class FrequencyDecompositionModule(nn.Module):
    def __init__(self, dim_latent, id_dim, v_dim, t_dim, M, ib_alpha, ib_mu, ib_phi_plus, hidden_dim=128):
        """Configure modality-wise SVD decomposition and band fusion."""
        super(FrequencyDecompositionModule, self).__init__()
        self.M = M
        self.id_dim = id_dim
        self.v_dim = v_dim
        self.t_dim = t_dim
        self.all_dim = id_dim + v_dim + t_dim
        self.dim_latent = dim_latent
        self.fusion_layer_user = TaskAwareFrequencyFusion(self.M, self.all_dim, ib_alpha, ib_mu, ib_phi_plus)
        self.fusion_layer_item = TaskAwareFrequencyFusion(self.M, self.all_dim, ib_alpha, ib_mu, ib_phi_plus)

    def frequency_decompose_svd(self, rep):
        """Reconstruct contiguous bands from ordered singular components."""
        M = self.M
        # rep: [N, F]
        rep = torch.nan_to_num(rep)
        try:
            U, S, Vh = torch.linalg.svd(rep, full_matrices=False)
        except RuntimeError:
            try:
                if rep.is_cuda:
                    U, S, Vh = torch.linalg.svd(rep, full_matrices=False, driver="gesvd")
                else:
                    raise
            except RuntimeError:
                rep_cpu = rep.double().cpu()
                U, S, Vh = torch.linalg.svd(rep_cpu, full_matrices=False)
                U = U.to(device=rep.device, dtype=rep.dtype)
                S = S.to(device=rep.device, dtype=rep.dtype)
                Vh = Vh.to(device=rep.device, dtype=rep.dtype)
        # U:  [N, F]
        # S:  [F, ]  (one dimension)
        # Vh: [F, F]

        N, F_ = U.shape  # F_ should be as F
        assert F_ == S.shape[0] and F_ == Vh.shape[0] == Vh.shape[1], \
            f"Shape mismatch in SVD: U({U.shape}), S({S.shape}), Vh({Vh.shape})"

        # calculate each band size
        # e.g. F=192, M=3, split_sizes = [64,64,64]
        split_sizes = []
        base = F_ // M  # base size for each band
        remainder = F_ % M
        for i in range(M):
            size_i = base + (1 if i < remainder else 0)
            split_sizes.append(size_i)
        # sum(split_sizes) == F_

        # splitting S, U, Vh one by one
        freq_components = []
        start = 0
        for size_i in split_sizes:
            end = start + size_i

            # S_i: [size_i, ]
            S_i = S[start:end]
            # U_i: [N, size_i]
            U_i = U[:, start:end]
            # V_i: [size_i, F]
            V_i = Vh[start:end, :]

            # reconstruct corresponding matrix for the band = sum_{k in [start, end]}(S_k * U_{:,k} outer Vh_{k,:})
            # U_i @ diag(S_i) @ V_i
            diag_S_i = torch.diag(S_i)              # (size_i, size_i)
            partial_rep = U_i @ diag_S_i @ V_i      # (N, F)
            freq_components.append(partial_rep)

            start = end

        return freq_components  # List[M], with each element shape (N, F)

    def frequency_decompose_svd_separate(self, rep):
        """Decompose each channel independently and concatenate matching bands."""
        M = self.M

        id_rep, visual_rep, text_rep = torch.split(rep, [self.dim_latent, self.dim_latent, self.dim_latent], dim=-1)

        # Decomposition for different modalities with SVD
        id_frequencies = self.frequency_decompose_svd(id_rep)    # List[M], with each element (num_samples, each_dim)
        visual_frequencies = self.frequency_decompose_svd(visual_rep)
        text_frequencies = self.frequency_decompose_svd(text_rep)

        # re-concat to original dimensions
        freq_components = [torch.cat([id_frequencies[i], visual_frequencies[i], text_frequencies[i]], dim=-1) for i in range(M)]

        return freq_components  # List[M], with each element shape (num_samples, all_dim)

class TaskAwareFrequencyFusion(nn.Module):
    """Fuse spectral bands with adaptive residual gates and capacity control."""

    def __init__(self, M, embed_dim, ib_alpha=1.0, ib_mu=1.0, ib_phi_plus=0.0):
        super().__init__()
        self.M = M
        self.embed_dim = embed_dim

        self.ib_alpha = ib_alpha
        self.ib_mu = ib_mu
        self.ib_phi_plus = ib_phi_plus

        self.num_bands = M
        # Learnable frequency weights
        self.freq_weights = nn.Parameter(torch.ones(M))  # (M,)

        self.freq_gate = nn.Sequential(
            nn.Linear(embed_dim, M),
            nn.Sigmoid(),
        )
        self.gate_scale = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

    def ib_surrogate_loss_from_gate(
        self,
        gate_values: torch.Tensor,   # (N, M, 1) or (N, M)
        alpha: float = 1.0,
        mu: float = 1.0,
        phi_plus: float = 0.0,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        """Penalize large gate increments within and across spectral bands."""
        if gate_values.dim() == 3:
            gate_values = gate_values.squeeze(-1)  # (N, M)
        assert gate_values.dim() == 2, f"Expected (N,M), got {gate_values.shape}"

        delta = F.relu(gate_values - 1.0)  # (N, M)

        delta_norm_sq = torch.sum(delta * delta, dim=1)  # (N,)
        term1 = alpha * delta_norm_sq.mean()

        delta_norm = torch.sqrt(delta_norm_sq + eps)  # (N,)
        exceed = F.relu(delta - phi_plus)  # (N, M)
        exceed_sum = torch.sum(exceed, dim=1)            # (N,)
        term2 = mu * (delta_norm * exceed_sum).mean()

        return term1 + term2

    def forward(
        self,
        band_components,                # List[M], each (N, D)
        task_emb: torch.Tensor,         # (N, D)
    ):
        """Apply residual gates and global weights, then sum the bands."""
        band_tensor = torch.stack(band_components, dim=1)  # (N, M, D)

        band_gates = 1.0 + self.gate_scale * self.freq_gate(task_emb)  # (N, M)
        ib_loss = self.ib_surrogate_loss_from_gate(
            band_gates,
            alpha=self.ib_alpha,
            mu=self.ib_mu,
            phi_plus=self.ib_phi_plus,
        )

        band_gates = band_gates.unsqueeze(-1)  # (N, M, 1)

        band_weights = torch.sigmoid(self.freq_weights).view(1, self.num_bands, 1)  # (1, M, 1)
        gated_bands = band_gates * band_tensor  # (N, M, D)

        fused_emb = torch.sum(band_weights * gated_bands, dim=1)  # (N, D)
        return fused_emb, ib_loss


def _config_value(config, key, default):
    if key not in config:
        return default
    value = config[key]
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    return value


class BandLevelMask(nn.Module):
    """Randomly mask complete spectral bands during training."""

    def __init__(self, band_num, mask_rate=0.2, rescale=True, ensure_one_band=True):
        super().__init__()
        if not 0.0 <= mask_rate < 1.0:
            raise ValueError("mask_rate must be in [0, 1).")
        self.band_num = int(band_num)
        self.mask_rate = float(mask_rate)
        self.rescale = bool(rescale)
        self.ensure_one_band = bool(ensure_one_band)

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError("BandLevelMask expects (N, M, D).")
        n_nodes, band_num, _ = x.shape
        if band_num != self.band_num:
            raise ValueError("band_num mismatch.")
        keep = (torch.rand(n_nodes, band_num, 1, device=x.device) > self.mask_rate).to(x.dtype)
        if self.ensure_one_band:
            all_zero = keep.sum(dim=1).eq(0).squeeze(-1)
            if all_zero.any():
                keep[torch.where(all_zero)[0], 0, 0] = 1.0
        out = x * keep
        if self.rescale:
            out = out / max(1e-12, 1.0 - self.mask_rate)
        return out


class FrequencyBandModulation(nn.Module):
    """Apply band-level masking to a list or tensor of spectral bands."""

    def __init__(self, band_num, mask_rate=0.2):
        super().__init__()
        self.band_drop = BandLevelMask(band_num, mask_rate=mask_rate, rescale=True, ensure_one_band=True)

    def forward(self, bands):
        is_list = isinstance(bands, (list, tuple))
        band_tensor = torch.stack(list(bands), dim=1) if is_list else bands
        masked = self.band_drop(band_tensor)
        if is_list:
            return [masked[:, i, :] for i in range(masked.size(1))]
        return masked


class SAFIB(SAFIBBackbone):
    """Train and evaluate the complete SAFIB recommendation pipeline."""

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.sbm_weight = float(_config_value(config, "sbm_weight", 0.01))
        self.sbm_mask_rate = float(_config_value(config, "sbm_mask_rate", 0.2))
        self.SBM_Modulator = FrequencyBandModulation(
            self.num_freq_bands,
            mask_rate=self.sbm_mask_rate,
        )
        self._last_sbm_loss = None

    def _zero_loss(self):
        return self.item_id_embedding.weight.new_tensor(0.0)

    def _finite(self, x):
        return torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)

    def _frequency_bands(self, user_rep, item_rep, item_graph_features, user_graph_features):
        """Combine spectral bands from base and graph-enhanced views."""
        op = self.FeqDecomposeOperator
        user_frequencies = op.frequency_decompose_svd_separate(user_rep)
        item_frequencies = op.frequency_decompose_svd_separate(item_rep)
        user_graph_frequencies = op.frequency_decompose_svd_separate(user_graph_features)
        item_graph_frequencies = op.frequency_decompose_svd_separate(item_graph_features)
        user_bands = [user_graph_frequencies[i] + user_frequencies[i] for i in range(op.M)]
        item_bands = [item_graph_frequencies[i] + item_frequencies[i] for i in range(op.M)]
        return user_bands, item_bands

    def _build_safib_representations(self):
        """Build multimodal user and item representations for decomposition."""
        id_rep, id_preference = self.id_gcn(self.edge_index_dropv, self.edge_index, self.item_id_embedding.weight)
        v_rep, v_preference = self.v_gcn(self.edge_index_dropv, self.edge_index, self.image_embedding.weight)
        t_rep, t_preference = self.t_gcn(self.edge_index_dropt, self.edge_index, self.text_embedding.weight)

        representation = torch.cat((id_rep, v_rep, t_rep), dim=1)
        id_user = id_rep.unsqueeze(2)[:self.num_user]
        v_user = v_rep.unsqueeze(2)[:self.num_user]
        t_user = t_rep.unsqueeze(2)[:self.num_user]
        user_rep = torch.cat((id_user, v_user, t_user), dim=2)
        user_rep = self.weight_u.transpose(1, 2) * user_rep
        user_rep = torch.cat((user_rep[:, :, 0], user_rep[:, :, 1], user_rep[:, :, 2]), dim=1)

        item_rep = representation[self.num_user:]
        item_graph_features = item_rep
        for _ in range(self.n_layers):
            item_graph_features = torch.sparse.mm(self.mm_adj, item_graph_features)

        user_graph_features = torch.cat((id_preference, v_preference, t_preference), dim=1)
        return user_rep, item_rep, item_graph_features, user_graph_features

    def _apply_sbm(
        self,
        user_bands,
        item_bands,
        user_rep,
        item_task_emb,
        user_full,
        item_full,
        ib_loss,
        user_indices=None,
        pos_item_indices=None,
        neg_item_indices=None,
    ):
        """Apply band masking and compute consistency loss for the masked view."""
        self._last_sbm_loss = self._zero_loss()
        if not self.training:
            return ib_loss

        masked_user_bands = self.SBM_Modulator(user_bands)
        masked_item_bands = self.SBM_Modulator(item_bands)
        user_masked, ib_user_masked = self.FeqDecomposeOperator.fusion_layer_user(masked_user_bands, user_rep)
        item_masked, ib_item_masked = self.FeqDecomposeOperator.fusion_layer_item(masked_item_bands, item_task_emb)
        if user_indices is not None and pos_item_indices is not None and neg_item_indices is not None:
            user_teacher = user_full.detach()
            item_teacher = item_full.detach()
            sbm_loss = (
                F.mse_loss(user_masked[user_indices], user_teacher[user_indices])
                + F.mse_loss(item_masked[pos_item_indices], item_teacher[pos_item_indices])
                + F.mse_loss(item_masked[neg_item_indices], item_teacher[neg_item_indices])
            )
        else:
            sbm_loss = F.mse_loss(user_masked, user_full.detach()) + F.mse_loss(item_masked, item_full.detach())
        self._last_sbm_loss = torch.nan_to_num(sbm_loss, nan=0.0, posinf=1e4, neginf=0.0)
        ib_loss = ib_loss + ib_user_masked + ib_item_masked
        return ib_loss

    def _sbm_weighted_loss(self):
        if self._last_sbm_loss is None:
            return self._zero_loss()
        return self.sbm_weight * torch.nan_to_num(self._last_sbm_loss, nan=0.0, posinf=1e4, neginf=0.0)

    def forward(self, user_indices=None, pos_item_indices=None, neg_item_indices=None):
        """Build the full SAFIB user and item representations."""
        user_rep, item_rep, item_graph_features, user_graph_features = self._build_safib_representations()
        user_bands, item_bands = self._frequency_bands(user_rep, item_rep, item_graph_features, user_graph_features)
        user_rep_multi, ib_loss_user = self.FeqDecomposeOperator.fusion_layer_user(user_bands, user_rep)
        item_rep_multi, ib_loss_item = self.FeqDecomposeOperator.fusion_layer_item(item_bands, item_rep)
        ib_loss = ib_loss_user + ib_loss_item
        ib_loss = self._apply_sbm(
            user_bands,
            item_bands,
            user_rep,
            item_rep,
            user_rep_multi,
            item_rep_multi,
            ib_loss,
            user_indices=user_indices,
            pos_item_indices=pos_item_indices,
            neg_item_indices=neg_item_indices,
        )
        return user_rep_multi, item_rep_multi, ib_loss, user_bands, item_bands

    def calculate_loss(self, interaction):
        """Compute the recommendation and spectral regularization losses."""
        user_indices = interaction[0]
        pos_item_indices = interaction[1]
        neg_item_indices = interaction[2]

        user_emb_all, item_emb_all, ib_loss, user_frequencies, item_frequencies = self.forward(
            user_indices=user_indices,
            pos_item_indices=pos_item_indices,
            neg_item_indices=neg_item_indices,
        )
        user_emb_all = self._finite(user_emb_all)
        item_emb_all = self._finite(item_emb_all)
        ib_loss = torch.nan_to_num(ib_loss, nan=0.0, posinf=1e4, neginf=0.0)

        user_emb_batch = user_emb_all[user_indices]
        pos_item_emb_batch = item_emb_all[pos_item_indices]
        neg_item_emb_batch = item_emb_all[neg_item_indices]

        if len(user_frequencies) > 0 and len(item_frequencies) > 0 and self.reg_weight > 0:
            user_band_emb_batch = [self._finite(band)[user_indices] for band in user_frequencies]
            item_band_emb_batch = [self._finite(band)[pos_item_indices] for band in item_frequencies]
            cl_loss = torch.nan_to_num(
                self.modality_frequency_contrastive_loss(user_band_emb_batch, item_band_emb_batch),
                nan=0.0,
                posinf=1e4,
                neginf=0.0,
            )
        else:
            cl_loss = self._zero_loss()
        batch_mf_loss = torch.nan_to_num(
            self.cross_entropy_loss(user_emb_batch, pos_item_emb_batch, neg_item_emb_batch),
            nan=0.0,
            posinf=1e4,
            neginf=0.0,
        )
        total_loss = batch_mf_loss + ib_loss * self.ib_weight + cl_loss * self.reg_weight + self._sbm_weighted_loss()
        return torch.nan_to_num(total_loss, nan=0.0, posinf=1e4, neginf=0.0)

    def full_sort_predict(self, interaction):
        user = interaction[0]
        restore_user_e, restore_item_e, _, _, _ = self.forward()
        u_embeddings = restore_user_e[user]
        return torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))

    def post_epoch_processing(self):
        if self._last_sbm_loss is None:
            return None
        try:
            weighted = self._sbm_weighted_loss()
            return (
                "safib losses "
                f"[consistency: {self._last_sbm_loss.detach().item():.4f}, "
                f"weighted: {weighted.detach().item():.6f}, "
                f"mask rate: {self.sbm_mask_rate:.2f}]"
            )
        except Exception:
            return None
