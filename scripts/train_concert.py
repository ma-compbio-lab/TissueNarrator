#!/usr/bin/env python3
"""
CONCERT — training on the Perturb-FISH 154-gene panel.

Upstream repo : https://github.com/mims-harvard/CONCERT
Upstream guide: notebooks/run_concert_perturbMap.ipynb
                (script form: src/run_concert_map.py)

Input: per-cell arrays from the Perturb-FISH HDF5 —
  X       (n_cells, 154) integer counts
  pos_raw (n_cells, 2)   xy coordinates
  tissue  (n_cells,)     tissue label       ('cancer' / 'T cells' / ...)
  perturb (n_cells,)     perturbation label ('Control' / 'MAP2K2' / ...)
"""
import numpy as np
import scanpy as sc
import torch
from sklearn.preprocessing import MinMaxScaler

# Vendored upstream CONCERT.
from concert_map import CONCERT
from preprocess import normalize
from run_concert_map import auto_batch_size, build_inducing_points


def build_attributes(perturb, tissue):
    tissue_dict = {t: i for i, t in enumerate(sorted(set(tissue)))}
    tissue_idx = np.array([tissue_dict[t] for t in tissue], dtype=int)
    bg = {"NA", "Control"}
    real = sorted(p for p in set(perturb) if p not in bg)
    pert_map = {**{b: 0 for b in bg}, **{p: i + 1 for i, p in enumerate(real)}}
    perturb_idx = np.array([pert_map.get(p, 0) for p in perturb], dtype=int)
    return tissue_idx, perturb_idx, tissue_dict, pert_map


def train(X, pos_raw, tissue, perturb, out_pt, *, maxiter=5000, patience=200,
          lr=1e-4, weight_decay=1e-6, loc_range=20.0, kernel_scale=10.0, seed=0):
    """Fit CONCERT on the full multi-perturbation dataset.

    X        : (n_cells, n_genes) integer counts
    pos_raw  : (n_cells, 2) spatial coordinates
    tissue   : (n_cells,) tissue label per cell (e.g. 'cancer' / 'T cells')
    perturb  : (n_cells,) perturbation label per cell (e.g. 'Control' / 'MAP2K2')
    """
    np.random.seed(seed); torch.manual_seed(seed)
    tissue_idx, perturb_idx, tissue_dict, pert_map = build_attributes(perturb, tissue)

    # size-factor + log + scale (upstream preprocess.normalize).
    adata = sc.AnnData(X.copy(), dtype="float32")
    adata = normalize(adata, size_factors=True, normalize_input=True, logtrans_input=True)

    cell_atts = np.c_[tissue_idx, perturb_idx].astype(int)
    n_batch = int(len(np.unique(perturb_idx)))
    batch = np.eye(n_batch, dtype=np.float32)[perturb_idx]

    # Scale coords and append the perturbation one-hot for kernel conditioning.
    pos_scaled = MinMaxScaler().fit_transform(pos_raw) * loc_range
    pos_batched = np.concatenate([pos_scaled, batch], axis=1).astype(np.float32)

    inducing = build_inducing_points(pos_batched=pos_batched, n_batch=n_batch,
                                     steps=6, loc_range=loc_range, grid=True, k_clusters=None)
    spatial_dims = pos_batched.shape[1] - n_batch
    kernel_scale_arr = np.full((n_batch, spatial_dims), kernel_scale, dtype=np.float32)
    cutoff = np.full(pos_scaled.shape[0], 0.5, dtype=np.float32)

    model = CONCERT(
        cell_atts=cell_atts, num_genes=adata.n_vars,
        encoder_dim=256, GP_dim=2, Normal_dim=8, n_batch=n_batch,
        encoder_layers=[128, 64], decoder_layers=[128],
        noise=0.0, encoder_dropout=0.0, decoder_dropout=0.0,
        shared_dispersion=False, fixed_inducing_points=True,
        initial_inducing_points=inducing, fixed_gp_params=False,
        kernel_scale=kernel_scale_arr, multi_kernel_mode=True,
        N_train=adata.n_obs,
        KL_loss=0.5, dynamicVAE=True,                     
        init_beta=1.0, min_beta=0.5, max_beta=5.0,        
        mask_cutoff=cutoff, dtype=torch.float32, device="cuda",
    )

    model.train_model(
        pos=pos_batched, ncounts=adata.X, raw_counts=adata.raw.X,
        size_factors=adata.obs.size_factors, batch=batch,
        lr=lr, weight_decay=weight_decay, batch_size=auto_batch_size(X.shape[0]),
        num_samples=1, train_size=0.95, maxiter=maxiter, patience=patience,
        save_model=True, model_weights=out_pt,
    )


if __name__ == "__main__":
    train(X=None, pos_raw=None, tissue=None, perturb=None, out_pt="model_pertfish.pt")
