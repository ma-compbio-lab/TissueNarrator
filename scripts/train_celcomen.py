#!/usr/bin/env python3
"""
Celcomen — training on the Perturb-FISH 154-gene panel.

Upstream repo : https://github.com/Teichlab/celcomen
Upstream guide: Tutorial_Celcomen_on_Xenium.ipynb (Xenium GBM example)

Training loop = celcomen.training_plan.train.train (lr=1e-1, zmft=1e-1,
epochs=200, kNN(6) spatial graph). At inference, we drop the `log_z_mft`
partition term from the SimComen SGD because we found that with it the
counterfactual response is sign-inverted on Perturb-FISH,
whereas dropping it recovers the correct sign.

Input: AnnData `celcomen_input.h5ad` — X = log-normalised expression
(n_cells x 154 genes), obs["sample"] = section id, obsm["spatial"] = xy coords.
"""
import anndata as ad
import numpy as np
import torch

from celcomen.models.celcomen import celcomen as Celcomen
from celcomen.training_plan.train import train as celcomen_train
from celcomen.utils.helpers import normalize_g2g
from celcomen_dataloader import get_dataset_loaders_sparse


def train(input_h5ad, out_pt, *, epochs=200, lr=1e-1, zmft=1e-1,
          n_neighbors=6, seed=0, device="cuda"):
    """Fit Celcomen's two gene-gene coupling matrices (intra + inter)."""
    adata = ad.read_h5ad(input_h5ad)        # log-normalised counts; obs["sample"]
    n = adata.n_vars

    # kNN(6) spatial graph — identical to the tutorial's dataloader.
    loader = get_dataset_loaders_sparse(
        h5ad_path=input_h5ad, sample_id_name="sample",
        n_neighbors=n_neighbors, distance=None, device=device, verbose=True,
    )

    # Symmetric, normalised random init for both coupling matrices.
    model = Celcomen(input_dim=n, output_dim=n, n_neighbors=n_neighbors, seed=seed)
    rng = np.random.default_rng(seed)
    g2g_init = rng.uniform(size=(n, n)).astype("float32")
    g2g_init = normalize_g2g((g2g_init + g2g_init.T) / 2)
    model.set_g2g(torch.from_numpy(g2g_init))        # intercellular
    model.set_g2g_intra(torch.from_numpy(g2g_init))  # intracellular
    model.to(device)

    losses = celcomen_train(epochs, lr, model, loader,
                            zmft_scalar=zmft, seed=seed, device=device)

    torch.save({"g2g_inter": model.conv1.lin.weight.detach().cpu(),
                "g2g_intra": model.lin.weight.detach().cpu(),
                "n": n, "n_neighbors": n_neighbors}, out_pt)
    return losses


if __name__ == "__main__":
    train("celcomen_input.h5ad", "celcomen_model.pt", epochs=200)
