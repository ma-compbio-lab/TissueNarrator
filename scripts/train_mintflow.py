#!/usr/bin/env python3
"""
MintFlow — training on the Perturb-FISH panel.

Upstream repo : https://github.com/Lotfollahi-lab/mintflow
Upstream guide: docs/notebooks/singleslice_tutorial.ipynb

Training flow: get_default_configurations -> setup_data -> setup_model ->
Trainer -> init_optimisers -> train_one_epoch -> dump_checkpoint (default
configs).

Input: AnnData `mintflow_input.h5ad` — cancer cells labelled
`cancer_<perturbation>` in obs["celltype2"] (the intervention is encoded as a
new cell type), obs["x_um"], obs["y_um"] = xy coords, obs["slice_id"] = section.
"""
import mintflow


def build_configs(input_h5ad, epochs):
    """Populate the four MintFlow config dicts for a single tissue section."""
    cfg_train, cfg_eval, cfg_model, cfg_training = mintflow.get_default_configurations(
        num_tissue_sections_training=1, num_tissue_sections_evaluation=1,
    )
    binding = {                                    # BINDING — our AnnData obs keys
        "file": input_h5ad,
        "obskey_cell_type": "celltype2",
        "obskey_sliceid_to_checkUnique": "slice_id",
        "obskey_x": "x_um",
        "obskey_y": "y_um",
        "obskey_biological_batch_key": "slice_id",
    }
    for cfg in (cfg_train, cfg_eval):
        cfg["list_tissue"]["anndata1"].update(binding)
    cfg_training["flag_enable_wandb"] = "False"
    cfg_training["flag_use_GPU"] = "True"
    cfg_training["num_training_epochs"] = epochs

    cfg_train = mintflow.verify_and_postprocess_config_data_train(cfg_train)
    cfg_eval = mintflow.verify_and_postprocess_config_data_evaluation(cfg_eval)
    cfg_model = mintflow.verify_and_postprocess_config_model(cfg_model, num_tissue_sections=1)
    cfg_training = mintflow.verify_and_postprocess_config_training(cfg_training)
    return {"config_data_train": cfg_train, "config_data_evaluation": cfg_eval,
            "config_model": cfg_model, "config_training": cfg_training}


def train(input_h5ad, out_ckpt, *, epochs=50):
    """Fit MintFlow and dump a checkpoint (tutorial flow, default configs)."""
    cfgs = build_configs(input_h5ad, epochs)

    data_mintflow = mintflow.setup_data(
        cfgs, flag_verbose=True, flag_visualise_tissue_sections=False)
    model = mintflow.setup_model(
        cfgs, data_mintflow, flag_verbose=True, flag_visualise_tissue_sections=False)

    trainer = mintflow.Trainer(cfgs, model, data_mintflow)
    trainer.init_optimisers()
    for _ in range(epochs):
        trainer.train_one_epoch()

    mintflow.dump_checkpoint(model, data_mintflow, cfgs, out_ckpt)


if __name__ == "__main__":
    train("mintflow_input.h5ad", "checkpoint.pt", epochs=50)
