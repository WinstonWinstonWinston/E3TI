import json
import os
import sys
# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
# print(project_root)
sys.path.insert(0, project_root)

import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
from deeptime import decomposition
from data.utils import (
    get_ala2_top,
)
import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf
from hydra.utils import instantiate
from hydra.utils import get_class

@hydra.main(config_path="../configs", config_name="base", version_base=None)
def main(cfg: DictConfig):
    
    analyse_cfg = cfg.analysis
#     args.root = os.path.realpath(args.root)
    topology = get_ala2_top(cfg.data.path)
#     ckpt_epoch = args.ckpt_epoch
    analysis_dir = os.path.join(analyse_cfg.root, analyse_cfg.analysis_dir)
    os.makedirs(analysis_dir, exist_ok=True)

    trajs = np.load(analyse_cfg.trajs)
#     print(np.var(trajs, axis=0))
#     print(trajs.shape, trajs[0,:3,:,:])
    vamp2_score = get_vamp2(trajs=trajs, topology=topology, lag=1)

    ## Get reference trajs
    filenames = [os.path.join(cfg.data.path, f"alanine-dipeptide-{i}-250ns-nowater.xtc") for i in range(3)]
#     print(filenames)
    topology = os.path.join(cfg.data.path, "alanine-dipeptide-nowater.pdb")
    topology = md.load(topology).topology
    ref_trajs = [md.load_xtc(fn, topology) for fn in filenames]
    ref_trajs = [t.center_coordinates().xyz for t in ref_trajs]
    
#     ref_trajs = data.get_ala2_trajs(args.root)
    ref_vamp2_score = get_vamp2(ref_trajs, topology=topology, lag=analyse_cfg.lag)
    json.dump(
        {"vamp2": vamp2_score, "ref_vamp2": ref_vamp2_score},
        open(os.path.join(analysis_dir, "vamp2_scores.json"), "w"),
        indent=4,
    )

    print(f"VAMP2 score: {vamp2_score}")
    print(f"Reference VAMP2 score: {ref_vamp2_score}")

    phi, psi = compute_dihedral_angles(np.concatenate(trajs), topology)
    phi_ref, psi_ref = compute_dihedral_angles(np.concatenate(ref_trajs), topology)

    _, ax = plt.subplots(1, 2, figsize=(10, 4))

    plot_marginal(ax[0], phi)
    plot_marginal(ax[0], phi_ref, label="MD")
    ax[0].set_xlabel("Phi")
    ax[0].set_ylabel("Frequency")
    if not analyse_cfg.no_plot_start:
        ax[0].vlines(phi[0], 0, 1, linestyle="--", color="k")

    plot_marginal(ax[1], psi, label="ITO")
    plot_marginal(ax[1], psi_ref, label="MD")
    ax[1].set_xlabel("Psi")
    if not analyse_cfg.no_plot_start:
        ax[1].vlines(psi[0], 0, 1, linestyle="--", color="k", label="Start")

    ax[1].legend()
    plt.savefig(os.path.join(analysis_dir, "marginals.pdf"))

    _, ax = plt.subplots(figsize=(5, 4))
    plt.plot(phi, psi, ".", alpha=0.1, label="ITO")
    ax.set_xlabel("Phi")
    ax.set_ylabel("Psi")
    if not analyse_cfg.no_plot_start:
        plt.plot(phi[0], psi[0], "k", marker="x", label="Start")

    ax.legend()
    plt.savefig(os.path.join(analysis_dir, "ramachandran.pdf"))
    plt.show()


def plot_marginal(ax, marginal, bins=64, label=None):
    bin_heights, _ = np.histogram(
        marginal, range=(-np.pi, np.pi), bins=bins, density=True
    )
    plot_marginal_dist(ax, bin_heights, label=label, linestyle="-")


def plot_marginal_dist(ax, bin_heights, linestyle="-", c=None, label=None):
    bin_edges = np.linspace(-np.pi, np.pi, len(bin_heights) + 1)
    bin_widths = np.diff(bin_edges)
    bin_heights /= bin_heights.mean() * bin_widths.sum()
    bin_heights = np.append(bin_heights, bin_heights[-1])

    ax.step(
        bin_edges,
        bin_heights,
        where="post",
        linestyle=linestyle,
        c=c,
        label=label,
    )
    ax.semilogy()


def compute_dihedral_angles(traj, topology):
    traj = md.Trajectory(xyz=traj, topology=topology)
    phi_atoms_idx = [4, 6, 8, 14]
    phi = md.compute_dihedrals(traj, indices=[phi_atoms_idx])[:, 0]
    psi_atoms_idx = [6, 8, 14, 16]
    psi = md.compute_dihedrals(traj, indices=[psi_atoms_idx])[:, 0]

    return phi, psi


def featurize_trajs(trajs, topology):
    featurized_trajs = np.stack([featurize_traj(traj, topology) for traj in trajs])
    nan_mask = np.isnan(featurized_trajs).any(axis=(1, 2))

    return featurized_trajs[~nan_mask]


def featurize_traj(traj, topology):
    phi, psi = compute_dihedral_angles(traj, topology)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    cos_psi = np.cos(psi)
    sin_psi = np.sin(psi)

    features = np.stack([cos_phi, sin_phi, cos_psi, sin_psi]).T
    return features


def get_vamp2(trajs, lag, topology):
    featurized_trajs = featurize_trajs(trajs, topology)
    vamp = decomposition.VAMP(lag).fit_fetch(featurized_trajs)
    vamp2_score = vamp.score(2)

    return vamp2_score


if __name__ == "__main__":
    main()
