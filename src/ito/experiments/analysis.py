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
from matplotlib.backends.backend_pdf import PdfPages
from typing import List, Union
from itertools import combinations


TrajsIn = Union[np.ndarray, List[np.ndarray], md.Trajectory, List[md.Trajectory]]

@hydra.main(config_path="../configs", config_name="base", version_base=None)
def main(cfg: DictConfig):
    
    analyse_cfg = cfg.analysis
    topology = get_ala2_top(cfg.data.path)
    analysis_dir = os.path.join(analyse_cfg.root, analyse_cfg.analysis_dir)
    os.makedirs(analysis_dir, exist_ok=True)

    # VAMP2 scores
    trajs = np.load(analyse_cfg.trajs)
    std = 0.1661689
    trajs = trajs*std

    ## Get reference trajs
    filenames = [os.path.join(cfg.data.path, f"alanine-dipeptide-{i}-250ns-nowater.xtc") for i in range(3)]
    topology = os.path.join(cfg.data.path, "alanine-dipeptide-nowater.pdb")
    topology = md.load(topology).topology
    ref_trajs = [md.load_xtc(fn, topology) for fn in filenames]
    ref_trajs = [t.center_coordinates().xyz for t in ref_trajs]
    
#     print(len(ref_trajs))
#     stop
    
    if analyse_cfg.VAMP2 is True:
        vamp2_score = get_vamp2(trajs=trajs, topology=topology, lag=1)
        ref_vamp2_score = get_vamp2(ref_trajs, topology=topology, lag=analyse_cfg.lag)
        json.dump(
            {"vamp2": vamp2_score, "ref_vamp2": ref_vamp2_score},
            open(os.path.join(analysis_dir, "vamp2_scores.json"), "w"),
            indent=4,
        )

        print(f"VAMP2 score: {vamp2_score}")
        print(f"Reference VAMP2 score: {ref_vamp2_score}")
    
    # Marginals of phi and psi

    phi, psi = compute_dihedral_angles(np.concatenate(trajs), topology)
    phi_ref, psi_ref = compute_dihedral_angles(np.concatenate(ref_trajs), topology)
    
    if analyse_cfg.Marginals is True:
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
    
    # Get evolution of marginals with sampling
    
    if analyse_cfg.marginal_evol:
        if "marg_evol_part" in analyse_cfg:
            n_parts = analyse_cfg.marg_evol_part
            chunk = len(phi) // n_parts               # integer size

            for i in range(n_parts):
                end = (i + 1) * chunk if i < n_parts - 1 else len(phi)
                _, ax = plt.subplots(1, 2, figsize=(10, 4))

                plot_marginal(ax[0], phi[:end])
                plot_marginal(ax[0], phi_ref, label="MD")
                ax[0].set_xlabel("Phi")
                ax[0].set_ylabel("Frequency")
                ax[0].set_title(f'Phi Marginal - {end} samples')
                if not analyse_cfg.no_plot_start:
                    ax[0].vlines(phi[0], 0, 1, linestyle="--", color="k")

                plot_marginal(ax[1], psi[:end], label="ITO")
                plot_marginal(ax[1], psi_ref, label="MD")
                ax[1].set_xlabel("Psi")
                ax[1].set_title(f'Psi Marginal - {end} samples')

                if not analyse_cfg.no_plot_start:
                    ax[1].vlines(psi[0], 0, 1, linestyle="--", color="k", label="Start")

                ax[1].legend()
                plt.savefig(os.path.join(analysis_dir, f"marginals_{end}samples.pdf"))
        
        if "marg_evol_samples" in analyse_cfg:
            for end in analyse_cfg.marg_evol_samples:
                _, ax = plt.subplots(1, 2, figsize=(10, 4))
                plot_marginal(ax[0], phi[:end])
                plot_marginal(ax[0], phi_ref, label="MD")
                ax[0].set_xlabel("Phi")
                ax[0].set_ylabel("Frequency")
                ax[0].set_title(f'Phi Marginal - {end} samples')
                if not analyse_cfg.no_plot_start:
                    ax[0].vlines(phi[0], 0, 1, linestyle="--", color="k")

                plot_marginal(ax[1], psi[:end], label="ITO")
                plot_marginal(ax[1], psi_ref, label="MD")
                ax[1].set_xlabel("Psi")
                ax[1].set_title(f'Psi Marginal - {end} samples')

                if not analyse_cfg.no_plot_start:
                    ax[1].vlines(psi[0], 0, 1, linestyle="--", color="k", label="Start")

                ax[1].legend()
                plt.savefig(os.path.join(analysis_dir, f"marginals_{end}samples.pdf"))
    
    # Ramachandran Plots
    
    if analyse_cfg.Ramachandran is True:
        _, ax = plt.subplots(figsize=(5, 4))
        plt.plot(phi, psi, ".", alpha=0.1, label="ITO")
        ax.set_xlabel("Phi")
        ax.set_ylabel("Psi")
        if not analyse_cfg.no_plot_start:
            plt.plot(phi[0], psi[0], "k", marker="x", label="Start")

        ax.legend()
        plt.savefig(os.path.join(analysis_dir, "ramachandran.pdf"))
        plt.show()

    # Visualize trajectories
    
    if analyse_cfg.gen_visual is True:
        xtc_out = analyse_cfg.xtc_name
        gro_out = analyse_cfg.gro_name
        std = 0.1661689
        ## Load numpy coords (T, N, 3)
        xyz = np.load(analyse_cfg.trajs)            # (n_frames, n_atoms, 3) in nm
        if xyz.ndim ==4:
            n_traj, T, N, _ = xyz.shape
            xyz = xyz.reshape(n_traj * T, N, 3)  # keeps traj-major, then time order
        ## Scale the traj back to original size
        xyz = xyz*std
        ## Get topology
        top = get_ala2_top(cfg.data.path)
        traj = md.Trajectory(xyz=xyz, topology=top)
        ## Save to XTC
        xtcout_path = os.path.join(analysis_dir, xtc_out)
        groout_path = os.path.join(analysis_dir, gro_out)        
        traj.save(xtcout_path)
        traj[-1].save_gro(groout_path)
        print(f"Wrote {xtcout_path} and {groout_path}")
        
    # Get bond distributions
    
    if analyse_cfg.bond_dist is True:
        # Load topology
        top = get_ala2_top(cfg.data.path)

        # Normalize and concatenate
        samp = _concat(_to_list(trajs))
        ref  = _concat(_to_list(ref_trajs))

        if analyse_cfg.max_frames is not None:
            max_frames = analyse_cfg.max_frames
            samp = samp[:max_frames]
#             ref  = ref [:max_frames]

        # Build trajectories
        t_s = md.Trajectory(xyz=samp, topology=top)
        t_r = md.Trajectory(xyz=ref,  topology=top)

        # All bonds from topology
        bonds = [(a.index, b.index) for a, b in top.bonds]

        # Compute distances (nm), shape: (T, n_bonds)
        d_s = md.compute_distances(t_s, bonds)
        d_r = md.compute_distances(t_r, bonds)

        bondout_path = os.path.join(analysis_dir, "bond_dist.pdf")
        with PdfPages(bondout_path) as pdf:
            for bi, (i, j) in enumerate(bonds):
                ai, aj = top.atom(i), top.atom(j)
                title = f"Bond: {_atom_label(ai)} — {_atom_label(aj)}"

                plt.figure(figsize=(7, 4))
                plt.hist(d_r[:, bi], bins=50, density=True, alpha=0.6, label="Ref (MD)")
                plt.hist(d_s[:, bi], bins=50, density=True, alpha=0.6, label="ITO")
                plt.xlabel("Distance (nm)")
                plt.ylabel("Density")
                plt.title(title)
                plt.legend()
                plt.tight_layout()
                pdf.savefig()
                plt.close()

        print(f"Saved bond distributions to {bondout_path}")
        
    # Get bond time series
    
    if analyse_cfg.bond_time_series is True:
        # Load topology
        top = get_ala2_top(cfg.data.path)

        # Normalize and concatenate
        samp = _concat(_to_list(trajs))
        ref  = _concat(_to_list(ref_trajs))

        if analyse_cfg.max_frames is not None:
            max_frames = analyse_cfg.max_frames
            samp = samp[:max_frames]
#             ref  = ref [:max_frames]

        # Build trajectories
        t_s = md.Trajectory(xyz=samp, topology=top)
        t_r = md.Trajectory(xyz=ref,  topology=top)

        # All bonds from topology
        bonds = [(a.index, b.index) for a, b in top.bonds]

        # Compute distances (nm), shape: (T, n_bonds)
        d_s = md.compute_distances(t_s, bonds)
        d_r = md.compute_distances(t_r, bonds)
        
        # Time lag assignment
        lag = analyse_cfg.lag
        idx = np.arange(0, d_s.shape[0])
        t_ps_vec = idx * lag

        
        bondtsout_path = os.path.join(analysis_dir, "bond_time_series.pdf")
        with PdfPages(bondtsout_path) as pdf:
            for bi, (i, j) in enumerate(bonds):
                ai, aj = top.atom(i), top.atom(j)
                title = f"Bond length vs time: {_atom_label(ai)} — {_atom_label(aj)}"
                ys = d_s[:, bi]
                yr = d_r[idx*100, bi]
                plt.figure(figsize=(8, 4))
                plt.plot(t_ps_vec, yr, lw=1.0, alpha=1, label="Ref (MD)")
                plt.plot(t_ps_vec, ys, lw=1.0, alpha=0.7, label="ITO")
                plt.xlabel("Time (ps)")
                plt.ylabel("Distance (nm)")
                plt.title(title)
                plt.legend()
                plt.tight_layout()
                pdf.savefig()
                plt.close()

        print(f"Saved bond time series to {bondtsout_path}")
        
    ## Get angle distributions

    if analyse_cfg.angle_dist is True:
        
        # Load topology
        top = get_ala2_top(cfg.data.path)

        # Normalize and concatenate
        samp = _concat(_to_list(trajs))
        ref  = _concat(_to_list(ref_trajs))

        if analyse_cfg.max_frames is not None:
            max_frames = analyse_cfg.max_frames
            samp = samp[:max_frames]
#             ref  = ref [:max_frames]

        # Build trajectories
        t_s = md.Trajectory(xyz=samp, topology=top)
        t_r = md.Trajectory(xyz=ref,  topology=top)

        # All angles from topology
        angles = _angle_triplets_from_topology(top)
        
        # Compute angles (radians) and convert to degrees
        a_s = np.degrees(md.compute_angles(t_s, angles))   # (T, n_angles)
        a_r = np.degrees(md.compute_angles(t_r, angles))
        
        angleout_path = os.path.join(analysis_dir, "angle_dist.pdf")
        # Write all histograms into one PDF
        with PdfPages(angleout_path) as pdf:
            for ai, (i, j, k) in enumerate(angles):
                at_i, at_j, at_k = top.atom(i), top.atom(j), top.atom(k)
                title = f"Angle: {_atom_label(at_i)} — {_atom_label(at_j)} — {_atom_label(at_k)}"

                plt.figure(figsize=(7, 4))
                plt.hist(a_r[:, ai], bins=50, density=True, alpha=0.6, label="Ref (MD)")
                plt.hist(a_s[:, ai], bins=50, density=True, alpha=0.6, label="ITO")
                plt.xlabel("Angle (degrees)")
                plt.ylabel("Density")
                plt.title(title)
                plt.legend()
                plt.tight_layout()
                pdf.savefig()
                plt.close()

        print(f"Saved angle distributions PDF to {angleout_path}")

    ## Get angle distributions

    if analyse_cfg.angle_time_series is True:
        
        # Load topology
        top = get_ala2_top(cfg.data.path)

        # Normalize and concatenate
        samp = _concat(_to_list(trajs))
        ref  = _concat(_to_list(ref_trajs))

        if analyse_cfg.max_frames is not None:
            max_frames = analyse_cfg.max_frames
            samp = samp[:max_frames]
#             ref  = ref [:max_frames]

        # Build trajectories
        t_s = md.Trajectory(xyz=samp, topology=top)
        t_r = md.Trajectory(xyz=ref,  topology=top)

        # All angles from topology
        angles = _angle_triplets_from_topology(top)
        
        # Compute angles (radians) and convert to degrees
        a_s = np.degrees(md.compute_angles(t_s, angles))   # (T, n_angles)
        a_r = np.degrees(md.compute_angles(t_r, angles))
        
        # Time lag assignment
        lag = analyse_cfg.lag
        idx = np.arange(0, a_s.shape[0])
        t_ps_vec = idx * lag

        
        angletsout_path = os.path.join(analysis_dir, "angle_time_series.pdf")
        # Write all histograms into one PDF
        with PdfPages(angletsout_path) as pdf:
            for ai, (i, j, k) in enumerate(angles):
                at_i, at_j, at_k = top.atom(i), top.atom(j), top.atom(k)
                title = f"Angle vs time: {_atom_label(at_i)} — {_atom_label(at_j)} — {_atom_label(at_k)}"
                
                ys = a_s[:, ai]
                yr = a_r[idx*100, ai]
                plt.figure(figsize=(8, 4))
                plt.plot(t_ps_vec, yr, lw=1.0, alpha=1, label="Ref (MD)")
                plt.plot(t_ps_vec, ys, lw=1.0, alpha=0.7, label="ITO")
                plt.ylabel("Angle (degrees)")
                plt.xlabel("Time (ps)")
                plt.title(title)
                plt.legend()
                plt.tight_layout()
                pdf.savefig()
                plt.close()

        print(f"Saved angle time series PDF to {angletsout_path}")
        
### Functions to help with analysis
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

def _to_list(trajs: TrajsIn) -> List[np.ndarray]:
    """Return list of (T_i, N, 3) arrays."""
    if isinstance(trajs, md.Trajectory):
        return [trajs.xyz]
    if isinstance(trajs, list):
        out = []
        for t in trajs:
            out.append(t.xyz if isinstance(t, md.Trajectory) else np.asarray(t))
        return out
    arr = np.asarray(trajs)
    if arr.ndim == 3:
        return [arr]
    if arr.ndim == 4:
        return [arr[i] for i in range(arr.shape[0])]
    raise ValueError(f"Unsupported shape: {arr.shape}")

def _concat(list_xyz: List[np.ndarray]) -> np.ndarray:
    return np.concatenate(list_xyz, axis=0)

def _atom_label(atom: md.core.topology.Atom) -> str:
    return f"{atom.residue.name}{atom.residue.index}:{atom.name}({atom.index})"
    
def _angle_triplets_from_topology(top: md.Topology):
    """All unique (i, j, k) with j as vertex and i,k bonded to j."""
    nbrs = {a.index: set() for a in top.atoms}
    for a, b in top.bonds:
        nbrs[a.index].add(b.index)
        nbrs[b.index].add(a.index)
    triplets = []
    for j in nbrs:
        for i, k in combinations(sorted(nbrs[j]), 2):
            triplets.append((i, j, k))
    return triplets

if __name__ == "__main__":
    main()
