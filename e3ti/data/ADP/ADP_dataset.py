import pickle
# import gzip
import zstandard as zstd
import torch
from torch_geometric.data import Data
from omegaconf import DictConfig
from torch import Tensor
import parmed as pmd
import mdtraj as md
from e3nn.o3 import Irreps
from e3ti.data.dataset import E3TIDataset

class ADPDataset(E3TIDataset):
    """
    TODO: Modify this to store the e3nn irreps of the object as attributes
    """
    def __init__(self,
                 data_dir: str, 
                 data_proc_fname: str, 
                 data_proc_ext: str,
                 data_raw_fname: str, 
                 data_raw_ext: str,
                 split: str,
                 total_frames_train: int,
                 total_frames_test: int,
                 total_frames_valid: int,
                 lag: DictConfig,
                 normalize: DictConfig,
                 node_features: DictConfig,
                 augement_rotations: bool):

        # set filenames for stored data
        self.pdb_file = data_dir+"/alanine-dipeptide-nowater.pdb"
        self.parm_file = data_dir+"/AA.prmtop"

        super().__init__(data_dir, data_proc_fname, data_proc_ext,data_raw_fname,data_raw_ext,
                        split, total_frames_train, total_frames_test, total_frames_valid,
                        lag, normalize, node_features, augement_rotations)
    
    def _preprocess_node_features(self,parm) -> dict[str, Tensor]:
        """
        TODO: Fix indexing praveen mentioned (not an issue for ADP)
        """

        sigmas   = torch.tensor([a.sigma for a in parm.atoms])
        epsilons = torch.tensor([a.epsilon for a in parm.atoms])
        charges  = torch.tensor([a.charge for a in parm.atoms])/0.05487686460574314 # convert to my akma derived
        mass     = torch.tensor([a.mass for a in parm.atoms])
        name     = [a.name for a in parm.atoms]

        uniq = list(dict.fromkeys(name))
        name2int = {n: i for i, n in enumerate(uniq)}
        atom_type = torch.tensor([name2int[n] for n in name], dtype=torch.long)  # encoded sequence
        
        node_f = dict()

        node_f["name"] =  name
        node_f["atom_type"] =  atom_type

        if self.node_features.sigma:
            node_f["sigma"] = sigmas
        
        if self.node_features.epsilon:
            node_f["epsilon"] = epsilons
        
        if self.node_features.charge:
            node_f["charge"] = charges

        if self.node_features.mass:
            node_f["mass"] = mass

        return node_f
    
    def _preprocess(self, raw_path, processed_path):
        """
        Preprocess ADP prmtop and xtc file and save results into a compressed pickle file.

        Args:
            raw_path: Path to the prmtop file containing ADP trajectory.
            processed_path: Path to save the processed data (.pkl.gz).
        """

        parm = pmd.load_file(self.parm_file)
        traj = md.load_xtc(raw_path, self.pdb_file)

        # process node features
        node_feats = self._preprocess_node_features(parm)

        # process trajectory
        # eq dataset for boltzmann derived distribution

        if self.lag.equilibrium:
            frames,mean,std = self._preprocess_traj_equilibrium(traj) # tensor traj is [T,N,3]

            results = []
            for frame in frames:
                processed_frame = self._preprocess_one_equilibrium(frame, node_feats)
                results.append(processed_frame)

        # lagged dataset for fokker-planck derived distribution
        else:
            frames_start, frames_end, mean, std, delta_idx = self._preprocess_traj_nonequilibrium(traj)
            
            results = []
            for frame_start,frame_end,t in zip(frames_start,frames_end, delta_idx):
                processed_frame = self._preprocess_one_nonequilibrium(frame_start, frame_end, t,node_feats)
                results.append(processed_frame)

        return results
        
        self.mean = mean
        self.std = std

        # # Save processed data to a compressed pickle file
        # print(f"Saving {len(results)} processed frame(s) to {processed_path}")
        # with gzip.open(processed_path, 'wb') as f:
        #     pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL) # type: ignore

        # print(f"Saving {len(results)} processed frame(s) to {processed_path}")
        # with open(processed_path, "wb") as raw:
        #     with zstd.ZstdCompressor(level=12).stream_writer(raw) as f:
        #         pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    def summarize_cfg(self) -> None:
        """
        TODO: DO this ??Prints details about the configuration defining the corrector
        """
        raise NotImplementedError

    def __getitem__(self, idx):
        """
        Returns: torch_geometric Data object with following keys:
            - x: [N, 3]
            - x_0: [N, 3] (only if noneq)
            - t: [N] (only if noneq)
            - charge: [N]
            - mass: [N]
            - sigma: [N]
            - epsilon: [N]
        """

        data = self.processed_data[idx]
        
        if self.lag.equilibrium:
            return Data(
                x=data['x'],
                charge=data['charge'],
                atom_type=data['atom_type'],
                mass=data['mass'],
                sigma=data['sigma'],
                epsilon=data['epsilon'],

                keys = ["x","charge","atom_type","mass","sigma","epsilon"],

                name=data['name'],

                x_irrep=Irreps("1o"),
                charge_irrep=Irreps("0e"),
                atom_type_irrep=Irreps("0e"),
                mass_irrep=Irreps("0e"),
                sigma_irrep=Irreps("0e"),
                epsilon_irrep=Irreps("0e"),
            )

        return Data(
                t=data['t'],
                x_0=data['ic'],
                x=data['x'],
                charge=data['charge'],
                atom_type=data['atom_type'],
                mass=data['mass'],
                sigma=data['sigma'],
                epsilon=data['epsilon'],

                keys = ["t","x_0","x","charge","atom_type","mass","sigma","epsilon"],

                name=data['name'],
                
                t_irrep=Irreps("0e"),
                x_0_irrep=Irreps("1o"),
                x_irrep=Irreps("1o"),
                charge_irrep=Irreps("0e"),
                atom_type_irrep=Irreps("0e"),
                mass_irrep=Irreps("0e"),
                sigma_irrep=Irreps("0e"),
                epsilon_irrep=Irreps("0e"),
            )