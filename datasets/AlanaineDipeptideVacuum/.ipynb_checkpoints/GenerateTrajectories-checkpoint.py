import sys
import os
current_dir = os.path.dirname(os.curdir)
parent_dir = os.path.abspath(os.path.join(current_dir, "../../../SPARK"))
sys.path.append(parent_dir)
import parmed as pmd
import torch
import numpy as np
from system.topology import Topology
from forces.twobody import MixtureLJ, Harmonic_Bond, MixtureCoulomb
from forces.threebody import Harmonic_Angle
from forces.fourbody import Dihedral
import system.units as units
import system.box as box
import system.system as sys
from integrators.NVE import NVE
from integrators.NVT import NVT
from integrators.ParallelTempering import ParallelTempering
import matplotlib.pyplot as plt
import math
from tqdm import trange
from utils import *
from scipy.ndimage import gaussian_filter
dtype=torch.float32
device="cuda"

top, node_features, mass, energy_dict = build_top_and_features("alanine-dipeptide.prmtop")
B = 64
pos = torch.tensor(pmd.load_file("alanine-dipeptide.pdb").coordinates,dtype=dtype,device=device).unsqueeze(0).expand(B, -1, -1).contiguous()
atomic_numbers = [a.atomic_number for a in pmd.load_file("alanine-dipeptide.pdb").atoms]
b = box.Box([1000,1000,1000],["s","s","s"])
u = units.UnitSystem.akma()
mom = 0.5*torch.randn_like(pos)

S = sys.System(pos, mom, mass, top, b, energy_dict, u, node_features)
S._potential_energy(pos)
S.compile_force_fn()
S.pos = S.pos - (S.mass.unsqueeze(-1) * S.pos).sum(dim=1, keepdim=True) / S.mass.sum(dim=1, keepdim=True).unsqueeze(-1)

integrator = NVT(0.005, 20, 10_000)
print(integrator)

steps = 20_000
with torch.no_grad():
    for i in trange(steps, desc=f"Running simulation at T = {integrator.T} to span phase space"):
        integrator.step(S)
        S.pos = S.pos - (S.mass.unsqueeze(-1) * S.pos).sum(dim=1, keepdim=True) / S.mass.sum(dim=1, keepdim=True).unsqueeze(-1)


integrator = NVT(0.005, 20, 298)
print(integrator)

steps = 512000
skip = 10

positions = torch.empty((int(steps / skip), B, 22, 3), device=device)
forces = torch.empty((int(steps / skip), B, 22, 3), device=device)
momenta = torch.empty((int(steps / skip), B, 22, 3), device=device)
energy = torch.empty((int(steps / skip), B), device=device)

with torch.no_grad():
    for i in trange(steps, desc=f"Running simulation at T = {298} for sampling"):
        # Save traj here
        if i % skip == 0:
            positions[int(i / skip)] = S.pos
            momenta[int(i / skip)] = S.mom
            energy[int(i / skip)] = S.kinetic_energy() + S.potential_energy()
            forces[int(i / skip)] = S.force()
            
        integrator.step(S)
        S.pos = S.pos - (S.mass.unsqueeze(-1) * S.pos).sum(dim=1, keepdim=True) / S.mass.sum(dim=1, keepdim=True).unsqueeze(-1)
        S.reset_cache()

name = "ADP_Vacuum"

torch.save(positions, name+"_position_long")
torch.save(momenta, name+"_momentum_long")
# torch.save(forces, name+"_force")
# torch.save(energy, name+"_energy")

