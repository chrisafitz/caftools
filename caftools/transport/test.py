from itertools import combinations
import os 
import mbuild as mb
import numpy as np 
import matplotlib.pyplot as plt
import mdtraj as md
import MDAnalysis as mda
from mtools.gromacs.gromacs import make_comtrj
from mtools.gromacs.gromacs import unwrap_trj
from mtools.post_process import calc_msd
from mtools.post_process import compute_cn
import unyt as u
from scipy import stats
from mtools.post_process import calc_density
import time
#from scattering.utils.utils import get_dt, get_unique_atoms
from progressbar import ProgressBar
import warnings
import scipy.spatial as ss
from mdtraj.core.element import virtual_site



current_path = os.getcwd()
split_path = current_path.split('/')
home_path = '/'.join(split_path[:-2])
base_path = '{}/project'.format(home_path)
workspace_path = '{}/workspace'.format(base_path)


def unwrap():

    def make_comtrj(trj):
        """Takes a trj and returns a trj with COM positions as atoms"""

        comtop = md.Topology()
        coords = np.ndarray(shape=(trj.n_frames, trj.n_residues, 3))

        for j, res in enumerate(trj.topology.residues):
            comtop.add_atom(res.name, virtual_site, comtop.add_residue(res.name, comtop.add_chain()))
            res_frame = trj.atom_slice([at.index for at in res.atoms])
            coords[:, j, :] = md.compute_center_of_mass(res_frame)

        comtrj = md.Trajectory(xyz=coords,
                            topology=comtop,
                            time=trj.time,
                            unitcell_angles=trj.unitcell_angles,
                            unitcell_lengths=trj.unitcell_lengths)

        return comtrj

    xtc_file = ('sample.xtc')
    gro_file = ('sample.gro')
    tpr_file = ('sample.tpr')
    if os.path.isfile(xtc_file) and os.path.isfile(gro_file):
        print('huh')
        os.system('echo 0 | gmx trjconv -f {0} -o {1} -s {2} -skip 10 -pbc nojump'.format(xtc_file, 'sample_unwrapped.xtc', tpr_file))
        unwrapped_trj = ('sample_unwrapped.xtc')
        
        os.system('echo 0 | gmx trjconv -f {0} -o {1} -s {2} -skip 10 -pbc res'.format(xtc_file, 'sample_res.xtc', tpr_file))
        res_trj = ('sample_res.xtc')
        
        trj1 = md.load(res_trj, top=gro_file)
        trj2= md.load(unwrapped_trj, top=gro_file)
        comtrj = make_comtrj(trj2)
        print(comtrj)
        comtrj.save_xtc('sample_com_unwrapped.xtc')
        comtrj[-1].save_gro('comtest.gro')
        print('make whole')
        
        os.system('echo 0 | gmx trjconv -f {0} -o {1} -s {2} -skip 10 -pbc whole'.format(xtc_file,'sample_whole.xtc', gro_file))
        whole_trj =  ('sample_whole.xtc')
        trj_whole = md.load(whole_trj, top=gro_file)
        trj_whole_com = make_comtrj(trj_whole)
        trj_whole_com[-1].save_gro('comtest.gro')
        print('make whole')
        print('saving')
        trj_whole_com.save_xtc('sample_com_whole.xtc')
        print(trj_whole_com)




os.system('cd /home/chrisfitz/project/workspace/emim_tfsi_neat')



unwrap()
