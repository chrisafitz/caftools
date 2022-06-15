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


current_path = os.getcwd()
split_path = current_path.split('/')
base_path = '/'.join(split_path[:-1])
workspace_path = '{}/workspace'.format(base_path)

folder_to_analyze = 'bmim_tfsi_neat'
os.chdir('{}/{}'.format(workspace_path,folder_to_analyze))


### Unwrapping
def unwrap():

    xtc_file = ('sample.xtc')
    gro_file = ('sample.gro')
    tpr_file = ('sample.tpr')
    if os.path.isfile(xtc_file) and os.path.isfile(gro_file):
        os.system('echo 0 | gmx trjconv -f {0} -o {1} -s {2} -skip 10 -pbc nojump'.format(xtc_file, 'sample_unwrapped.xtc', tpr_file))
        unwrapped_trj = ('sample_unwrapped.xtc')
    
        #com_trj = ( 'sample_com.xtc')
        #unwrapped_com_trj = ('sample_com_unwrapped.xtc')
    
        os.system('echo 0 | gmx trjconv -f {0} -o {1} -s {2} -skip 10 -pbc res'.format(xtc_file, 'sample_res.xtc', tpr_file))
        res_trj = ('sample_res.xtc')
        trj = md.load(res_trj, top=gro_file)
        trj = md.load(unwrapped_trj, top=gro_file)
        comtrj = make_comtrj(trj)
        comtrj.save_xtc('sample_com_unwrapped.xtc')
        comtrj[-1].save_gro('com.gro')
        print('make whole')
    
        #whole_trj =  ('sample_whole.xtc')
        #whole_com_trj = ('sample_com_whole.xtc')
    
        os.system('echo 0 | gmx trjconv -f {0} -o {1} -s {2} -skip 10 -pbc whole'.format(xtc_file,'sample_whole.xtc', gro_file))
        whole_trj =  ('sample_whole.xtc')
        trj_whole = md.load(whole_trj, top=gro_file)
        trj_whole_com = make_comtrj(trj_whole)
        print('saving')
        trj_whole_com.save_xtc('sample_com_whole.xtc')
 
 
### Mean Squared Displacement
def msd():

    def _run_overall(trj, mol):
        D, MSD, x_fit, y_fit = calc_msd(trj)
        return D, MSD

    def _save_overall( mol, trj, MSD):
        name = "Christopher_2022"
        np.savetxt( 'msd-{}-overall-{}.txt'.format(mol, name),np.transpose(np.vstack([trj.time, MSD])),header='# Time (ps)\tMSD (nm^2)')
        tempe = 298 #write the temperature
        res = stats.linregress(trj.time, MSD)
        fig, ax = plt.subplots()
        ax.plot(trj.time, MSD)
        ax.plot(trj.time, res.intercept + res.slope*(trj.time), 'r', alpha=0.3, linewidth= 0.8)
        slope = '{:.2e}'.format(res.slope)
        dif_c = '{:.2e}'.format((res.slope)*(1/(1*(10**18)))*(1/6)*(1*(10**12)))
        ax.text(((max(trj.time)/6)*1.5), (max(MSD)/5)*4.5,"Slope: {} nm^2/ps \n Diffussion coef: {} m^2/s \n T:{}K \n ".format(slope,dif_c, tempe) , horizontalalignment='center', verticalalignment = 'center',bbox=dict(facecolor='orange', alpha=0.2))
        ax.set_xlabel('Simulation time (ps)')
        ax.set_ylabel('MSD (nm^2)')
        fig.savefig('msd-{}-overall-{}.pdf'.format(mol,name))
    
    def _run_multiple(trj):
        D_pop = list()
        num_frame = trj.n_frames
        chunk = 5000
        for start_frame in np.linspace(0, num_frame - chunk, num = 200, dtype=int):
            end_frame = start_frame + chunk
            sliced_trj = trj[start_frame:end_frame]
            D_pop.append(calc_msd(sliced_trj)[0])
            D_avg = np.mean(D_pop)
            D_std = np.std(D_pop)
        return D_avg, D_std


    print('Loading trj ')
    top_file = ('sample.gro')
    trj_file = ('sample.xtc')
    trj = md.load(trj_file,top=top_file)
    temp = 298  #write temperature as number  298

    #in selections you need to write the name of your resiudes for example: tfsi and li
    # what the code does, it makes a "new trajectory only with the molecule you are interested"
    selections = {
                        'li': trj.top.select("resname li"),
                        'tfsi': trj.top.select("resname tfsi"),
                        'wat': trj.top.select("resname wat"),
                        'emim': trj.top.select("resname emim"),
                        'bmim': trj.top.select("resname bmim")
                        }


    for mol, indices in selections.items():
        print('\tConsidering {}'.format(mol))
        if indices.size == 0:
            print('{} does not exist in this statepoint'.format(mol))
            continue
        print(mol)
        sliced = trj.atom_slice(indices)
        print("Sliced selection in pore!")
        D, MSD = _run_overall(sliced, mol)
        _save_overall( mol, sliced, MSD)
        ###
       
       
### Radial Distribution Function
def rdf(atom1,atom2,stride=100):         # ex for atom1 -- resname bmim and name N
    
    print('Loading trj ')
    top_file = ('sample.gro')
    trj_file = ('sample.xtc')
    trj = md.load(trj_file, top=top_file, stride = stride)
    print(trj.n_frames)
    #first_frame = trj[0]
    temp = 298 #"write temperature" as number  298

    selections = dict()
    if 'li' in atom1 or 'emim' in atom1 or 'bmim' in atom1:
        selections['cation'] = trj.topology.select(atom1)
    elif 'tfsi' in atom1:
        selections['anion'] = trj.topology.select(atom1)
    elif 'acn' in atom1 or 'wat' in atom1:
        selections['solvent'] = trj.topology.select(atom1)
    if 'li' in atom2 or 'emim' in atom2 or 'bmim' in atom2:
        selections['cation'] = trj.topology.select(atom2)
    elif 'tfsi' in atom2:
        selections['anion'] = trj.topology.select(atom2)
    elif 'acn' in atom2 or 'wat' in atom2:
        selections['solvent'] = trj.topology.select(atom2)
    

    combos = combinations(list(selections.keys()),2)


    for combo in combos:
        if len(selections[combo[0]]) != 0 and len(selections[combo[1]]) != 0:
            fig, ax = plt.subplots()
            print('running rdf between {0} ({1}) \tand\t{2} ({3})\t...'.format(combo[0],
                                                                            len(selections[combo[0]]),
                                                                            combo[1],
                                                                            len(selections[combo[1]])))
            print(combo)
            r, gr = md.compute_rdf(trj, pairs=trj.topology.select_pairs(selections[combo[0]], selections[combo[1]]), r_range=((0.0, 2.0)))
            
            plt.plot(r,gr)
            plt.xlabel('distance (nm)')
            plt.ylabel('g(r)')
            name1 = atom1.split()
            atom1_title = '{}({})'.format(name1[1],name1[-1])
            name2 = atom2.split()
            atom2_title = '{}({})'.format(name2[1],name2[-1])
            fig.suptitle('RDF {} - {}'.format(atom1_title,atom2_title))
            plt.savefig('rdf {} - {}.pdf'.format(atom1_title,atom2_title))
            print('done')
        


### Nernst-Einstein Conductivity
def neconductivity(ion,stride=100):   ### enter ion as ex. 'resname tfsi'

    top_file = ('com.gro')
    trj_file = ('sample_com_unwrapped.xtc')
    trj = md.load(trj_file, top=top_file, stride = stride)
    ion = trj.topology.select(ion)




    D_cat = 2.61e-09
    D_an = 2.61e-09
    T = 298
    q = 0
    N = len(ion)
    V = 125e-27 # m^3



    D_cat *= u.m**2 / u.s
    D_an *= u.m**2 / u.s
    kT = T * 1.3806488e-23 * u.joule
    q *= u.elementary_charge
    q = q.to('Coulomb')
    V *= u.m**3

    cond = N / (V*kT) * q ** 2 * (D_cat + D_an)

    print("         The Nernst-Einstein conductivity is: "+ str(cond))