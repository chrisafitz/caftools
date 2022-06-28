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

#folder_to_analyze = input("List the folder to analyze: ")
#os.chdir('{}/{}'.format(workspace_path,folder_to_analyze))


### Unwrapping
def unwrap():

    def make_comtrj(trj):
        """Takes a trj and returns a trj with COM positions as atoms"""

        comtop = md.Topology()
        coords = np.ndarray(shape=(trj.n_frames, trj.n_residues, 3))

        for j, res in enumerate(trj.topology.residues):
            print(j)
            print(res)
            
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
        os.system('echo 0 | gmx trjconv -f {0} -o {1} -s {2} -skip 10 -pbc nojump'.format(xtc_file, 'sample_unwrapped.xtc', tpr_file))
        unwrapped_trj = ('sample_unwrapped.xtc')
        
        os.system('echo 0 | gmx trjconv -f {0} -o {1} -s {2} -skip 10 -pbc res'.format(xtc_file, 'sample_res.xtc', tpr_file))
        res_trj = ('sample_res.xtc')
        
        trj1 = md.load(res_trj, top=gro_file)
        trj2= md.load(unwrapped_trj, top=gro_file)
        print(trj1)
        print(trj2)
        comtrj = make_comtrj(trj2)
        print(comtrj)
        comtrj.save_xtc('sample_com_unwrapped.xtc')
        comtrj[-1].save_gro('comtest.gro')
        print('make whole')
        
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
    
    
    def _std(sliced,n_chunks):
        frames = sliced.n_frames
        each_chunk = frames//n_chunks
        start_frame = 0
        D_values = []
        for i in range(n_chunks):
            if i == 0:
                pass
            else:
                traj_chunk = sliced[i*each_chunk:(i+1)*each_chunk]
                D, MSD = _run_overall(traj_chunk, mol)
                D_values.append(D)           
        stdev = np.std(D_values)
        return stdev
        
    
    def _save_overall( mol, trj, MSD, stdev):
        name = "Christopher_2022"
        np.savetxt( 'msd-{}-overall-{}.txt'.format(mol, name),np.transpose(np.vstack([trj.time, MSD])),header='# Time (ps)\tMSD (nm^2)')
        tempe = 298 #write the temperature
        res = stats.linregress(trj.time, MSD)
        fig, ax = plt.subplots()
        ax.plot(trj.time, MSD)
        ax.plot(trj.time, res.intercept + res.slope*(trj.time), 'r', alpha=0.3, linewidth= 0.8)
        slope = '{:.2e}'.format(res.slope)
        dif_c = '{:.2e}'.format((res.slope)*(1/(1*(10**18)))*(1/6)*(1*(10**12)))
        stddev = '{:.2e}'.format(stdev)
        ax.text(((max(trj.time)/6)*1.5), (max(MSD)/5)*4.5,"Slope: {} nm^2/ps \n Diffussion coef: {} +- {} m^2/s \n T:{}K \n ".format(slope,dif_c,stddev,tempe) , horizontalalignment='center', verticalalignment = 'center',bbox=dict(facecolor='orange', alpha=0.2))
        ax.set_xlabel('Simulation time (ps)')
        ax.set_ylabel('MSD (nm^2)')
        fig.suptitle('MSD for {}'.format(mol))
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
                        'water': trj.top.select("resname water"),
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
        stdev = _std(sliced,3)
        _save_overall( mol, sliced, MSD, stdev)
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
    elif 'acn' in atom1 or 'water' in atom1:
        selections['solvent'] = trj.topology.select(atom1)
    if 'li' in atom2 or 'emim' in atom2 or 'bmim' in atom2:
        selections['cation'] = trj.topology.select(atom2)
    elif 'tfsi' in atom2:
        selections['anion'] = trj.topology.select(atom2)
    elif 'acn' in atom2 or 'water' in atom2:
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
            plt.ylabel('g (r)')
            name1 = atom1.split()
            atom1_title = '{}({})'.format(name1[1],name1[-1])
            name2 = atom2.split()
            atom2_title = '{}({})'.format(name2[1],name2[-1])
            fig.suptitle('RDF {} - {}'.format(atom1_title,atom2_title))
            plt.savefig('rdf {} - {}.pdf'.format(atom1_title,atom2_title))
            print('done')
        


### Nernst-Einstein Conductivity
def neconductivity(ion,D_cat,D_an,V=343,T=298,q=1,stride=100):   ### enter ion as ex. 'resname tfsi'

    # ion - enter as ex. 'resname tfsi'
    # D_cat/D_an - enter D of both from msd in m^2/s
    # V in nm^3
    # T in K
    
    
    top_file = ('com.gro')
    trj_file = ('sample_com_unwrapped.xtc')
    trj = md.load(trj_file, top=top_file, stride = stride)
    ion = trj.topology.select(ion)


    N = 2*len(ion)
    V *= 1e-27 # m^3



    D_cat *= u.m**2 / u.s
    D_an *= u.m**2 / u.s
    kT = T * 1.3806488e-23 * u.joule
    q *= u.elementary_charge
    q = q.to('Coulomb')
    V *= u.m**3

    conductivity = N / (V*kT) * q ** 2 * (D_cat + D_an)
    print("The Nernst-Einstein conductivity is: "+ str(conductivity))
    with open("NE_Conductivity.txt","w") as file:
        file.write("The Nernst-Einstein conductivity is: "+ str(conductivity))


### Density
def density(molecule):
    
    def kdtree(index_in, cutoff, vol, my_l, pmol):
        # build the KDTree using the *larger* points array
        dens_l = []
        for l in my_l:
            ln = index_in[l]
            #print(index_in.index(ln))
            points1 =np.vstack(ln)
            tree = ss.cKDTree(points1)
            lo = []
            for p in ln:
                points2 = np.array(p)
                groups = tree.query_ball_point(points2, cutoff)
                elements = list(tree.data[groups])
                dens = (((len(elements)/vol)*pmol)/((1*(10**-9)**3)*(6.022*(10**23))))*(1/1000)
                lo.append(dens)
            arr = np.array(lo)
            #print(np.mean(arr))
            dens_l.append(lo)
        return dens_l
    
    def tuning_cutoff(index,my_l,pmol):
        cutoffs = []
        rhos = []
        cutoff = 0.2
        while cutoff <= 0.5:
            vol = 4/3*np.pi* ((cutoff+0.025)**3)
            dens_l = kdtree(index, cutoff, vol, my_l, pmol)
            average = ([np.mean(np.array(x)) for x in dens_l])
            avg = round(np.mean(np.array(average)), 2)
            cutoffs.append(cutoff)
            rhos.append(avg)
            cutoff += 0.01
        return rhos,cutoffs
    
    
  
    
    
    print('loading trj')
    top_file = ('com_whole.gro')
    trj_file = ('sample_com_whole.xtc')
    trj = md.load(trj_file, top=top_file,stride = 500)
    weight = {
                "acn": 41.05,
                "wat": 18.01528,
                'tfsi': 280.149,
                'li': 6.941,
                'emim': 111.17,
                'bmim': 139.21
            }  # g/mol
    
    selections = {
                'li': 'resname li',
                'bmim': 'resname bmim',
                'emim': 'resname emim',
                'wat': 'resname wat',
                'tfsi': 'all',
                'acn': 'resname acn'
                }
    
    
    frames_ = trj.atom_slice(trj.topology.select(selections[molecule]))
    array_frames = np.arange(frames_.n_frames)
    pmol = weight[molecule]
    #pmol = pmol/2
    index = []
    for elem in array_frames:
        w =frames_[elem]
        l = []
        for particle in (w.xyz)[0]:
            l.append(particle.tolist())
        index.append(l)
    print("Sliced selection in pore!")
    print(index)
    print(np.shape(index))
    
    stride = 50
    my_list = [*range(0, len(index), stride)]
    my_l = [int(x) for x in my_list]
    threshold, cutoff_list = tuning_cutoff(index,my_l,pmol)
    fig,ax = plt.subplots()
    plt.plot(cutoff_list,threshold)
    plt.xlabel('Cutoff Value (nm)')
    plt.ylabel('Density (kg/m^3)')
    fig.suptitle('K-D Tree Optimization of the Cut-Off Value')
    plt.show()
    minimum_val = threshold.index(min(threshold))
    print(cutoff_list[minimum_val])
    print(threshold[minimum_val])
    
    
    
    cutoff = cutoff_list[minimum_val]
    stride = 50
    vol = 4/3*np.pi* ((cutoff+0.025)**3)
    time = trj.time
    time_list = [time[x] for x in my_list]
    dens_l = kdtree(index, cutoff, vol, my_l, pmol)


    average = ([np.mean(np.array(x)) for x in dens_l])
    avg = round(np.mean(np.array(average)), 2)
    std = round(np.std(np.array(average)), 1)
    
    print('The density is: {} +- {} kg/m^3'.format(avg,std))


    

### Van Hove Function
def vhf(atom1,atom2,temp=298,stride=100,numplots = 3):
    
    def compute_partial_van_hove(
        trj,
        chunk_length=10,
        selection1=None,
        selection2=None,
        r_range=(0, 1.0),
        bin_width=0.005,
        n_bins=200,
        self_correlation=True,
        periodic=True,
        opt=True,
        ):
        
        """
        Compute the partial van Hove function of a trajectory
        Parameters
        ----------
        trj : mdtraj.Trajectory
            trajectory on which to compute the Van Hove function
        chunk_length : int
            length of time between restarting averaging
        selection1 : str
            selection to be considered, in the style of MDTraj atom selection
        selection2 : str
            selection to be considered, in the style of MDTraj atom selection
        r_range : array-like, shape=(2,), optional, default=(0.0, 1.0)
            Minimum and maximum radii.
        bin_width : float, optional, default=0.005
            Width of the bins in nanometers.
        n_bins : int, optional, default=None
            The number of bins. If specified, this will override the `bin_width`
            parameter.
        self_correlation : bool or str, default=True, other: False, 'self'
            Whether or not to include the self-self correlations.
            if 'self', only self-correlations are computed.
        Returns
        -------
        r : numpy.ndarray
            r positions generated by histogram binning
        g_r_t : numpy.ndarray
            Van Hove function at each time and position
        """
    
        unique_elements = (
            set([a.element for a in trj.atom_slice(trj.top.select(selection1)).top.atoms]),
            set([a.element for a in trj.atom_slice(trj.top.select(selection2)).top.atoms]),
        )

        if any([len(val) > 1 for val in unique_elements]):
            raise UserWarning(
                "Multiple elements found in a selection(s). Results may not be "
                "direcitly comprable to scattering experiments."
            )

        # Don't need to store it, but this serves to check that dt is constant
        #dt = get_dt(trj)

        pairs = trj.top.select_pairs(selection1=selection1, selection2=selection2)
        if self_correlation == "self":
            if selection1 != selection2:
                raise ValueError(
                    "selection1 does not equal selection2, cannot calculate self-correltions."
                )
            pairs_set = np.unique(pairs)
            pairs = np.vstack([pairs_set, pairs_set]).T
            # TODO: Find better way to only use self-pairs
            # This is hacky right now
            self_correlation = False

        # Check if pair is monatomic
        # If not, do not calculate self correlations
        if selection1 != selection2 and self_correlation == True:
            warnings.warn(
                "Partial VHF calculation: No self-correlations for {} and {}, setting `self_correlation` to `False`.".format(
                    selection1, selection2
                )
            )
            self_correlation = False

        n_chunks = int(trj.n_frames / chunk_length)

        g_r_t = None
        pbar = ProgressBar()

        #for i in pbar(range(n_chunks)):
        for i in pbar(range(1)):
            times = list()
            for j in range(chunk_length):
                times.append([chunk_length * i, chunk_length * i + j])
            r, g_r_t_frame = md.compute_rdf_t(
                traj=trj,
                pairs=pairs,
                times=times,
                r_range=r_range,
                bin_width=bin_width,
                n_bins=n_bins,
                period_length=chunk_length,
                self_correlation=self_correlation,
                periodic=periodic,
                opt=opt,
            )

            if g_r_t is None:
                g_r_t = np.zeros_like(g_r_t_frame)
            g_r_t += g_r_t_frame
            

        return r, g_r_t
    
    
    #dd = md.compute_vhf
    print('loading trj')
    top_file = ('sample.gro')
    trj_file = ('sample_whole.xtc')
    trj = md.load(trj_file, top=top_file)
    
    #temp = job.statepoint()["temperature"]
    #case = job.statepoint()["case"]
    radius = 1 #job.statepoint()["r"]
    print("loaded trajectory")
    #dt = get_dt(trj)
    tot_time = trj.n_frames*trj.timestep
    topology = trj.topology
   
    selections = dict()
    if 'li' in atom1 or 'emim' in atom1 or 'bmim' in atom1:
        selections['cation'] = atom1
    elif 'tfsi' in atom1:
        selections['anion'] = atom1
    elif 'acn' in atom1 or 'wat' in atom1:
        selections['solvent'] = atom1
    if 'li' in atom2 or 'emim' in atom2 or 'bmim' in atom2:
        selections['cation'] =atom2
    elif 'tfsi' in atom2:
        selections['anion'] = atom2
    elif 'acn' in atom2 or 'wat' in atom2:
        selections['solvent'] = atom2

    

    combos = combinations(list(selections.keys()),2)
    chunk_length = 200 # frames, 10 fs output
    cpu_count = 32
    n_chunks = 2000 # make higher if more averaging needed for smoother curves
    r_max = radius # in nm
    #Define chunk start frames
    chunk_starts = np.linspace(0, trj.n_frames-chunk_length, n_chunks, dtype=int)
    start=time.time()
    for pair in combos:
        print(pair)
        if len(trj.topology.select(selections[pair[0]])) != 0 and len(trj.topology.select(selections[pair[1]])) != 0:
            print('running vhf between {0} ({1}) \tand\t{2} ({3})\t...'.format(pair[0],
                                                                        len(trj.topology.select(selections[pair[0]])),
                                                                        pair[1],
                                                                        len(trj.topology.select(selections[pair[1]]))))
            val1 = (selections[pair[0]])
            val2 = (selections[pair[1]])
            print(val1,val2)
            start_time=time.time()
            print(f"{pair[0]}-{pair[1]} pvhf calc starting now...")
            r, g_r_t = compute_partial_van_hove(trj,chunk_length = chunk_length, selection1 = val1,selection2 = val2, self_correlation = False, r_range=(0,r_max))
            t = trj.time[:chunk_length+1]
            t_save = t - t[0]
            
            fig,ax = plt.subplots()
            numplots = numplots
            frames = []
            for ii in range(numplots):
                frame = ii * ((g_r_t.shape[0])//(numplots-1) - 1)
                frames.append(frame)
                plt.plot(r,g_r_t[frame])
                plt.legend(frames[:])

            plt.xlabel('distance (nm)')
            plt.ylabel('g (r,t)')
            name1 = atom1.split()
            atom1_title = '{}({})'.format(name1[1],name1[-1])
            name2 = atom2.split()
            atom2_title = '{}({})'.format(name2[1],name2[-1])
            fig.suptitle('VHF {} - {}'.format(atom1_title,atom2_title))
            plt.savefig('vhf {} - {}.pdf'.format(atom1_title,atom2_title))
            print('done')
            
            
    '''
    #saving to .txt file
    np.savetxt(os.path.join(job.workspace(),f"pvhf_{pair[0]}{pair[1]}_{r_max}nm_{n_chunks}chunks_{temp}_distinct.txt"),g_r_t, header = "# Van Hove Function, dt: {} fs, dr: {}".format(dt, np.unique(np.round(np.diff(trj.time), 6))[0]),)
    np.savetxt(os.path.join(job.workspace(),f"r_{pair[0]}{pair[1]}_{r_max}nm_{n_chunks}chunks_{temp}_distinct.txt"), r, header="# Positions" )
    np.savetxt(os.path.join(job.workspace(),f"t_{pair[0]}{pair[1]}_{r_max}nm_{n_chunks}chunks_{temp}_distinct.txt"), t_save, header="# Time")
    end_time = time.time()
    print(f"total time for {pair[0]}{pair[1]} pair was {(end_time - start_time)/60} minutes")
    end = time.time()
    print(f"total time for {n_chunks} loops is: {(end-start)/60} minutes")
    df_vh = pd.read_csv(os.path.join(job.workspace(),f"pvhf_{pair[0]}{pair[1]}_{r_max}nm_{n_chunks}chunks_{temp}_distinct.txt"), sep=" ", skiprows = 1,header=None)
    df_vh = df_vh.apply(pd.to_numeric, errors='coerce')
    header_list = ["r"]
    df_r = pd.read_csv(os.path.join(job.workspace(),f"r_{pair[0]}{pair[1]}_{r_max}nm_{n_chunks}chunks_{temp}_distinct.txt"), sep=" ", skiprows = 1,header=None,names=header_list)
    header_list = ["time"]
    df_t = pd.read_csv(os.path.join(job.workspace(),f"t_{pair[0]}{pair[1]}_{r_max}nm_{n_chunks}chunks_{temp}_distinct.txt"), sep=" ",skiprows = 1,header=None,names=header_list)
    df1_vh = df_vh
    df_concat = pd.concat([df_t,df_r, df1_vh], axis=1)
    index_list = [0,int((df_concat.shape[0])/3),int((df_concat.shape[0])/2), int((df_concat.shape[0]-1))]
    color_list = [x/max(index_list) for x in index_list]
    labels = []
    
    
    fig, ax = plt.subplots()
    for x in index_list:
        index = index_list.index(x)
        color_val = color_list[index]
        b=df_concat.iloc[x,2:]
        a = df_concat.iloc[:,1]
        h = (df_concat.iloc[x,0]) *100
        data = pd.DataFrame({'r':a, 'Vh(r)':b})
        c = sns.color_palette("hls", 8)
        ax = sns.lineplot(data = data, x="r", y="Vh(r)", color=c[index],legend = False)
        ax.set_title("{}-{}".format(pair[0], pair[1]))
        text = '{}-ps'.format(str(h))
        labels.append(text)
    ax.legend(labels)
    plt.savefig(os.path.join(job.workspace(),f"t_{pair[0]}{pair[1]}_{r_max}nm_{n_chunks}chunks_{temp}_distinct.pdf"))
    '''
