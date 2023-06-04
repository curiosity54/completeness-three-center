import pyscf
from pyscf import gto
from pyscf import scf,dft
from scipy.optimize import minimize, check_grad
from pyscf.pbc.tools.pyscf_ase import atoms_from_ase
from pyscf.geomopt.geometric_solver import optimize as pyscf_geopt
from ase.units import Bohr, Hartree
from pyscf.pbc.tools.pyscf_ase import atoms_from_ase
from generate_bispectrum_structures import generate_nu3_degen_structs, analytical_partial_derivative
def dftblock(frames, start, end, basis='ccpvdz', xc='pbe'):
#     frames = read(prefix+".xyz", "%d:%d" % (start,end))
    idx_nconv=[]
    for i, f in enumerate(tqdm.tqdm(frames[start:end])):
        if "label" in f.info:
            label = f.info["label"]
        else:
            label = str(start+i)
        pars = f.info["pars"]
        mol = pyscf.gto.M(atom = atoms_from_ase(f), basis = basis)
        rks = pyscf.dft.RKS(mol)
        rks = pyscf.scf.addons.smearing_(rks, sigma=.01, method='fermi')
        rks.xc = xc
        rks.grids.level = 5
        rks.verbose = 3
        rks.init_guess = 'minao'
        rks.scf(print=True)
        try:
            assert rks.converged==True
            print("converged")
        except:
            print(i)
            idx_nconv.append(start+i)
        dft = rks.e_free # energy_tot()
        ddft = rks.nuc_grad_method().grad()
        f.info["energy_ha"] = dft
        f.arrays["gradient_habohr"] = ddft
        f.info["label"] = label
        f.info["basis"] = basis
        f.info["xc"] = xc
        f.info["pars"] = pars
    return frames, idx_nconv

np.random.seed(12345)
ngen = 8000
# generate 4000 pairs of bispectrum degenerate frames
frames = []
for f in tqdm.tqdm(range(ngen)):
#     print(i)
    for i in range(8000):
        r = np.random.uniform(0.5,2)
        z1 = np.random.uniform(0.5,2)
        z2 = np.random.uniform(1.5,1.8)
        psi = np.random.uniform(0,np.pi)
        phi1 = np.random.uniform(np.pi/6, np.pi)
        phi2 = np.random.uniform(np.pi/6, np.pi)
        fr = generate_nu3_degen_structs(r, [0, phi1, phi1+phi2], 
                                   psi, z1, z2, "B", "B", "B")
        dist = fr[0].get_all_distances()
        if dist[0,1:].min()>1.5 and dist[0,1:].max()<1.8 and dist[np.triu_indices(len(dist),1)].flatten().min()>1.2:
            fr[0].info["pars"] = str((f, "+")+(r, z1, z2, psi, phi1, phi2))
            fr[1].info["pars"] = str((f, "-")+(r, z1, z2, psi, phi1, phi2))
#             print(i)
            break
    if (i>7990):
        print("oops!")
    frames += fr

# run DFT calculations for the dataset
dft_frames = dftblock(frames, 0, 8000, basis='ccpvdz', xc='pbe')
write("../data/boron8_8000_pbeccpvdz.xyz", dft_frames)


### Minimum energy structure #####
def run_dft(structure, basis="ccpvdz", xc="pbe", chk_file="None"):
    mol = pyscf.gto.M(atom = atoms_from_ase(structure), basis = basis)
    rks = pyscf.dft.RKS(mol)
    if chk_file != "None":        
        rks.chkfile=chk_file
        rks.init_guess='chk'
    rks.diis_space = 20
    rks.xc = xc
    rks = pyscf.scf.addons.smearing_(rks, sigma=.01, method='fermi')    
    rks.scf(print=True)
    rks.energy = rks.e_free   # e_free is the correct energy when using a smearing...
    print("energy", rks.e_tot, " free energy", rks.e_free)
    return rks

def struc_energy(pos, symbols="Li8", basis="ccpvdz", xc="pbe", chk_file="None"):    
    struc = ase.Atoms(positions=pos.reshape((-1, 3)), symbols=symbols, pbc=False)
    rks = run_dft(struc, basis, xc, chk_file)
    return rks.energy

def struc_energy_grad(pos, symbols="Li8", basis="ccpvdz", xc="pbe", chk_file="None"):    
    struc = ase.Atoms(positions=pos.reshape((-1, 3)), symbols=symbols, pbc=False)
    rks = run_dft(struc, basis, xc, chk_file)
    dft = rks.energy # energy_tot()
    ddft = rks.nuc_grad_method().grad()/0.529177 # to Ha/Å
    return dft, ddft.flatten()

#initialize positions
positions0 = np.concatenate( [ [[0,0,0]],
                 [[1.6*np.cos(t), 1.6*np.sin(t),0] for t in np.linspace(0,np.pi*2,8)[1:] ] ]).flatten()
minstruc = minimize(struc_energy_grad, positions0, args=(labels[0]*8, basis, xc, "min.chk"), options={"maxiter": 100, "disp": True}, method="BFGS", jac=True)
frame =frames[0].copy()

frame.positions[:] = minstruc["x"].reshape((-1,3))
frame_eq_energy = minstruc["fun"]
frame.info = {}
frame.info["energy_ha"] = minstruc["fun"]
write("../data/minimized_b8_struct.xyz",frame  )
