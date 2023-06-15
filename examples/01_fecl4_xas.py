#    PyXray: a library for ab-initio X-ray spectrum simulation
#    Copyright (C) 2023  Seunghoon Lee <seunghoonlee89@gmail.com>
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
#

'''
An example to run L-edge XAS spectrum calculation. 
'''

#==================================================================
# 0. Initial Settings for Ledge XAS 
#==================================================================
# parameters for calculation
model = 'feII_8o12e_lunoloc'
#model = 'feIII_8o11e_lunoloc'
method = 'casci'
somf = True 
dip_elems = ['x', 'y', 'z'] 

# parameters for system
scratch = './nodex_%s' % model
save_dir = './nodex_%s_save' % model # save mps for gs and cv
n_threads = 28
verbose = 2 # 0: quite, 1: minimal, >1: debug
dbg = True

# parameters for GS DMRG calculation
n_sweeps_gs = 20 
bond_dims_gs = [500] * 20 
noises_gs = [1e-4] * 4 + [1e-5] * 4 + [0]
thrds_gs = [1e-6] * 8

# parameters for DDMRG calculation
n_sweeps_cps = 8
bond_dims_cps = [500] * n_sweeps_cps 
thrds_cps = [1e-10] * n_sweeps_cps

n_sweeps_cv = 20 
bond_dims_cv = [500] * n_sweeps_cv 
noises_cv = [1e-4] * (n_sweeps_cv - 14) + [0.] * 6 
thrds_cv = [1e-4] * n_sweeps_cv

# parameters for XAS
import numpy as np
from pyscf.data import nist
HARTREE2EV = nist.HARTREE2EV  
freqs_gap = 0.1
freq_min = 705
freq_max = 735
freqs = np.arange(freq_min, freq_max + 1e-5, freqs_gap)
freqs /= HARTREE2EV
etas = np.array([0.3] * len(freqs)) / HARTREE2EV

#==================================================================
# 1. Generate Active Space Model 
#==================================================================
# 1-1] Define molecule
from pyxray.model import fecl4
m_fecl4 = fecl4.ActiveSpaceModel(model)
mol = m_fecl4.gen_mol()

# 1-2] Generate orbitals
import os
dumpfile = '%s.h5' % model
if not os.path.isfile(dumpfile):
    # 1-2-1] UKS orbital optimization 
    dumpfile_mf = '%s_uks.h5' % model
    mf = m_fecl4.do_mf(mol, chkfile=dumpfile_mf)
    
    # 1-2-2] localization of nat orbs using spin averaged UKS density 
    from pyxray.utils.lunoloc import dumpLUNO
    lmo, enorb, occ = dumpLUNO(mol, mf.mo_coeff, mf.mo_energy, thresh=0.05, 
                               dumpname=dumpfile, dbg=dbg)
    if dbg:
        from pyscf.tools import molden
        with open('lunoloc.molden','w') as thefile:
            molden.header(mol, thefile)
            molden.orbital_coeff(mol, thefile, lmo, ene=enorb, occ=occ)
    
# 1-3] Generate active space model and Hamiltonian 
m_fecl4.init_model(dumpfile, method=method, dbg=dbg)
h1e, g2e, ecore = m_fecl4.gen_ham(tol=1e-6)
hso, hso2e = m_fecl4.gen_hso(somf=somf) # hso2e is None if somf is True
hr = m_fecl4.gen_hr()

n_mo, n_elec, n_core = m_fecl4.norb, m_fecl4.n_elec, m_fecl4.n_core
n_core, n_inactive, n_external, n_active = m_fecl4.n_core, m_fecl4.n_inactive, m_fecl4.n_external, m_fecl4.n_active
print("orb space: nc, ninact, next, nact = ", n_core, n_inactive, n_external, n_active)

#==================================================================
# 2. Solve Ground State Problem:
#    H |Psi0> = E_0 |Psi0>
#==================================================================
# 2-1] Initialize block2 solver 
na, nb = n_elec 
twos = 0                # for general spin
n_gmo = n_mo * 2
n_gcore = n_core * 2
n_ginactive = n_inactive * 2
n_gexternal = n_external * 2
n_gactive = n_active * 2
orb_sym = [0] * n_gmo

from pyblock2.driver.core import SymmetryTypes
from pyxray.solver.block.core import XrayDriver
driver = XrayDriver(
             scratch=scratch, symm_type=SymmetryTypes.SGFCPX, 
             n_threads=n_threads, clean_scratch=False, verbose=verbose
         )
driver.initialize_system(
    n_sites=n_gmo, n_elec=na+nb, spin=twos, n_core=n_gcore, n_inact=n_ginactive, 
    n_exter=n_gexternal, n_act=n_gactive, orb_sym=orb_sym
)

# 2-2] Generate mpo 
from pyxray.utils.integral_helper import somf_integrals, bpsoc_integrals
if somf:
    h1e, g2e = somf_integrals(h1e, g2e, hso, n_mo) 
else:
    h1e, g2e = bpsoc_integrals(h1e, g2e, hso, hso2e, n_mo, tol=int_tol) 
# check Hermitian
assert np.linalg.norm(h1e - h1e.transpose(1, 0).conj()) < 1e-9
assert np.linalg.norm(g2e - g2e.transpose(1, 0, 3, 2).conj()) < 1e-9
mpo = driver.get_qc_mpo(h1e=h1e, g2e=g2e, ecore=ecore, iprint=verbose)

# 2-3] Prepare initial mps 
mps_gs = driver.get_random_restricted_mps(
             tag="GS", bond_dim=bond_dims_gs[0], n_hole=0
         )

# 2-4] Solve ground-state problem
gs_energy = driver.dmrg(mpo, mps_gs, n_sweeps=n_sweeps_gs, bond_dims=bond_dims_gs, 
                        noises=noises_gs, thrds=thrds_gs, iprint=verbose)
print('DMRG energy = %20.15f' % gs_energy)

#==================================================================
# 3. Solve Response Problem:
#    (w - (H-E_0) + i eta) |A> = mu |Psi0>
#==================================================================
dip_dic = {'x': 0, 'y': 1, 'z': 2}
# 3-1] Generate mpo 
mpo = -1.0 * mpo
mpo.const_e += gs_energy 
from pyxray.utils.integral_helper import spatial_to_spin_integrals
hr = spatial_to_spin_integrals(hr)
mpos_dip = [None] * 3
for r in dip_elems: # loop for x, y, z
    ii = dip_dic[r]
    mpos_dip[ii] = driver.get_qc_mpo(h1e=hr[ii], g2e=None, ecore=0, iprint=verbose)

# 3-2] Prepare initial mps for |A> 
mpss_a = [None] * 3
for r in dip_elems: # loop for x, y, z
    ii = dip_dic[r]
    mpss_a[ii] = driver.get_random_restricted_mps(
                    tag="MUKET%d" % ii, bond_dim=bond_dims_cps[0], n_hole=1
                 )

    driver.comp_gf(
        mpss_a[ii], mpos_dip[ii], mps_gs, n_sweeps=n_sweeps_cps,
        bra_bond_dims=bond_dims_cps,
        bond_dims=[mps_gs.info.bond_dim],
        thrds=thrds_cps, save_tag="MUKET%d" % ii
    )

# 3-3] Solve response problem
gf_mat = np.zeros((3, len(freqs)), dtype=complex)
for r in dip_elems: # loop for x, y, z
    ii = dip_dic[r]
    gf_mat[ii], _ = driver.linear_gf(
                        mpo, mpss_a[ii], mpos_dip[ii], mps_gs, freqs, etas,
                        bra_bond_dims=bond_dims_cv, bond_dims=bond_dims_cps[-1:], 
                        noises=noises_cv, n_sweeps=n_sweeps_cv, 
                        thrds=thrds_cv, save_tag="A%d" % ii, iprint=verbose
                    )

#==================================================================
# 4. Compute XAS Spectral Function 
#==================================================================
spect_func = (-1 / np.pi) * gf_mat.imag.sum(axis=0)
print("spectral function for XAS = ", spect_func)

# dump the results
import h5py
dip = ''
for r in dip_elems:
    dip += '_' + r
dump_filename = "result_%s_%s_m%d%s.h5" % (method, model, bond_dims_cv[-1], dip)
ff = h5py.File(dump_filename, "w")
ff.create_dataset('omega', data=freqs * HARTREE2EV)
ff.create_dataset('spectral_function', data=spect_func)
ff.create_dataset('gfmat', data=gf_mat)
ff.close() 

if dbg:
    #==================================================================
    # 5. Plot XAS spectrum 
    #==================================================================
    import h5py
    dip = ''
    for r in dip_elems:
        dip += '_' + r
    dump_filename = "result_%s_%s_m%d%s.h5" % (method, model, bond_dims_cv[-1], dip)
    ff = h5py.File(dump_filename, "r")
    freqs = np.array(ff['omega'])
    spect_func = np.array(ff['spectral_function'])

    const_shift = 7.4
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(15,6))
    ax.plot(freqs - const_shift, spect_func, '-o', color='black', markersize=2)
    ax.set_xlabel('Incident Energy - %3.1f (eV)' % const_shift)
    ax.set_ylabel('XAS Spectral Function')
    ax.set_xlim([700, 725])
    plt.savefig('LedgeXAS_%s_%s_m%d%s.png' % (method, model, bond_dims_cv[-1], dip), dpi=200)
    #plt.show()
    
