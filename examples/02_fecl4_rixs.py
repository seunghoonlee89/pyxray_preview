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
An example to run 2p3d RIXS spectrum calculation. 
'''

#==================================================================
# 0. Initial Settings for 2p3d RIXS 
#==================================================================
# parameters for calculation
model = 'feII_8o12e_lunoloc'
#model = 'feIII_8o11e_lunoloc'
method = 'mrcis'
somf = True 
pol_elems = ['xx', 'xy', 'xz', 'yx', 'yy', 'yz', 'zx', 'zy', 'zz'] 

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

# parameters for RIXS
import numpy as np
from pyscf.data import nist
HARTREE2EV = nist.HARTREE2EV  
fmax_l3xas = 714.5 if 'feII_' in model else 715.5
freqs_ex = np.array([fmax_l3xas]) / HARTREE2EV
etas_ex = np.array([0.3]) / HARTREE2EV
freqs_sc_gap = 0.02
freq_sc_min = 0 
freq_sc_max = 6.6 
freqs_sc = np.arange(freq_sc_min, freq_sc_max + 1e-5, freqs_sc_gap)
freqs_sc /= HARTREE2EV
etas_sc = np.array([0.1] * len(freqs_sc)) / HARTREE2EV

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
assert np.linalg.norm(h1e - h1e.transpose(1, 0).conj()) < 1e-9 # check Hermitian
assert np.linalg.norm(g2e - g2e.transpose(1, 0, 3, 2).conj()) < 1e-9 # check Hermitian
mpo = driver.get_qc_mpo(h1e=h1e, g2e=g2e, ecore=ecore, iprint=verbose)

# 2-3] Prepare initial mps 
mps_gs = driver.get_random_restricted_mps(
             tag="GS", bond_dim=bond_dims_gs[0], n_hole=0, nrank_mrci=1,
         )

# 2-4] Solve ground-state problem
gs_energy = driver.dmrg(mpo, mps_gs, n_sweeps=n_sweeps_gs, bond_dims=bond_dims_gs, 
                        noises=noises_gs, thrds=thrds_gs, iprint=verbose)
print('DMRG energy = %20.15f' % gs_energy)

#==================================================================
# 3. Solve Response Problem 1:
#    (w_ex - (H-E_0) + i eta_ex) |A> = mu |Psi0>
#==================================================================
dip_dic = {'x': 0, 'y': 1, 'z': 2}
dip_elems = list(set([r for r1r2 in pol_elems for r in r1r2]))
dip_elems_ex = list(set([r1r2[0] for r1r2 in pol_elems]))
assert len(dip_elems_ex) > 0 and len(dip_elems_ex) <= 3 
assert len(set(dip_elems_ex) - set(['x', 'y', 'z'])) == 0
assert len(dip_elems) > 0 and len(dip_elems) <= 3 
assert len(set(dip_elems) - set(['x', 'y', 'z'])) == 0
for r1r2 in pol_elems:
    assert len(r1r2) == 2

# 3-1] Generate mpo 
mpo = -1.0 * mpo
mpo.const_e += gs_energy 
from pyxray.utils.integral_helper import spatial_to_spin_integrals
hr = spatial_to_spin_integrals(hr)
mpos_dip = [None] * 3
for r in dip_elems:
    ii = dip_dic[r]
    mpos_dip[ii] = driver.get_qc_mpo(h1e=hr[ii], g2e=None, ecore=0, iprint=verbose)

# 3-2] Prepare initial mps for |A> 
mpss_a = [None] * 3
for r in dip_elems_ex:
    ii = dip_dic[r]
    mpss_a[ii] = driver.get_random_restricted_mps(
                    tag="MUKET%d" % ii, bond_dim=bond_dims_cps[0], 
                    n_hole=1, nrank_mrci=1
                 )
    driver.comp_gf(
        mpss_a[ii], mpos_dip[ii], mps_gs, n_sweeps=n_sweeps_cps,
        bra_bond_dims=bond_dims_cps,
        bond_dims=[mps_gs.info.bond_dim],
        thrds=thrds_cps, save_tag="MUKET%d" % ii
    )

# 3-3] Solve response problem 1
for r in dip_elems_ex:
    ii = dip_dic[r]
    driver.linear_gf(
        mpo, mpss_a[ii], mpos_dip[ii], mps_gs, freqs_ex, etas_ex,
        bra_bond_dims=bond_dims_cv, bond_dims=bond_dims_cps[-1:], 
        noises=noises_cv, n_sweeps=n_sweeps_cv, 
        thrds=thrds_cv, save_tag="A%d" % ii, iprint=verbose
    )

#==================================================================
# 4. Solve Response Problem 2:
#    (w_sc - (H-E_0) + i eta_sc) |B> = mu |A>
#    where w_sc = w_ex - w_em
#==================================================================

gf_mat = np.zeros((3, 3, len(freqs_ex), len(freqs_sc)), dtype=complex)
# 4-1] Solve response problem 2 
mpss_b = [None] * 3
for r1r2 in pol_elems:
    ii = dip_dic[r1r2[0]]
    jj = dip_dic[r1r2[1]]
    for iw in range(len(freqs_ex)):
        w1 = freqs_ex[iw]
        # 4-1-1] load mps for |A>
        tag = 'A%d_%d' % (ii, iw)
        mps_a = driver.load_mps(tag)

        # 4-1-2] Prepare initial mps for |B> 
        mpss_b[jj] = driver.get_random_restricted_mps(
                        tag="B%d" % jj, bond_dim=bond_dims_cps[0],
                        n_hole=0, nrank_mrci=1
                     )

        driver.comp_gf(
            mpss_b[jj], mpos_dip[jj], mps_a, n_sweeps=n_sweeps_cps,
            bra_bond_dims=bond_dims_cps,
            bond_dims=[mps_a.info.bond_dim],
            thrds=thrds_cps, save_tag="MUA%d_%d_%d" % (ii, jj, iw) 
        )

        # 4-1-3] Solve response problem 2 
        gf_mat[ii, jj, iw], _ = driver.linear_gf(
                            mpo, mpss_b[jj], mpos_dip[jj], mps_a, freqs_sc, etas_sc,
                            bra_bond_dims=bond_dims_cv, bond_dims=bond_dims_cv[-1:], 
                            noises=noises_cv, n_sweeps=n_sweeps_cv, 
                            thrds=thrds_cv, save_tag="B%d_%d_%d" % (ii, jj, iw), iprint=verbose
                        )

#==================================================================
# 4. Compute RIXS Cross Section
#==================================================================
cross_section = - 16 / 9 * np.pi * gf_mat.imag.sum(axis=(0,1)) 
c4 = m_fecl4.LIGHT_SPEED**4
for iex, fex in enumerate(freqs_ex):
    for isc, fsc in enumerate(freqs_sc):
        fem = fex - fsc
        cross_section[iex, isc] *= fem ** 3 * fex / c4 
print("cross section for RIXS = ", cross_section)

# dump the results
pol = ''
for r1r2 in pol_elems:
    pol += '_' + r1r2
import h5py
ff = h5py.File("result_%s_%s_m%d%s.h5"%(method, model, bond_dims_cv[-1], pol),"w")
ff.create_dataset('omega_ex', data=freqs_ex * HARTREE2EV)
ff.create_dataset('omega_sc', data=freqs_sc * HARTREE2EV)
ff.create_dataset('cross_section', data=cross_section)
ff.create_dataset('gfmat', data=gf_mat)
ff.close() 

if dbg:
    #==================================================================
    # 5. Plot RIXS spectrum 
    #==================================================================
    import h5py
    pol = ''
    for r1r2 in pol_elems:
        pol += '_' + r1r2
    dump_filename = "result_%s_%s_m%d%s.h5" % (method, model, bond_dims_cv[-1], pol)
    ff = h5py.File(dump_filename, "r")
    freqs_sc = np.array(ff['omega_sc'])
    cross_section = np.array(ff['cross_section'])

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(15,6))
    ax.plot(freqs_sc, cross_section[0], '-o', color='black', markersize=2)
    ax.set_xlabel('Energy transfer (eV)')
    ax.set_ylabel('RIXS cross section')
    ax.set_xlim([0, 6.5])
    plt.savefig('2p3dRIXS_%s_%s_m%d%s.png' % (method, model, bond_dims_cv[-1], pol), dpi=200)
    #plt.show()
    
