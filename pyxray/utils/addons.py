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

import numpy as np
import scipy.linalg
from pyscf import tools, gto, scf, dft
import h5py
from functools import reduce

def sqrtm(s):
    e, v = np.linalg.eigh(s)
    return np.dot(v*np.sqrt(e), v.T.conj())

def lowdin(s):
   e, v = np.linalg.eigh(s)
   return np.dot(v / np.sqrt(e), v.T.conj())

def lowdinPop(mol, coeff, enorb=None, occ=None):
    print('Lowdin population for MOs:')
    nb, nc = coeff.shape
    if enorb is None:
        enorb = np.zeros((nc))
    if occ is None:
        occ = np.zeros((nc))
    ova = mol.intor_symmetric("cint1e_ovlp_sph")
    s12 = sqrtm(ova)
    lcoeff = s12.dot(coeff)
    diff = reduce(np.dot, (lcoeff.T, lcoeff)) - np.identity(nc)
    print(np.linalg.norm(diff))
    #assert np.linalg.norm(diff) < 1e-8
    labels = mol.spheric_labels()
    for iorb in range(nc):
       vec = lcoeff[:,iorb]**2
       idx= np.where(vec == np.amax(vec))[0]
       print(' iorb=', iorb, ' occ=', occ[iorb], ' <i|F|i>=', enorb[iorb])
       for iao in idx:
          print('    iao=', labels[iao], ' pop=', vec[iao])

def cart_to_frac(cart_coord, lattice_consts):
    a, b, c, alpha, beta, gamma = lattice_consts
    v = np.sqrt(1 -np.cos(alpha)*np.cos(alpha) - np.cos(beta)*np.cos(beta) - np.cos(gamma)*np.cos(gamma) + 2*np.cos(alpha)*np.cos(beta)*np.cos(gamma))
    tmat = np.array( [
    [ 1.0 / a, -np.cos(gamma)/(a*np.sin(gamma)), (np.cos(alpha)*np.cos(gamma)-np.cos(beta)) / (a*v*np.sin(gamma))  ],
    [ 0.0,     1.0 / (b*np.sin(gamma)),         (np.cos(beta) *np.cos(gamma)-np.cos(alpha))/ (b*v*np.sin(gamma))  ],
    [ 0.0,     0.0,                        np.sin(gamma) / (c*v)                                    ] ])
    frac_coord = np.dot(cart_coord, tmat.T)
    return frac_coord

def frac_to_cart(frac_coord, lattice_consts):
    a, b, c, alpha, beta, gamma = lattice_consts
    cosa = np.cos(alpha)
    sina = np.sin(alpha)
    cosb = np.cos(beta)
    sinb = np.sin(beta)
    cosg = np.cos(gamma)
    sing = np.sin(gamma)
    volume = 1.0 - cosa**2.0 - cosb**2.0 - cosg**2.0 + 2.0 * cosa * cosb * cosg
    volume = np.sqrt(volume)
    r = np.zeros((3, 3))
    r[0, 0] = a
    r[0, 1] = b * cosg
    r[0, 2] = c * cosb
    r[1, 1] = b * sing
    r[1, 2] = c * (cosa - cosb * cosg) / sing
    r[2, 2] = c * volume / sing
    cart_coord = np.dot(r, frac_coord.T)
    return cart_coord.T

if __name__ == '__main__':   
    from pyscf import lib
    chkfile = '../../data/fecl4/feII_uhf.h5'
    mol = lib.chkfile.load_mol(chkfile)
    f = h5py.File(chkfile, 'r')
    occ = np.array(f['scf']['mo_occ'])
    mo = np.array(f['scf']['mo_coeff'])
    ene = np.array(f['scf']['mo_energy'])
    f.close()
 
    from pyscf.tools import molden
    with open('%s_a.molden' % model, 'w') as f1:
        molden.header(mol, f1)
        molden.orbital_coeff(mol, f1, mo[0], ene=ene[0], occ=occ[0])
    with open('%s_b.molden' % model, 'w') as f1:
        molden.header(mol, f1)
        molden.orbital_coeff(mol, f1, mo[1], ene=ene[1], occ=occ[1])
    print('alpha')
    lowdinPop(mol, mo[0], ene[0], occ[0]) 
    print('beta')
    lowdinPop(mol, mo[1], ene[1], occ[1]) 

