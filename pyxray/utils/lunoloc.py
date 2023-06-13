#
# Localization based on UNO from UHF/UKS
#
import math
import numpy as np
import scipy.linalg
import h5py
from functools import reduce

from pyscf import tools, gto, scf, dft
from pyscf.tools import molden
from pyxray.utils.addons import sqrtm, lowdin, lowdinPop

def scdm(coeff, ova, aux):
    no = coeff.shape[1]	
    ova = reduce(np.dot, (coeff.T, ova, aux))
    q, r, piv = scipy.linalg.qr(ova, pivoting=True)
    bc = ova[:, piv[:no]]
    ova2 = np.dot(bc.T, bc)
    s12inv = lowdin(ova2)
    cnew = reduce(np.dot, (coeff, bc, s12inv))
    return cnew

def psort(ova, fav, pT, coeff):
    pTnew = 2.0 * reduce(np.dot, (coeff.T, ova, pT, ova, coeff))
    nocc  = np.diag(pTnew)
    enorb = np.diag(reduce(np.dot, (coeff.T, ova, fav, ova, coeff)))
    index = np.argsort(enorb)
    ncoeff = coeff[:, index]
    nocc = nocc[index]
    enorb = enorb[index]
    return ncoeff, nocc, enorb

def dumpLUNO(mol, mo_coeff, mo_energy, thresh=0.05, dumpname='lunoloc.h5', local=True, dbg=True):
    ova = mol.intor_symmetric("cint1e_ovlp_sph")
    nb = mo_coeff.shape[1]
    ma = mo_coeff[0]
    mb = mo_coeff[1]
    nalpha = (mol.nelectron + mol.spin) // 2
    nbeta  = (mol.nelectron - mol.spin) // 2
    if dbg:
        print('nb, nalpha, nbeta =', nb, nalpha, nbeta)

    # Spin-averaged DM
    pTa = np.dot(ma[:, :nalpha], ma[:, :nalpha].T)
    pTb = np.dot(mb[:, :nbeta], mb[:, :nbeta].T)
    pT = 0.5 * (pTa + pTb)

    # Nat MO in Lowdin basis
    s12 = sqrtm(ova)
    s12inv = lowdin(ova)
    pTOAO = reduce(np.dot, (s12, pT, s12))
    eig, coeff = scipy.linalg.eigh(-pTOAO)
    eig = -2.0 * eig
    eig[eig < 0.0] = 0.0
    eig[abs(eig) < 1.e-14] = 0.0
    #if dbg:
    #   import matplotlib.pyplot as plt
    #   plt.plot(range(nb), eig, 'ro')
    #   plt.savefig('natocc.png', bbox_inches = 'tight', dpi=200)
    # Nat MO in AO basis
    coeff = np.dot(s12inv, coeff)

    # Nat MO energy <i|F|i>
    fa = reduce(np.dot, (ma, np.diag(mo_energy[0]), ma.T))
    fb = reduce(np.dot, (mb, np.diag(mo_energy[1]), mb.T))
    fav = 0.5 * (fa + fb)
    fexpt = reduce(np.dot, (coeff.T, ova, fav, ova, coeff))
    enorb = np.diag(fexpt)
    nocc = eig.copy()

    # define orbital space using Nat OCC 
    idx = 0
    active = []
    for i in range(nb):
       if nocc[i] <= 2.0 - thresh and nocc[i] >= thresh:
          active.append(True)
       else:
          active.append(False)
    if dbg:
        print ('\nNatural orbitals:')
        for i in range(nb):
           print ('orb:', i, active[i], nocc[i], enorb[i])
    active = np.array(active)
    actIndices = list(np.argwhere(active==True).flatten())
    cOrbs = coeff[:,:actIndices[0]]
    aOrbs = coeff[:,actIndices]
    vOrbs = coeff[:,actIndices[-1]+1:]
    nb = cOrbs.shape[0]
    nc = cOrbs.shape[1]
    na = aOrbs.shape[1]
    nv = vOrbs.shape[1]
    if dbg:
        print ('core orbs:', cOrbs.shape)
        print ('act  orbs:', aOrbs.shape)
        print ('vir  orbs:', vOrbs.shape)
    assert nc + na + nv == nb

   #################################### check this
    clmo = cOrbs
    almo = aOrbs
    vlmo = vOrbs

    if local:
        # Split-localization with PMloc 
        from pyscf import lo
        clmo = lo.PM(mol, clmo).kernel()
        #almo = lo.PM(mol, almo).kernel()
        #ierr, uc = loc(mol, clmo)
        #clmo = clmo.dot(uc)
        ierr, ua = loc(mol, almo)
        almo = almo.dot(ua)
        vlmo = scdm(vlmo, ova, s12inv)

    mo_c, n_c, e_c = psort(ova, fav, pT, clmo)
    mo_o, n_o, e_o = psort(ova, fav, pT, almo)
    mo_v, n_v, e_v = psort(ova, fav, pT, vlmo)
    lmo = np.hstack((mo_c, mo_o, mo_v)).copy()
    enorb = np.hstack([e_c, e_o, e_v])
    occ = np.hstack([n_c, n_o, n_v])

    # Lowdin Population Analysis and Dump lunoloc
    lowdinPop(mol, lmo, enorb=enorb, occ=occ)

    f = h5py.File(dumpname, 'w')
    g = f.create_group('luno')
    g.create_dataset('mo_occ', data=occ)
    g.create_dataset('mo_coeff', data=lmo)
    g.create_dataset('mo_energy', data=enorb)
    f.close()

    if dbg:
        with open('luno.molden','w') as thefile:
            molden.header(mol, thefile)
            molden.orbital_coeff(mol, thefile, lmo, ene=enorb, occ=occ)

    return lmo, enorb, occ

def dumpLUNOCLOC(mol, lmo, enorb, occ, c_list, a_list, v_list, base=1, dumpname='lunocloc.h5', dbg=True):
    idx_c = np.array(c_list) - base
    idx_a = np.array(a_list) - base
    idx_v = np.array(v_list) - base

    from functools import reduce
    enorb_ao = reduce(np.dot, (lmo, np.diag(enorb), lmo.T))
    dmat_ao = reduce(np.dot, (lmo, np.diag(occ), lmo.T))
    
    clmo = lmo[:, idx_c].copy() 
    almo = lmo[:, idx_a].copy() 
    vlmo = lmo[:, idx_v].copy() 
    
    from pyscf import lo
    clmo = lo.PM(mol, clmo).kernel()
    almo = lo.PM(mol, almo).kernel()
    
    ova = mol.intor_symmetric("cint1e_ovlp_sph")
    from pyxray.utils.lunoloc import psort
    clmo, n_c, e_c = psort(ova, enorb_ao, dmat_ao, clmo)
    almo, n_o, e_o = psort(ova, enorb_ao, dmat_ao, almo)
    
    lcmo = np.hstack((clmo, almo, vlmo)).copy()
    enorb_c = np.hstack((e_c, e_o, enorb[idx_v])).copy()
    occ_c = np.hstack((n_c, n_o, occ[idx_v])).copy()

    lowdinPop(mol, lcmo, enorb=enorb_c, occ=occ_c)

    import h5py
    f = h5py.File(dumpname, 'w')
    g = f.create_group('luno')
    g.create_dataset('mo_occ', data=occ_c)
    g.create_dataset('mo_coeff', data=lcmo)
    g.create_dataset('mo_energy', data=enorb_c)
    f.close()

    return lcmo, enorb_c, occ_c 

#------------------------------------------------
# Boys/PM-Localization
#------------------------------------------------
def loc(mol, mocoeff, tol=1.e-6, maxcycle=1000, iop=0):
    partition = gen_atom_partition(mol)
    ova = mol.intor_symmetric("cint1e_ovlp_sph")
    ierr, u = loc_kernel(mol, mocoeff, ova, partition, tol, maxcycle, iop)
    return ierr, u

def loc_kernel(mol, mocoeff, ova, partition, tol, maxcycle, iop):
    debug = False
    print ()
    print ('[pm_loc_kernel]')
    print (' mocoeff.shape=', mocoeff.shape)
    print (' tol=', tol)
    print (' maxcycle=', maxcycle)
    print (' partition=', len(partition), '\n', partition)
    k = mocoeff.shape[0]
    n = mocoeff.shape[1]
    natom = len(partition)
 
    def gen_paij(mol, mocoeff, ova, partition, iop):
        c = mocoeff.copy()
        # Mulliken matrix
        if iop == 0:
            cts = c.T.dot(ova)
            natom = len(partition)
            pija = np.zeros((natom,n,n))
            for iatom in range(natom):
                idx = partition[iatom]
                tmp = np.dot(cts[:,idx], c[idx,:])
                pija[iatom] = 0.5 * (tmp + tmp.T)
        # Lowdin
        elif iop == 1:
            s12 = sqrtm(ova)
            s12c = s12.T.dot(c)
            natom = len(partition)
            pija = np.zeros((natom, n, n))
            for iatom in range(natom):
                idx = partition[iatom]
                pija[iatom] = np.dot(s12c[idx,:].T, s12c[idx,:])
        # Boys
        elif iop == 2:
             rmat = mol.intor_symmetric('cint1e_r_sph', 3)
             pija = np.zeros((3, n, n))
             for icart in range(3):
                 pija[icart] = reduce(np.dot, (c.T, rmat[icart], c))
        # P[i,j,a]
        pija = pija.transpose(1,2,0).copy()
        return pija
 
    ## Initial from random unitary
    u = np.identity(n)
    pija = gen_paij(mol, mocoeff, ova, partition, iop)
    if debug:
        pija0 = pija.copy()
 
    # Start
    def funval(pija):
       return np.einsum('iia,iia', pija, pija)
 
    fun = funval(pija)
    print (' initial funval = ', fun)
    #
    # Iteration
    #
    for icycle in range(maxcycle):
        delta = 0.0
        # i>j
        ijdx = []
        for i in range(n-1):
            for j in range(i+1, n):
                bij = abs(np.sum(pija[i, j] * (pija[i, i] - pija[j, j])))
                ijdx.append((i, j, bij))
        # The delta value generally decay monotonically (adjoint diagonalization)
        ijdx = sorted(ijdx, key=lambda x:x[2], reverse=True)
        for i, j, bij in ijdx:
            #
            # determine angle
            #
            vij = pija[i,i] - pija[j,j] 
            aij = np.dot(pija[i,j], pija[i,j]) - 0.25 * np.dot(vij, vij)
            bij = np.dot(pija[i,j], vij)
            if abs(aij) < 1.e-10 and abs(bij) < 1.e-10:
                continue
            p1 = math.sqrt(aij**2 + bij**2)
            cos4a = -aij / p1
            sin4a = bij / p1
            cos2a = math.sqrt((1 + cos4a) * 0.5)
            sin2a = math.sqrt((1 - cos4a) * 0.5)
            cosa  = math.sqrt((1 + cos2a) * 0.5)
            sina  = math.sqrt((1 - cos2a) * 0.5)
            if sin4a < 0.0:
               cos2a = -cos2a
               sina,cosa = cosa,sina
            # stationary condition
            if abs(cosa - 1.0) < 1.e-10:
                continue
            if abs(sina - 1.0) < 1.e-10:
                continue
            # incremental value
            delta += p1*(1-cos4a)
            # Transformation
            if debug:
                g = np.identity(n)
                g[i,i] = cosa
                g[j,j] = cosa
                g[i,j] = -sina
                g[j,i] = sina
                ug = u.dot(g)
                pijag = np.einsum('ik,jl,ija->kla',ug,ug,pija0)
            # Urot
            ui = u[:,i]*cosa+u[:,j]*sina
            uj = -u[:,i]*sina+u[:,j]*cosa
            u[:,i] = ui.copy() 
            u[:,j] = uj.copy()
            # Bra-transformation of Integrals
            tmp_ip = pija[i,:,:]*cosa+pija[j,:,:]*sina
            tmp_jp = -pija[i,:,:]*sina+pija[j,:,:]*cosa
            pija[i,:,:] = tmp_ip.copy() 
            pija[j,:,:] = tmp_jp.copy()
            # Ket-transformation of Integrals
            tmp_ip = pija[:,i,:]*cosa+pija[:,j,:]*sina
            tmp_jp = -pija[:,i,:]*sina+pija[:,j,:]*cosa
            pija[:,i,:] = tmp_ip.copy()
            pija[:,j,:] = tmp_jp.copy()
            if debug:
                diff1 = np.linalg.norm(u-ug)
                diff2 = np.linalg.norm(pija-pijag)
                cu = np.dot(mocoeff,u)
                pija2 = gen_paij(cu,ova,partition)
                fun2 = funval(pija2)
                diff = abs(fun+delta-fun2)
                print ('diff(u/p/f)=', diff1, diff2, diff)
                if diff1 > 1.e-6:
                    print ('Error: ug', diff1)
                    exit()
                if diff2 > 1.e-6:
                    print ('Error: pijag', diff2)
                    exit()
                if diff > 1.e-6: 
                    print ('Error: inconsistency in PMloc: fun/fun2=', fun + delta, fun2)
                    exit()

            fun = fun + delta
            print ('icycle=', icycle, 'delta=', delta, 'fun=', fun)
            if delta < tol:
                break
    # Check 
    ierr = 0
    if delta < tol: 
        print ('CONG: PMloc converged!')
    else:
        ierr = 1
        print ('WARNING: PMloc not converged')
    return ierr, u

def gen_atom_partition(mol):
    part = {}
    for iatom in range(mol.natm):
        part[iatom] = []
    ncgto = 0
    for binfo in mol._bas:
        atom_id = binfo[0]
        lang = binfo[1]
        ncntr = binfo[3]
        nbas = ncntr*(2*lang + 1)
        part[atom_id] += range(ncgto,ncgto+nbas)
        ncgto += nbas
    partition = []
    for iatom in range(mol.natm):
        partition.append(part[iatom])
    return partition
 
