import numpy as np
import copy

def get_jk(mol, dm0):
    hso2e = mol.intor('int2e_p1vxp1', 3).reshape(
        3, mol.nao, mol.nao, mol.nao, mol.nao)
    vj = np.einsum('yijkl,lk->yij', hso2e, dm0)
    vk1 = np.einsum('yijkl,jk->yil', hso2e, dm0)
    vk2 = np.einsum('yijkl,li->ykj', hso2e, dm0)
    return vj, vk1, vk2

def get_jk_amfi(mol, dm0):
    '''Atomic-mean-field approximation'''
    ao_loc = mol.ao_loc_nr()
    nao = ao_loc[-1]
    vj = np.zeros((3, nao, nao))
    vk1 = np.zeros((3, nao, nao))
    vk2 = np.zeros((3, nao, nao))
    atom = copy.copy(mol)
    aoslice = mol.aoslice_by_atom(ao_loc)
    for ia in range(mol.natm):
        b0, b1, p0, p1 = aoslice[ia]
        atom._bas = mol._bas[b0:b1]
        vj1p, vk1p, vk2p = get_jk(atom, dm0[p0:p1, p0:p1])
        vj[:, p0:p1, p0:p1] = vj1p
        vk1[:, p0:p1, p0:p1] = vk1p
        vk2[:, p0:p1, p0:p1] = vk2p
    return vj, vk1, vk2

def compute_bpso_ao(mol):
    from pyscf.data import nist
    alpha2 = nist.ALPHA ** 2
    hso1e = mol.intor_asymmetric('int1e_pnucxp', 3) * alpha2 / 4
    hso2e = mol.intor('int2e_p1vxp1', 3).reshape(
            3, mol.nao, mol.nao, mol.nao, mol.nao)
    hso2e = 2 * hso2e.transpose(0, 3, 4, 1, 2) + hso2e 
    hso2e *= alpha2 / 4
    return hso1e, hso2e

def compute_bpso_mo(mol, mocas):
    hso1e, hso2e = compute_bpso_ao(mol)
    hso1e *= 2.
    hso1e = np.einsum('rmn,mi,nj->rij', hso1e, mocas, mocas, optimize=True)
    tmp1  = np.einsum('rmnop,mi->rinop', hso2e, mocas, optimize=True)
    hso2e = np.einsum('rinop,nj->rijop', tmp1, mocas, optimize=True)
    tmp1  = np.einsum('rijop,ok->rijkp', hso2e, mocas, optimize=True)
    hso2e = np.einsum('rijkp,pl->rijkl', tmp1, mocas, optimize=True)
    return hso1e * 1j, hso2e * 1j  

def compute_hso_ao(mol, dm0, qed_fac=1, amfi=False):
    from pyscf.data import nist
    alpha2 = nist.ALPHA ** 2
    hso1e = mol.intor_asymmetric('int1e_pnucxp', 3)
    uhf = True if len(dm0) == 2 else False
    rhf = not uhf
    if rhf:
        vj, vk1, vk2 = get_jk_amfi(mol, dm0) if amfi else get_jk(mol, dm0)
        hso2e = vj - (vk1 + vk2) * 1.5
        hso = qed_fac * (alpha2 / 4) * (hso1e + hso2e)
    else:
        nao = dm0[0].shape[0]
        vja, vk1a, vk2a = get_jk_amfi(mol, dm0[0]) if amfi else get_jk(mol, dm0[0])
        vjb, vk1b, vk2b = get_jk_amfi(mol, dm0[1]) if amfi else get_jk(mol, dm0[1])
        vj = vja + vjb
        hso = np.zeros((6, nao, nao))
        for i in range(6):
            hso[i, :, :] += hso1e[i // 2, :, :] + vj[i // 2, :, :]  
        hso[0, :, :] -= (vk1b[0, :, :] + vk2a[0, :, :]) * 1.5
        hso[1, :, :] -= (vk1a[0, :, :] + vk2b[0, :, :]) * 1.5
        hso[2, :, :] -= (vk1b[1, :, :] + vk2a[1, :, :]) * 1.5
        hso[3, :, :] -= (vk1a[1, :, :] + vk2b[1, :, :]) * 1.5
        hso[4, :, :] -= (vk1a[2, :, :] + vk2a[2, :, :]) * 1.5
        hso[5, :, :] -= (vk1b[2, :, :] + vk2b[2, :, :]) * 1.5
        hso *= qed_fac * (alpha2 / 4)
    return hso * 1j

def compute_hso_mo(mol, dm0, mocas, qed_fac=1, amfi=False):
    hsoao = compute_hso_ao(mol, dm0, amfi=amfi) * 2
    uhf = True if len(dm0) == 2 and len(mocas) == 2 else False
    rhf = not uhf
    if rhf:
        hso = np.einsum('rij,ip,jq->rpq', hsoao, mocas, mocas)
    else:
        moa = mocas[0]
        mob = mocas[1]
        assert moa.shape[1] == mob.shape[1]
        nmo = moa.shape[1]
        hso = np.zeros((6, nmo, nmo), dtype=complex)
        hso[0, :, :] = np.einsum('ij,ip,jq->pq', hsoao[0, :, :], moa, mob)
        hso[1, :, :] = np.einsum('ij,ip,jq->pq', hsoao[1, :, :], mob, moa)
        hso[2, :, :] = np.einsum('ij,ip,jq->pq', hsoao[2, :, :], moa, mob)
        hso[3, :, :] = np.einsum('ij,ip,jq->pq', hsoao[3, :, :], mob, moa)
        hso[4, :, :] = np.einsum('ij,ip,jq->pq', hsoao[4, :, :], moa, moa)
        hso[5, :, :] = np.einsum('ij,ip,jq->pq', hsoao[5, :, :], mob, mob)
    return hso

def spin_proj(cg, pdm, tjo, tjb, tjk):
    nmo = pdm.shape[0]
    ppdm = np.zeros((tjb + 1, tjk + 1, tjo + 1, nmo, nmo))
    for ibra in range(tjb + 1):
        for iket in range(tjk + 1):
            for iop in range(tjo + 1):
                tmb = -tjb + 2 * ibra
                tmk = -tjk + 2 * iket
                tmo = -tjo + 2 * iop
                factor = (-1) ** ((tjb - tmb) // 2) * \
                    cg.wigner_3j(tjb, tjo, tjk, -tmb, tmo, tmk)
                if factor != 0:
                    ppdm[ibra, iket, iop] = pdm * factor
    return ppdm

def xyz_proj(ppdm):
    xpdm = np.zeros(ppdm.shape, dtype=complex)
    xpdm[:, :, 0] = (0.5 + 0j) * (ppdm[:, :, 0] - ppdm[:, :, 2])
    xpdm[:, :, 1] = (0.5j + 0) * (ppdm[:, :, 0] + ppdm[:, :, 2])
    xpdm[:, :, 2] = (np.sqrt(0.5) + 0j) * ppdm[:, :, 1]
    return xpdm

def bpsoc_integrals(h1e, g2e, hso, hso2e, n, tol=1e-9):
    gh1e = np.zeros((n * 2, n * 2), dtype=complex)
    gg2e = np.zeros((n * 2, n * 2, n * 2, n * 2), dtype=complex)

    for i in range(n * 2):
        for j in range(i % 2, n * 2, 2):
            gh1e[i, j] = h1e[i // 2, j // 2]
    for i in range(n * 2):
        for j in range(i % 2, n * 2, 2):
            for k in range(n * 2):
                for l in range(k % 2, n * 2, 2):
                    gg2e[i, j, k, l] = g2e[i // 2, j // 2, k // 2, l // 2]

    #bpsoc
    if hso2e is not None:
        for i in range(n * 2):
            for j in range(n * 2):
                for k in range(n * 2):
                    for l in range(n * 2):
                        if (i % 2 == 0 and j % 2 == 0 and k % 2 == 0 and l % 2 == 0) or \
                           (i % 2 == 0 and j % 2 == 0 and k % 2 == 1 and l % 2 == 1):
                            gg2e[i, j, k, l] += hso2e[2, i // 2, j // 2, k // 2, l // 2]
                        elif (i % 2 == 0 and j % 2 == 1 and k % 2 == 0 and l % 2 == 0) or \
                             (i % 2 == 1 and j % 2 == 0 and k % 2 == 0 and l % 2 == 0) or \
                             (i % 2 == 0 and j % 2 == 1 and k % 2 == 1 and l % 2 == 1) or \
                             (i % 2 == 1 and j % 2 == 0 and k % 2 == 1 and l % 2 == 1):
                            gg2e[i, j, k, l] += hso2e[0, i // 2, j // 2, k // 2, l // 2]
                        elif (i % 2 == 1 and j % 2 == 1 and k % 2 == 0 and l % 2 == 0) or \
                             (i % 2 == 1 and j % 2 == 1 and k % 2 == 1 and l % 2 == 1):
                            gg2e[i, j, k, l] -= hso2e[2, i // 2, j // 2, k // 2, l // 2]
                        elif (i % 2 == 1 and j % 2 == 0 and k % 2 == 0 and l % 2 == 0) or \
                             (i % 2 == 1 and j % 2 == 0 and k % 2 == 1 and l % 2 == 1):
                            gg2e[i, j, k, l] += hso2e[1, i // 2, j // 2, k // 2, l // 2] * 1j
                        elif (i % 2 == 0 and j % 2 == 1 and k % 2 == 0 and l % 2 == 0) or \
                             (i % 2 == 0 and j % 2 == 1 and k % 2 == 1 and l % 2 == 1):
                            gg2e[i, j, k, l] -= hso2e[1, i // 2, j // 2, k // 2, l // 2] * 1j
    if hso is not None:
        for i in range(n * 2):
            for j in range(n * 2):
                if i % 2 == 0 and j % 2 == 0:
                    gh1e[i, j] += hso[2, i // 2, j // 2] * 0.5
                elif i % 2 == 1 and j % 2 == 1:
                    gh1e[i, j] -= hso[2, i // 2, j // 2] * 0.5
                elif i % 2 == 0 and j % 2 == 1:
                    gh1e[i, j] += (hso[0, i // 2, j // 2] - hso[1, i // 2, j // 2] * 1j) * 0.5
                elif i % 2 == 1 and j % 2 == 0:
                    gh1e[i, j] += (hso[0, i // 2, j // 2] + hso[1, i // 2, j // 2] * 1j) * 0.5
                else:
                    assert False
    return gh1e, gg2e


def somf_integrals(h1e, g2e, hso, n):
    gh1e = np.zeros((n * 2, n * 2), dtype=complex)
    gg2e = np.zeros((n * 2, n * 2, n * 2, n * 2), dtype=complex)

    uhf = True if len(h1e) == 2 and len(g2e) == 3 and len(hso) == 6 else False
    rhf = not uhf 
    if rhf:
        for i in range(n * 2):
            for j in range(i % 2, n * 2, 2):
                gh1e[i, j] = h1e[i // 2, j // 2]
        for i in range(n * 2):
            for j in range(i % 2, n * 2, 2):
                for k in range(n * 2):
                    for l in range(k % 2, n * 2, 2):
                        gg2e[i, j, k, l] = g2e[i // 2, j // 2, k // 2, l // 2]
        
        for i in range(n * 2):
            for j in range(n * 2):
                if i % 2 == 0 and j % 2 == 0:
                    gh1e[i, j] += hso[2, i // 2, j // 2] * 0.5
                elif i % 2 == 1 and j % 2 == 1:
                    gh1e[i, j] -= hso[2, i // 2, j // 2] * 0.5
                elif i % 2 == 0 and j % 2 == 1:
                    gh1e[i, j] += (hso[0, i // 2, j // 2] - hso[1, i // 2, j // 2] * 1j) * 0.5
                elif i % 2 == 1 and j % 2 == 0:
                    gh1e[i, j] += (hso[0, i // 2, j // 2] + hso[1, i // 2, j // 2] * 1j) * 0.5
                else:
                    assert False
    else:
        for i in range(n):
            iga = i * 2
            igb = i * 2 + 1
            for j in range(n):
                jga = j * 2
                jgb = j * 2 + 1 
                gh1e[iga, jga] = h1e[0][i, j]
                gh1e[igb, jgb] = h1e[1][i, j]
        for i in range(n):
            iga = i * 2
            igb = i * 2 + 1 
            for j in range(n):
                jga = j * 2
                jgb = j * 2 + 1
                for k in range(n):
                    kga = k * 2
                    kgb = k * 2 + 1
                    for l in range(n):
                        lga = l * 2
                        lgb = l * 2 + 1
                        gg2e[iga, jga, kga, lga] = g2e[0][i, j, k, l]
                        gg2e[iga, jga, kgb, lgb] = g2e[1][i, j, k, l]
                        gg2e[igb, jgb, kga, lga] = g2e[1][k, l, i, j]
                        gg2e[igb, jgb, kgb, lgb] = g2e[2][i, j, k, l]
        for i in range(n):
            iga = i * 2
            igb = i * 2 + 1
            for j in range(n):
                jga = j * 2
                jgb = j * 2 + 1 
                gh1e[iga, jga] +=  hso[4, i, j] * 0.5
                gh1e[igb, jgb] -=  hso[5, i, j] * 0.5
                gh1e[iga, jgb] += (hso[0, i, j] - hso[2, i, j] * 1j) * 0.5
                gh1e[igb, jga] += (hso[1, i, j] + hso[3, i, j] * 1j) * 0.5
    return gh1e, gg2e

def spatial_to_spin_integrals(hr):
    uhf = True if len(hr) == 2 else False
    rhf = not uhf 
    if rhf:
        a, b, c = hr.shape 
        ghr = np.zeros((a, b * 2, c * 2), dtype=complex)
        for i in range(b * 2):
            for j in range(c * 2):
                ghr[:, i, j] = hr[:, i // 2, j // 2]
    else:
        a, b, c = hr[0].shape 
        ghr = np.zeros((a, b * 2, c * 2), dtype=complex)
        for i in range(b):
            iga = i * 2
            igb = i * 2 + 1
            for j in range(c):
                jga = j * 2
                jgb = j * 2 + 1 
                ghr[:, iga, jga] = hr[0][:, i, j]
                ghr[:, igb, jgb] = hr[1][:, i, j]
    return ghr

def dump_FCIDUMP(filename, n_sites, n_elec, twos, isym, 
                 orb_sym, const_e, h1e, g2e, tol=1E-13, pg='c1',
                 general=True, tgeneral=True):
    """
    Write FCI options and integrals to FCIDUMP file.
    Args:
        filename : str
        tol : threshold for terms written into file
    """
    with open(filename, 'w') as fout:
        fout.write(' &FCI NORB=%4d,NELEC=%2d,MS2=%d,\n' %
                   (n_sites, n_elec, twos))
        #TODO: add PGsym
        #if orb_sym is not None and len(orb_sym) > 0:
        #    fout.write('  ORBSYM=%s,\n' % ','.join(
        #        [str(PointGroup[pg].index(x)) for x in orb_sym]))
        #    fout.write('  ISYM=%d,\n' % PointGroup[pg].index(ipg))
        #else:
        fout.write('  ORBSYM=%s\n' % ('1,' * n_sites))
        fout.write('  ISYM=1,\n')
        if isinstance(h1e, tuple) and len(h1e) == 2:
            fout.write('  IUHF=1,\n')
            assert isinstance(g2e, tuple)
        if general:
            fout.write('  IGENERAL=1,\n')
        if tgeneral:
            fout.write('  ITGENERAL=1,\n')
        fout.write(' &END\n')
        output_format = '%20.16f%4d%4d%4d%4d\n'
        output_format_cpx = '%20.16f%20.16f%4d%4d%4d%4d\n'
        npair = n_sites * (n_sites + 1) // 2
        nmo = n_sites

        def write_eri(fout, eri):
            assert eri.ndim in [1, 2, 4]
            if eri.ndim == 4:
                # general
                assert(eri.size == nmo ** 4)
                for i in range(nmo):
                    for j in range(nmo):
                        for k in range(nmo):
                            for l in range(nmo):
                                if abs(eri[i, j, k, l]) > tol:
                                    fout.write(output_format % (
                                        eri[i, j, k, l], i + 1, j + 1, k + 1, l + 1))
            elif eri.ndim == 2:
                # 4-fold symmetry
                assert eri.size == npair ** 2
                ij = 0
                for i in range(nmo):
                    for j in range(0, i + 1):
                        kl = 0
                        for k in range(0, nmo):
                            for l in range(0, k + 1):
                                if abs(eri[ij, kl]) > tol:
                                    fout.write(output_format % (
                                        eri[ij, kl], i + 1, j + 1, k + 1, l + 1))
                                kl += 1
                        ij += 1
            else:
                # 8-fold symmetry
                assert eri.size == npair * (npair + 1) // 2
                ij = 0
                ijkl = 0
                for i in range(nmo):
                    for j in range(0, i + 1):
                        kl = 0
                        for k in range(0, i + 1):
                            for l in range(0, k + 1):
                                if ij >= kl:
                                    if abs(eri[ijkl]) > tol:
                                        fout.write(output_format % (
                                            eri[ijkl], i + 1, j + 1, k + 1, l + 1))
                                    ijkl += 1
                                kl += 1
                        ij += 1

        def write_eri_cpx(fout, eri):
            assert eri.ndim in [1, 2, 4]
            if eri.ndim == 4:
                # general
                assert(eri.size == nmo ** 4)
                for i in range(nmo):
                    for j in range(nmo):
                        for k in range(nmo):
                            for l in range(nmo):
                                if abs(eri[i, j, k, l]) > tol:
                                    fout.write(output_format_cpx % (
                                        np.real(eri[i, j, k, l]), np.imag(eri[i, j, k, l]),
                                        i + 1, j + 1, k + 1, l + 1))
            elif eri.ndim == 2:
                # 4-fold symmetry
                assert eri.size == npair ** 2
                ij = 0
                for i in range(nmo):
                    for j in range(0, i + 1):
                        kl = 0
                        for k in range(0, nmo):
                            for l in range(0, k + 1):
                                if abs(eri[ij, kl]) > tol:
                                    fout.write(output_format_cpx % (
                                        np.real(eri[ij, kl]), np.imag(eri[ij, kl]),
                                        i + 1, j + 1, k + 1, l + 1))
                                kl += 1
                        ij += 1
            else:
                # 8-fold symmetry
                assert eri.size == npair * (npair + 1) // 2
                ij = 0
                ijkl = 0
                for i in range(nmo):
                    for j in range(0, i + 1):
                        kl = 0
                        for k in range(0, i + 1):
                            for l in range(0, k + 1):
                                if ij >= kl:
                                    if abs(eri[ijkl]) > tol:
                                        fout.write(output_format_cpx % (
                                            np.real(eri[ijkl]), np.imag(eri[ijkl]),
                                            i + 1, j + 1, k + 1, l + 1))
                                    ijkl += 1
                                kl += 1
                        ij += 1

        def write_h1e(fout, hx, tgen=False):
            h = hx.reshape(nmo, nmo)
            for i in range(nmo):
                for j in range(0, nmo if tgen else i + 1):
                    if abs(h[i, j]) > tol:
                        fout.write(output_format %
                                   (h[i, j], i + 1, j + 1, 0, 0))

        def write_h1e_cpx(fout, hx, tgen=False):
            h = hx.reshape(nmo, nmo)
            for i in range(nmo):
                for j in range(0, nmo if tgen else i + 1):
                    if abs(h[i, j]) > tol:
                        fout.write(output_format_cpx %
                                   (np.real(h[i, j]), np.imag(h[i, j]), i + 1, j + 1, 0, 0))
        
        def fold_eri(eri, fold=1):
            if fold == 1:
                return eri
            elif fold == 8:
                xeri = np.zeros((npair * (npair + 1) // 2, ), dtype=eri.dtype)
                ij = 0
                ijkl = 0
                for i in range(nmo):
                    for j in range(0, i + 1):
                        kl = 0
                        for k in range(0, i + 1):
                            for l in range(0, k + 1):
                                if ij >= kl:
                                    xeri[ijkl] = eri[i, j, k, l]
                                    ijkl += 1
                                kl += 1
                        ij += 1
                return xeri
            elif fold == 4:
                xeri = np.zeros((npair, npair))
                ij = 0
                for i in range(nmo):
                    for j in range(0, i + 1):
                        kl = 0
                        for k in range(0, nmo):
                            for l in range(0, k + 1):
                                xeri[ij, kl] = eri[i, j, k, l]
                                kl += 1
                        ij += 1

        if isinstance(g2e, tuple):
            assert len(g2e) == 3
            vaa, vab, vbb = g2e
            if not general:
                vaa = fold_eri(vaa, 8)
                vab = fold_eri(vab, 4)
                vbb = fold_eri(vbb, 8)
                assert vaa.ndim == vbb.ndim == 1 and vab.ndim == 2
            else:
                assert vaa.ndim == 4 and vbb.ndim == 4 and vab.ndim == 4
            if vaa.dtype == float or vaa.dtype == np.float64:
                for eri in [vaa, vbb, vab]:
                    write_eri(fout, eri)
                    fout.write(output_format % (0, 0, 0, 0, 0))
                assert len(h1e) == 2
                for hx in h1e:
                    write_h1e(fout, hx, tgeneral)
                    fout.write(output_format % (0, 0, 0, 0, 0))
                fout.write(output_format % (const_e, 0, 0, 0, 0))
            else:
                for eri in [vaa, vbb, vab]:
                    write_eri_cpx(fout, eri)
                    fout.write(output_format_cpx % (0, 0, 0, 0, 0, 0))
                assert len(h1e) == 2
                for hx in h1e:
                    write_h1e_cpx(fout, hx, tgeneral)
                    fout.write(output_format_cpx % (0, 0, 0, 0, 0, 0))
                fout.write(output_format_cpx % (np.real(const_e), np.imag(const_e), 0, 0, 0, 0))
        else:
            if g2e is not None and not general: 
                eri = fold_eri(g2e, 8)
            else:
                eri = g2e
            if h1e.dtype == float or h1e.dtype == np.float64:
                if g2e is not None:
                    write_eri(fout, eri)
                write_h1e(fout, h1e, tgeneral)
                fout.write(output_format % (const_e, 0, 0, 0, 0))
            else:
                if g2e is not None:
                    write_eri_cpx(fout, eri)
                write_h1e_cpx(fout, h1e, tgeneral)
                fout.write(output_format_cpx % (np.real(const_e), np.imag(const_e), 0, 0, 0, 0))

def orbital_reorder(h1e, g2e, method='gaopt'):
    """
    Find an optimal ordering of orbitals for DMRG.
    Ref: J. Chem. Phys. 142, 034102 (2015)

    Args:
        method :
            'gaopt' - genetic algorithm, take several seconds
            'fiedler' - very fast, may be slightly worse than 'gaopt'

    Return a index array "midx":
        reordered_orb_sym = original_orb_sym[midx]
    """
    if method is None:
        n_sites = h1e.shape[0]
        midx = np.array(range(n_sites))
        return midx

    n_sites = h1e.shape[0]
    hmat = np.zeros((n_sites, n_sites))
    xmat = np.zeros((n_sites, n_sites))
    from pyscf import ao2mo
    if not isinstance(h1e, tuple):
        hmat[:] = np.abs(h1e[:])
        g2e = ao2mo.restore(1, g2e, n_sites)
        for i in range(0, n_sites):
            for j in range(0, n_sites):
                xmat[i, j] = abs(g2e[i, j, j, i])
    else:
        assert SpinLabel == SZ
        assert isinstance(h1e, tuple) and len(h1e) == 2
        assert isinstance(g2e, tuple) and len(g2e) == 3
        hmat[:] = 0.5 * np.abs(h1e[0][:]) + 0.5 * np.abs(h1e[1][:])
        g2eaa = ao2mo.restore(1, g2e[0], n_sites)
        g2ebb = ao2mo.restore(1, g2e[1], n_sites)
        g2eab = ao2mo.restore(1, g2e[2], n_sites)
        for i in range(0, n_sites):
            for j in range(0, n_sites):
                xmat[i, j] = 0.25 * abs(g2eaa[i, j, j, i]) \
                    + 0.25 * abs(g2ebb[i, j, j, i]) \
                    + 0.5 * abs(g2eab[i, j, j, i])
    kmat = VectorFP((np.array(hmat) * 1E-7 + np.array(xmat)).flatten())
    if method == 'gaopt':
        n_tasks = 32
        opts = dict(
            n_generations=10000, n_configs=n_sites * 2,
            n_elite=8, clone_rate=0.1, mutate_rate=0.1
        )
        midx, mf = None, None
        for _ in range(0, n_tasks):
            idx = OrbitalOrdering.ga_opt(n_sites, kmat, **opts)
            f = OrbitalOrdering.evaluate(n_sites, kmat, idx)
            idx = np.array(idx)
            if mf is None or f < mf:
                midx, mf = idx, f
    elif method == 'fiedler':
        idx = OrbitalOrdering.fiedler(n_sites, kmat)
        midx = np.array(idx)
    return midx

def compute_hs2(ngmo, igeneral=True):
    """
        compute integrals of S^2 operator in second quantization form
    """
    if not igeneral:
        n = ngmo
        gh1e = np.zeros((n, n), dtype=float)
        gg2e = np.zeros((n, n, n, n), dtype=float)
        for i in range(n):
            gh1e[i, i] += 0.75
        for i in range(n):
            for j in range(n):
                gg2e[i, i, j, j] -= 0.5
                gg2e[i, j, j, i] -= 1.0
        ecore = 0. 
    else:
        gh1e = np.zeros((ngmo, ngmo), dtype=complex)
        gg2e = np.zeros((ngmo, ngmo, ngmo, ngmo), dtype=complex)
        n = ngmo // 2
        assert ngmo % 2 == 0
        for i in range(n):
            iga = i * 2
            igb = i * 2 + 1
            gh1e[iga, iga] += 0.75 
            gh1e[igb, igb] += 0.75 
        for i in range(n):
            iga = i * 2
            igb = i * 2 + 1
            for j in range(n):
                jga = j * 2
                jgb = j * 2 + 1
                gg2e[iga, iga, jga, jga] -= 0.5 
                gg2e[iga, iga, jgb, jgb] -= 0.5 
                gg2e[igb, igb, jga, jga] -= 0.5 
                gg2e[igb, igb, jgb, jgb] -= 0.5 
    
                gg2e[iga, jga, jga, iga] -= 1.0 
                gg2e[iga, jga, jgb, igb] -= 1.0 
                gg2e[igb, jgb, jga, iga] -= 1.0 
                gg2e[igb, jgb, jgb, igb] -= 1.0 
        ecore = 0. 
    return ecore, gh1e, gg2e

def compute_psp(s, sp, h1e_s2, g2e_s2):
    """
        compute integrals of spin projection operator P_{S'}(S)
        P_{S'} = (\hat{S}^2 - S'(S'+1)) / (S(S+1) - S'(S'+1))
        for spin projection operator P_S = \prod_{S' != S} P_{S'}(S)
        where P_S |Psi> = |Psi^{(S)}>
    """
    denom = s * (s + 1) - sp * (sp + 1)
    h1e_psp = h1e_s2.copy() / denom
    g2e_psp = g2e_s2.copy() / denom
    ecore_psp = - sp * (sp+1) / denom
    return h1e_psp, g2e_psp, ecore_psp

