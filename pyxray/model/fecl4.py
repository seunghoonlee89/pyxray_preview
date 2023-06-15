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

from pyscf import gto, scf, symm, ao2mo, cc, mcscf, tools
from pyscf.data import nist
import h5py
import numpy as np

class ActiveSpaceModel():
    def __init__(self, model):
        model_list = ['feIII_8o11e_lunoloc', 'feII_8o12e_lunoloc']
        assert model in model_list
        self.model = model

        self.mol = None 
        self.mf = None 
        self.mc = None 
        self.mo_coeff = None 
        self.mo_occ = None 
        self.norb = None 
        self.n_elec = None 
        self.twos = None 
        self.h1e = None 
        self.g2e = None
        self.ecore = None
        self.hso = None
        self.hr = None
        self.HARTREE2EV = nist.HARTREE2EV
        self.LIGHT_SPEED = nist.LIGHT_SPEED   # in a.u.
        self.n_core = None 
        self.n_active = None 
        self.n_inactive = 0 
        self.n_external = 0 
        self.init_sys = False 

    def gen_mol(self, verbose=5):
        if 'feII_' in self.model:
            atom = """  Fe -17.84299991694815 -0.53096694321123 6.09104775508499
        Cl -19.84288422845700 0.31089495619796 7.04101319789001
        Cl -17.84298666758073 0.11868125024595 3.81067954087770
        Cl -17.84301352218429 -2.87052442818457 6.45826391412877
        Cl -15.84311566482982 0.31091516495189 7.04099559201853 
          """
            charge = -2
            twos = 4    # nb - na 
        elif 'feIII_' in self.model:
            atom = """   Fe   -17.843000   -0.532968    6.088435
          Cl   -19.656055    0.318354    7.047960
          Cl   -17.843000   -0.023977    3.927464
          Cl   -17.843000   -2.740763    6.330181
          Cl   -16.029945    0.318353    7.047960
        """
            charge = -1
            twos = 5    # nb - na
        else:
            assert False 

        mpg = 'c1'  # point group: d2h or c1
        ano_fe = gto.basis.parse('''
Fe    S
4316265.                     0.00015003            -0.00004622             0.00001710            -0.00000353             0.00000423
 646342.4                    0.00043597            -0.00013445             0.00004975            -0.00001026             0.00001234
 147089.7                    0.00120365            -0.00037184             0.00013758            -0.00002838             0.00003400
  41661.52                   0.00312635            -0.00096889             0.00035879            -0.00007397             0.00008934
  13590.77                   0.00814591            -0.00253948             0.00094021            -0.00019410             0.00023098
   4905.750                  0.02133892            -0.00673001             0.00249860            -0.00051496             0.00062709
   1912.746                  0.05470838            -0.01768160             0.00657103            -0.00135801             0.00160135
    792.6043                 0.12845394            -0.04375410             0.01640473            -0.00338297             0.00416181
    344.8065                 0.25203824            -0.09601111             0.03637157            -0.00754121             0.00877359
    155.8999                 0.35484986            -0.16998599             0.06664937            -0.01380066             0.01738346
     72.23091                0.27043078            -0.18456376             0.07553682            -0.01588736             0.01718943
     32.72506                0.06476086             0.05826300            -0.02586806             0.00570363            -0.00196602
     15.66762               -0.00110466             0.52163758            -0.31230230             0.06807261            -0.09285258
      7.503483               0.00184555             0.49331062            -0.44997654             0.10526256            -0.11350600
      3.312223              -0.00085600             0.08632670             0.14773374            -0.04562463             0.01812457
      1.558471               0.00037119            -0.00285017             0.72995709            -0.21341607             0.41268036
      0.683914              -0.00014687             0.00165569             0.38458847            -0.24353659             0.10339104
      0.146757               0.00006097            -0.00049176             0.01582890             0.34358715            -0.89083095
      0.070583              -0.00005789             0.00047608            -0.00949537             0.46401833            -0.80961283
      0.031449               0.00002770            -0.00022820             0.00308038             0.34688312             1.52308946
      0.012580              -0.00000722             0.00006297            -0.00100526             0.01225841             0.09142619
Fe    P
   7721.489                  0.00035287            -0.00012803            -0.00013663             0.00003845
   1829.126                  0.00196928            -0.00071517            -0.00077790             0.00021618
    593.6280                 0.00961737            -0.00352108            -0.00375042             0.00105697
    226.2054                 0.03724273            -0.01379065            -0.01516741             0.00418424
     95.26145                0.11332297            -0.04331452            -0.04705206             0.01307817
     42.85920                0.25335172            -0.10061222            -0.11529630             0.03095510
     20.04971                0.38104215            -0.16161377            -0.17017078             0.04896849
      9.620885               0.30703250            -0.11214083            -0.13220830             0.03516849
      4.541371               0.08654534             0.18501865             0.53797582            -0.08338612
      2.113500               0.00359924             0.47893080             0.61199701            -0.17709305
      0.947201               0.00144059             0.40514792            -0.64465308            -0.11907766
      0.391243              -0.00029901             0.09872160            -0.61225551             0.12237413
      0.156497               0.00020351            -0.00148592             0.10798966             0.54998130
      0.062599              -0.00009626             0.00222977             0.37358045             0.39970337
      0.025040               0.00002881            -0.00072259             0.18782870             0.08298275
Fe    D
    217.3688                 0.00096699            -0.00098327
     64.99976                0.00793294            -0.00789694
     24.77314                0.03548314            -0.03644790
     10.43614                0.10769519            -0.10760712
      4.679653               0.22555488            -0.26104796
      2.125622               0.31942979            -0.29085509
      0.945242               0.32354390             0.01254821
      0.402685               0.24338270             0.40386046
      0.156651               0.10680569             0.38672483
      0.062660               0.02052711             0.24394500
Fe    F
     11.2749                 0.03802196
      4.4690                 0.25501829
      1.7713                 0.50897998
       .7021                 0.35473516
       .2783                 0.12763297
       .1103                 0.01946831
''')
        ano_cl = gto.basis.parse('''
Cl    S
 399432.47                   0.00030484            -0.00008544             0.00002805            -0.00002594
  56908.833                  0.00097690            -0.00027415             0.00008993            -0.00008327
  16874.769                  0.00213829            -0.00060181             0.00019805            -0.00018239
   6010.0278                 0.00596941            -0.00168491             0.00055165            -0.00051268
   2303.2830                 0.01618189            -0.00461440             0.00152335            -0.00139557
    915.65994                0.04523017            -0.01312940             0.00430471            -0.00399682
    371.72009                0.11792451            -0.03590977             0.01193457            -0.01084095
    152.89109                0.26154412            -0.08838331             0.02937135            -0.02689456
     63.436303               0.40192154            -0.17459524             0.06021462            -0.05308441
     26.481723               0.26874482            -0.17768036             0.06257166            -0.05725505
     11.104112               0.03281694             0.15212083            -0.05661174             0.06432674
      4.6716289              0.00355918             0.62550754            -0.32750975             0.30253800
      1.9704520             -0.00170778             0.36355752            -0.34199199             0.51304681
       .83279460             0.00129735             0.00960204             0.31850883            -1.13091935
       .35254090            -0.00065905             0.02635221             0.66176117            -0.72817381
       .14943410             0.00024579             0.00476606             0.27830911             1.17413304
       .05977360            -0.00000146             0.00086954             0.01682723             0.32946196
Cl    P
   1288.9716                 0.00093570            -0.00024782             0.00024047
    312.24430                0.00612519            -0.00162654             0.00152795
    111.34634                0.02562804            -0.00687102             0.00673753
     43.736087               0.09281756            -0.02535755             0.02388784
     17.856524               0.24346160            -0.06874067             0.06877323
      7.4327659              0.41401588            -0.12202766             0.11190534
      3.1282538              0.34722988            -0.10596481             0.12059088
      1.3257214              0.07646871             0.13708383            -0.22523997
       .56441910             0.00212249             0.42571611            -0.64105309
       .24107410            -0.00079088             0.41428846             0.05174053
       .10320890            -0.00020296             0.18204070             0.69480634
       .04128360            -0.00009274             0.03011165             0.30411632
Cl    D
      3.6204561              0.02670164
      1.4775717              0.17231497
       .60302300             0.58274040
       .24610430             0.33745593
       .09844170             0.04917881
''')
        self.mol = gto.M(atom=atom, symmetry=mpg, basis= {'Fe': ano_fe, 'Cl': ano_cl},
                         spin=twos, charge=charge, verbose=verbose)
        return self.mol

    def do_mf(self, mol, chkfile=None):
        from pyscf import scf
        mf = scf.sfx2c(scf.UKS(mol))
        if chkfile is not None:
            mf.chkfile = '%s_uks.h5' % self.model
        mf.max_cycle = 100
        mf.conv_tol=1.e-9
        mf.level_shift = 0.3
        mf.xc = 'b88,p86' 
        mf.kernel()
        
        mf = scf.newton(mf)
        mf.conv_tol = 1.e-12
        mf.kernel()
        return mf

    def init_model(self, chkfile, MPI=None, method='casci', local=False, dbg=False):
        self.init_sys = True
        assert method in ['casci', 'mrcis']
        model = self.model

        # load dumpfile 
        self.mf = scf.ROHF(self.mol)
        if '.h5' == chkfile[-3:] and '_lunoloc' in model:
            f = h5py.File(chkfile,'r')
            self.mo_coeff = np.array(f['luno']['mo_coeff'])
            f.close()
        else:
            assert False
        self.mf.mo_coeff = self.mo_coeff

        # define CAS model
        if 'feII_' in model:
            if '_lunoloc' in model:
                if '8o12e' in model:
                    act0 = [7,8,9] # fe 2p
                    act1 = [46,47,48,49,50] # fe 3d 
                    idx = act0+act1
                    na = len(act0+act1)
                    nb = len(act0) + 1
                    act_2p = act0
                    act_3d = act1
                    self.n_core = len(act0) 
                    self.n_active = len(act1) 
                else:
                    assert False
            else:
                assert False
        elif 'feIII_' in model:
            if '_lunoloc' in model:
                if '8o11e' in model:
                    act0 = [7,8,9] # fe 2p
                    act1 = [46,47,48,49,50] # fe 3d 
                    idx = act0+act1
                    na = len(act0+act1)
                    nb = len(act0)
                    act_2p = act0
                    act_3d = act1
                    self.n_core = len(act0) 
                    self.n_active = len(act1) 
                else:
                    assert False
            else:
                assert False
        else:
            assert False

        if method == 'mrcis':
            if 'feIII_' in model and '_lunoloc' in model:
                sigmap = [34,35,36,37] # fe-cl sigma 
                inact_idx = sigmap
                virt_idx = [] 
            elif 'feII_' in model and '_lunoloc' in model:
                sigmap = [34,35,36,37] # fe-cl sigma 
                inact_idx = sigmap
                virt_idx = [] 
            else:
                assert False

            self.n_inactive = len(inact_idx)
            self.n_external = len(virt_idx) 
            idx = act_2p + inact_idx + virt_idx + act_3d
            na += len(inact_idx)
            nb += len(inact_idx)

        self.idx = idx      
        self.n_elec = (na, nb) 
        self.norb = len(idx) 
        
        self.mc = mcscf.CASCI(self.mf, self.norb, self.n_elec)
        self.mc.mo_coeff = self.mo_coeff
        self.mo_coeff = self.mc.sort_mo(idx)
        if local:
            if method == 'casci':
                nc = self.mc.ncore
                norb = self.mc.ncas
                clmo = self.mo_coeff[:, :nc].copy() 
                almo = self.mo_coeff[:, nc:nc+norb].copy() 
                vlmo = self.mo_coeff[:, nc+norb:].copy() 
            elif method == 'mrcis':
                nc = self.mc.ncore
                nic = self.n_core
                ni = self.n_inactive
                na = self.n_active
                ne = self.n_external
                norb = self.mc.ncas
                assert ni + na + ne + 3 == norb
                clmo = self.mo_coeff[:, :nc+nic+ni].copy() 
                almo = self.mo_coeff[:, nc+nic+ni:nc+nic+ni+na].copy() 
                vlmo = self.mo_coeff[:, nc+nic+ni+na:].copy() 

            if dbg:
                print('before loc')
                from pyxray.utils.addons import lowdinPop
                lowdinPop(self.mol, almo)

            from pyscf import lo
            almo = lo.PM(self.mol, almo).kernel()
            lmo = np.hstack((clmo, almo, vlmo)).copy()
            self.mo_coeff = lmo 
            self.mc.mo_coeff = self.mo_coeff

            if dbg:
                print('after loc')
                from pyxray.utils.addons import lowdinPop
                lowdinPop(self.mol, almo)
        else:
            self.mc.mo_coeff = self.mo_coeff

            if dbg:
                print('Lowdin Pop: Active Orbitals')
                nc = self.mc.ncore
                norb = self.mc.ncas
                almo = self.mo_coeff[:, nc:nc+norb]
                from pyxray.utils.addons import lowdinPop
                lowdinPop(self.mol, almo)

        # check active space model
        if dbg:
            from pyscf.tools import molden
            nc  = self.mc.ncore
            nca = self.mc.ncore + self.norb
            mocas = self.mo_coeff[:,nc:nca]
            with open('%s.molden' % model, 'w') as f1:
                molden.header(self.mol, f1)
                molden.orbital_coeff(self.mol, f1, mocas)

    def gen_ham(self, tol=1e-9):
        # Integrals for Spin-Free ab-initio CAS Hamiltonian
        assert self.init_sys
        h1e, ecore = self.mc.get_h1eff()
        h1e[np.abs(h1e) < tol] = 0
        g2e = self.mc.get_h2eff()
        g2e = ao2mo.restore(1, g2e, self.norb)
        g2e[np.abs(g2e) < tol] = 0
        return h1e, g2e, ecore

    def gen_hso(self, somf=True, amfi=True):
        # Integrals for Breit-Pauli SOC Hamiltonian
        assert self.init_sys
        if somf:
            ncore = self.mc.ncore
            nmo = self.mf.mo_coeff.shape[0]
            na, nb = self.n_elec
            occa = np.array([1 if i < ncore + na else 0 for i in range(nmo)])
            occb = np.array([1 if i < ncore + nb else 0 for i in range(nmo)]) 
            self.mo_occ = occa + occb
            def gen_mf_dmao(mo_coeff, mo_occ):
                mocc = mo_coeff[:,mo_occ>0]
                return np.dot(mocc*mo_occ[mo_occ>0], mocc.conj().T)
    
            from pyxray.utils.integral_helper import compute_hso_mo  
            from pyxray.utils.integral_helper import compute_hso_ao  
            dmao = gen_mf_dmao(self.mo_coeff, self.mo_occ) 
            mocas = self.mc.mo_coeff[:,ncore:ncore+self.norb]
            hso = compute_hso_mo(self.mol, dmao, mocas, amfi=amfi)
            return hso, None
        else:
            ncore = self.mc.ncore
            mocas = self.mc.mo_coeff[:, ncore:ncore+self.norb]
            from pyxray.utils.integral_helper import compute_bpso_mo  
            hso1e, hso2e = compute_bpso_mo(self.mol, mocas)
            return hso1e, hso2e 

    def gen_hr(self):
        # Integrals for dipole operator 
        assert self.init_sys
        hrao = self.mol.intor('int1e_r')
        ncore = self.mc.ncore
        mocas = self.mc.mo_coeff[:, ncore:ncore+self.norb]
        hr = np.einsum('rij,ip,jq->rpq', hrao, mocas, mocas)
        return hr 

    def do_fci(self, h1e, g2e, norb=None, n_elec=None, nroots=1):
        assert self.init_sys == True
        if norb is None:
            norb = self.norb
        if n_elec is None:
            n_elec = self.n_elec
        from pyscf import fci
        e, fcivec = fci.direct_spin1.kernel(h1e, g2e, norb, n_elec, nroots=nroots,
                                            max_space=1000, max_cycle=1000)
        return e, fcivec

