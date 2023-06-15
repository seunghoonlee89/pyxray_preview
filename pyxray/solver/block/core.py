import time, os, shutil
import numpy as np
from pyblock2.driver.core import DMRGDriver, SymmetryTypes

class XrayDriver(DMRGDriver):
    def __init__(
        self,
        stack_mem=1 << 30,
        scratch="./nodex",
        clean_scratch=True,
        restart_dir=None,
        n_threads=None,
        symm_type=SymmetryTypes.SGFCPX,
        mpi=None,
        stack_mem_ratio=0.4,
        fp_codec_cutoff=1e-16,
        save_dir=None,
        verbose=1,
    ):
        super().__init__(
            stack_mem=stack_mem,
            scratch=scratch,
            clean_scratch=clean_scratch,
            restart_dir=restart_dir,
            n_threads=n_threads,
            symm_type=symm_type,
            mpi=mpi,
            stack_mem_ratio=stack_mem_ratio,
            fp_codec_cutoff=fp_codec_cutoff,
        )
        self.save_dir = save_dir 
        if save_dir is not None and not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        self.verbose = verbose

    def initialize_system(
        self,
        n_sites,
        n_elec,
        spin,
        n_core,
        n_inact,
        n_exter,
        n_act,
        pg_irrep=None,
        orb_sym=None,
        heis_twos=-1,
        heis_twosz=0,
        singlet_embedding=True,
        pauli_mode=False,
    ):
        bw = self.bw
        import numpy as np

        self.vacuum = bw.SX(0, 0, 0)
        if heis_twos != -1 and bw.SX == bw.b.SU2 and n_elec == 0:
            n_elec = n_sites * heis_twos
        elif heis_twos == 1 and SymmetryTypes.SGB in bw.symm_type and n_elec != 0:
            n_elec = 2 * n_elec - n_sites
        if pg_irrep is None:
            if hasattr(self, "pg_irrep"):
                pg_irrep = self.pg_irrep
            else:
                pg_irrep = 0
        if not SymmetryTypes.SU2 in bw.symm_type or heis_twos != -1:
            singlet_embedding = False
        if singlet_embedding:
            assert heis_twosz == 0
            self.target = bw.SX(n_elec + spin % 2, 0, pg_irrep)
            self.left_vacuum = bw.SX(spin % 2, spin, 0)
        else:
            self.target = bw.SX(
                n_elec if heis_twosz == 0 else heis_twosz, spin, pg_irrep
            )
            self.left_vacuum = self.vacuum
        self.n_sites = n_sites
        self.heis_twos = heis_twos
        self.n_elec = n_elec
        self.spin = spin
        self.n_core = n_core
        self.n_inact = n_inact
        self.n_exter = n_exter
        self.n_act = n_act
        if orb_sym is None:
            self.orb_sym = bw.b.VectorUInt8([0] * self.n_sites)
        else:
            if np.array(orb_sym).ndim == 2:
                self.orb_sym = bw.b.VectorUInt8(list(orb_sym[0]) + list(orb_sym[1]))
            else:
                self.orb_sym = bw.b.VectorUInt8(orb_sym)
        if pauli_mode:
            self.ghamil = self.get_pauli_hamiltonian()
        else:
            self.ghamil = bw.bs.GeneralHamiltonian(
                self.vacuum, self.n_sites, self.orb_sym, self.heis_twos
            )


    def get_random_restricted_mps(
        self,
        tag,
        bond_dim=500,
        center=0,
        dot=2,
        target=None,
        nroots=1,
        occs=None,
        full_fci=False,
        left_vacuum=None,
        nrank_mrci=0,
        n_hole=None,
        orig_dot=False,
    ):
        bw = self.bw
        if target is None:
            target = self.target
        if left_vacuum is None:
            left_vacuum = self.left_vacuum
        if nroots == 1:
            mps_info = bw.brs.MPSInfo(
                self.n_sites, self.vacuum, target, self.ghamil.basis
            )
            mps = bw.bs.MPS(self.n_sites, center, dot if orig_dot else 1)
        else:
            targets = bw.VectorSX([target]) if isinstance(target, bw.SX) else target
            mps_info = bw.brs.MultiMPSInfo(
                self.n_sites, self.vacuum, targets, self.ghamil.basis
            )
            mps = bw.bs.MultiMPS(self.n_sites, center, dot if orig_dot else 1, nroots)
        mps_info.tag = tag
        if full_fci:
            mps_info.set_bond_dimension_full_fci(left_vacuum, self.vacuum)
        else:
            mps_info.set_bond_dimension_fci(left_vacuum, self.vacuum)
        if n_hole is not None and self.n_core > 0:
            self.restrict_active_space(
                mps_info, nrank_mrci, n_hole, target
            )
        if occs is not None:
            mps_info.set_bond_dimension_using_occ(bond_dim, bw.b.VectorDouble(occs))
        else:
            mps_info.set_bond_dimension(bond_dim)
        mps_info.bond_dim = bond_dim
        mps.initialize(mps_info)
        mps.random_canonicalize()
        if nroots == 1:
            mps.tensors[mps.center].normalize()
        else:
            for xwfn in mps.wfns:
                xwfn.normalize()
        mps.save_mutable()
        mps_info.save_mutable()
        mps.save_data()
        mps_info.save_data(self.scratch + "/%s-mps_info.bin" % tag)
        if dot != 1 and not orig_dot:
            mps = self.adjust_mps(mps, dot=dot)[0]
        #mps.deallocate()
        #mps_info.deallocate_mutable()
        return mps

    def restrict_active_space(self, info, nrank, nhole, target, verbose=None):
        if verbose is None:
            verbose = self.verbose
        n_sites = info.n_sites
        nelec = self.n_elec

        if verbose >= 2:
            print('org mps_info left_dims_fci')
            for i in range(n_sites + 1):
                print(info.left_dims_fci[i])
            print('org mps_info right_dims_fci')
            for i in range(n_sites + 1):
                print(info.right_dims_fci[i])

        def restrict_qn(i, dims_fci, conds):
            dims = dims_fci[i]
            for cond in conds:
                for q in range(dims.n):
                    if cond(dims.quanta[q].n):
                        dims.n_states[q] = 0

        # left dim fci
        nc   = self.n_core
        nci  = nc + self.n_inact
        ncie = nci + self.n_exter
        nciea= ncie + self.n_act

        kp = np.array([0, nc, nci, ncie, nciea])
        nminp = np.array([0, nc-nhole, nci-nhole-nrank, nci-nhole-nrank, nelec])
        nmaxp = np.array([0, nc-nhole, nci-nhole, nci-nhole+nrank, nelec])

        for i in range(n_sites + 1):
            idx = np.argwhere(np.array(kp < i) == False).flatten()[0] - 1
            ip = i - kp[idx]

            if idx < 0:
                min_idx = max_idx = 0
            else:
                min_idx = np.maximum(nminp[idx], nminp[idx+1] - (kp[idx+1] - kp[idx] - ip))
                max_idx = np.minimum(nmaxp[idx+1], nmaxp[idx] + ip)

            cond1 = lambda n: n < min_idx 
            cond2 = lambda n: n > max_idx 
            restrict_qn(i, info.left_dims_fci, [cond1, cond2])

        # right dim fci
        na   = self.n_act
        nae  = na + self.n_exter
        naei = nae + self.n_inact
        naeic= naei + self.n_core
        ne_a = nelec - self.n_core - self.n_inact
        ne_ai= ne_a + self.n_inact 

        kp = np.array([0, na, nae, naei, naeic])
        nminp = np.array([0, ne_a+nhole-nrank, ne_a+nhole, ne_ai+nhole, nelec])
        nmaxp = np.array([0, ne_a+nhole+nrank, ne_a+nhole+nrank, ne_ai+nhole, nelec])

        for i in range(n_sites + 1):
            idx = np.argwhere(np.array(kp < i) == False).flatten()[0] - 1
            ip = i - kp[idx]

            if idx < 0:
                min_idx = max_idx = 0
            else:
                min_idx = np.maximum(nminp[idx], nminp[idx+1] - (kp[idx+1] - kp[idx] - ip))
                max_idx = np.minimum(nmaxp[idx+1], nmaxp[idx] + ip)

            cond1 = lambda n: n < min_idx 
            cond2 = lambda n: n > max_idx 
            restrict_qn(n_sites - i, info.right_dims_fci, [cond1, cond2])

        for ldf, rdf in zip(info.left_dims_fci, info.right_dims_fci):
            #StateInfo.filter(ldf, rdf, target)
            #StateInfo.filter(rdf, ldf, target)
            ldf.collect()
            rdf.collect()
        if verbose >= 2:
            print('mrci mps_info left_dims_fci')
            for i in range(self.n_sites+1):
                print(info.left_dims_fci[i])
            print('mrci mps_info right_dims_fci')
            for i in range(self.n_sites+1):
                print(info.right_dims_fci[i])
 
    def comp_gf(
        self,
        bra,
        mpo,
        ket,
        n_sweeps=10,
        tol=1e-8,
        bond_dims=None,
        bra_bond_dims=None,
        noises=None,
        noise_mpo=None,
        thrds=None,
        left_mpo=None,
        cutoff=1e-24,
        linear_max_iter=4000,
        save_tag=None,
        iprint=None,
    ):
        bw = self.bw
        if bra.info.tag == ket.info.tag:
            raise RuntimeError("Same tag for bra and ket!!")
        if save_tag is None:
           save_tag = bra.info.tag
        if iprint is None:
            iprint = self.verbose
        if iprint >= 2:
            print('>>> START Compression: %s <<<' % save_tag)
            t = time.perf_counter()
        if bond_dims is None:
            bond_dims = [ket.info.bond_dim]
        if bra_bond_dims is None:
            bra_bond_dims = [bra.info.bond_dim]
        if bra.center != ket.center:
            self.align_mps_center(bra, ket)
        if thrds is None:
            if SymmetryTypes.SP not in bw.symm_type:
                thrds = [1e-6] * 4 + [1e-7] * 1
            else:
                thrds = [1e-5] * 4 + [5e-6] * 1
        if noises is not None and noises[0] != 0 and noise_mpo is not None:
            pme = bw.bs.MovingEnvironment(noise_mpo, bra, bra, "PERT-CPS")
            pme.init_environments(iprint >= 2)
        elif left_mpo is not None:
            pme = bw.bs.MovingEnvironment(left_mpo, bra, bra, "L-MULT")
            pme.init_environments(iprint >= 2)
        else:
            pme = None
        me = bw.bs.MovingEnvironment(mpo, bra, ket, "MULT")
        me.delayed_contraction = bw.b.OpNamesSet.normal_ops()
        me.cached_contraction = pme is None  # not allowed by perturbative noise
        me.init_environments(iprint >= 2)
        if noises is None or noises[0] == 0:
            cps = bw.bs.Linear(
                pme, me, bw.b.VectorUBond(bra_bond_dims), bw.b.VectorUBond(bond_dims)
            )
        else:
            cps = bw.bs.Linear(
                pme,
                me,
                bw.b.VectorUBond(bra_bond_dims),
                bw.b.VectorUBond(bond_dims),
                bw.VectorFP(noises),
            )
        if noises is not None and noises[0] != 0:
            cps.noise_type = bw.b.NoiseTypes.ReducedPerturbative
            cps.decomp_type = bw.b.DecompositionTypes.SVD
        if noises is not None and noises[0] != 0 and left_mpo is None:
            cps.eq_type = bw.b.EquationTypes.PerturbativeCompression
        cps.iprint = iprint
        cps.cutoff = cutoff
        cps.linear_conv_thrds = bw.VectorFP(thrds)
        cps.linear_max_iter = linear_max_iter + 100
        cps.linear_soft_max_iter = linear_max_iter
        norm = cps.solve(n_sweeps, ket.center == 0, tol)
        if self.verbose >= 2:
            print(
                '>>> COMPLETE Compression | Time = %.2f <<<' %
                (time.perf_counter() - t)
            )
        bra.save_data()
        bra.info.save_data(self.scratch + "/%s-mps_info.bin" % save_tag)

        if self.save_dir is not None:
            print('>>> Saving |Phi> = mu |Psi0> <<<') 
            self.save_mps(save_tag, self.save_dir)
        if self.clean_scratch:
            me.remove_partition_files()
            if pme is not None:
                pme.remove_partition_files()
        if self.mpi is not None:
            self.mpi.barrier()
        return norm

    def linear_gf(
        self,
        left_mpo,
        bra,
        mpo,
        ket,
        freqs,
        etas, 
        extra_freqs=None,
        extra_etas=None,
        n_sweeps=10,
        tol=1e-8,
        bond_dims=None,
        bra_bond_dims=None,
        noises=None,
        noise_mpo=None,
        thrds=None,
        cutoff=1e-24,
        twodot_to_onedot=None,
        linear_max_iter=10000,
        linear_solver_params=(40,-1),
        solver_type=None,
        use_preconditioner=True,
        save_tag='A',
        iprint=0,
    ):
        bw = self.bw
        if bra.info.tag == ket.info.tag:
            raise RuntimeError("Same tag for bra and ket!!")
        if bond_dims is None:
            bond_dims = [ket.info.bond_dim]
        if bra_bond_dims is None:
            bra_bond_dims = [bra.info.bond_dim]
        if solver_type is None:
            solver_type = bw.b.LinearSolverTypes.Automatic
        self.align_mps_center(bra, ket)
        if thrds is None:
            if SymmetryTypes.SP not in bw.symm_type:
                thrds = [1e-6] * 4 + [1e-7] * 1
            else:
                thrds = [1e-5] * 4 + [5e-6] * 1
        pme = bw.bs.MovingEnvironment(left_mpo, bra, bra, "LHS")
        pme.init_environments(iprint >= 2)
        me = bw.bs.MovingEnvironment(mpo, bra, ket, "RHS")
        me.delayed_contraction = bw.b.OpNamesSet.normal_ops()
        #me.cached_contraction = pme is None  # not allowed by perturbative noise
        me.init_environments(iprint >= 2)
        if noises is None or noises[0] == 0:
            linear = bw.bs.Linear(
                pme, me, bw.b.VectorUBond(bra_bond_dims), bw.b.VectorUBond(bond_dims)
            )
        else:
            linear = bw.bs.Linear(
                pme,
                me,
                bw.b.VectorUBond(bra_bond_dims),
                bw.b.VectorUBond(bond_dims),
                bw.VectorFP(noises),
            )
        if noises is not None and noises[0] != 0:
            linear.noise_type = bw.b.NoiseTypes.ReducedPerturbative
            linear.decomp_type = bw.b.DecompositionTypes.SVD

        linear.linear_conv_thrds = bw.VectorFP(thrds)
        linear.linear_soft_max_iter = linear_max_iter
        linear.linear_max_iter = linear_max_iter + 1000
        linear.eq_type = bw.b.EquationTypes.GreensFunction
        linear.iprint = iprint 
        linear.cutoff = cutoff

        linear.linear_solver_params = linear_solver_params
        linear.solver_type = solver_type
        linear.linear_use_precondition = use_preconditioner       

        gf_mat = np.zeros((len(freqs)), dtype=complex)
        if extra_freqs is None:
            extra_freqs = [[]] * len(freqs)
        else:
            assert len(extra_freqs) == len(freqs)
        ex_gf_mats = [np.zeros((len(ex_fs)), dtype=complex) for ex_fs in extra_freqs]
        for iw, (w, xw) in enumerate(zip(freqs, extra_freqs)):
            if iprint >= 2:
                print('>>> START RESPONSE EQUATION FOR %s, OMEGA = %10.5f <<<' % (save_tag, w))
                t = time.perf_counter()

            linear.tme = None
            linear.noises[0] = noises[0]
            linear.noises = bw.VectorFP(noises)
            linear.bra_bond_dims = bw.b.VectorUBond(bra_bond_dims)
            linear.gf_omega = w
            linear.gf_extra_omegas_at_site = -1
            linear.gf_eta = etas[iw]
            linear.gf_extra_eta = 0 if extra_etas is None else extra_etas[iw]
            linear.conv_type = bw.b.ConvergenceTypes.MiddleSite
            if twodot_to_onedot is None or twodot_to_onedot > n_sweeps:
                linear.solve(n_sweeps, ket.center == 0, tol)
            else:
                if twodot_to_onedot != 0:
                    linear.solve(twodot_to_onedot, ket.center == 0, 0)
                linear.bra_bond_dims = bw.b.VectorUBond(bra_bond_dims[twodot_to_onedot:])
                linear.rme.dot = 1
                linear.lme.dot = 1
                bra.dot = 1
                ket.dot = 1
                linear.solve(len(bond_dims) - twodot_to_onedot, ket.center == 0, tol)
                if self.mpi is None or self.mpi.rank == 0:
                    bra.save_data()
                    ket.save_data() 

            # save MPS and MPSInfo for A
            print("Copy-final canonical form = ",
                   bra.canonical_form, bra.center, bra.dot)
            cp_rket = bra.deep_copy('%s_%d' % (save_tag, iw))
            if self.mpi is None or self.mpi.rank == 0:
                cp_rket.info.save_data(self.scratch + "/%s_%d-mps_info.bin" % (save_tag, iw))
                if self.save_dir is not None:
                    print('>>>   Saving |%s_%d>   <<<' 
                           % (save_tag, iw))
                    self.save_mps("%s_%d" % (save_tag, iw), self.save_dir)

            #TODO: mid_site where bond dimension maximum
            if self.symm_type != SymmetryTypes.SGFCPX: 
                min_site = np.argmin(np.array(linear.sweep_targets)[:, 1])
                mid_site = len(np.array(linear.sweep_targets)[:, 1]) // 2
            else: 
                min_site = np.argmin(np.array(linear.sweep_targets)[:, 0].imag)
                mid_site = len(np.array(linear.sweep_targets)[:, 0].imag) // 2

            if ket.center == 0:
                min_site = ket.n_sites - 2 - min_site
                mid_site = ket.n_sites - 2 - mid_site

            print("GF.IMAG MIN SITE = %4d" % min_site)
            print("GF.IMAG MID SITE = %4d" % mid_site)

            # take middle site GF
            if ket.center == 0:
                tmp = np.array(linear.sweep_targets)[::-1][mid_site, :]
            else:
                tmp = np.array(linear.sweep_targets)[mid_site, :]

            if self.symm_type != SymmetryTypes.SGFCPX: 
                rgf, igf = tmp
            else: 
                rgf = tmp.real
                igf = tmp.imag
            gf_mat[iw] = rgf + 1j * igf

            if iprint >= 1:
                print("=== %s (OMEGA = %10.5f ) = RE %20.15f + IM %20.15f === T = %7.2f" %
                       (save_tag, w, rgf, igf, time.perf_counter() - t))

            if len(xw) != 0:
                forward = ket.center == 0
                linear.noises = bw.VectorFP(noises[-1:])
                linear.bra_bond_dims = bw.b.VectorUBond(bond_dims[-1:])
                linear.gf_extra_omegas_at_site = mid_site
                if self.symm_type != SymmetryTypes.SGFCPX: 
                    linear.gf_extra_omegas = bw.b.VectorDouble(xw)
                else: 
                    linear.gf_extra_omegas = bw.b.VectorComplexDouble(xw)
                linear.solve(1, ket.center == 0, 0)

                for ixw in range(len(xw)):
                    tmp = np.array(linear.gf_extra_targets[ixw])
    
                    if self.symm_type != SymmetryTypes.SGFCPX: 
                        xrgf, xigf = tmp
                    else: 
                        xrgf = tmp.real
                        xigf = tmp.imag
                    ex_gf_mats[iw][ixw] = xrgf + 1j * xigf
                    if iprint >= 1:
                        print("=== EXT %s (OMEGA = %10.5f ) = RE %20.15f + IM %20.15f === T = %7.2f" %
                            (save_tag, xw[ixw], xrgf, xigf, 0))

            if iprint >= 2:
                print('>>> COMPLETE RESPONSE EQUATION FOR %s OMEGA = %10.5f | Time = %.2f <<<' %
                       (save_tag, w, time.perf_counter() - t))

        if self.clean_scratch:
            me.remove_partition_files()
            if pme is not None:
                pme.remove_partition_files()

        if self.mpi is not None:
            self.mpi.barrier()
        return gf_mat, ex_gf_mats 

    def save_mps(self, tag, save_dir):
        if self.mpi is None or self.mpi.rank == 0:
            for k in os.listdir(self.scratch):
                if tag in k:
                    shutil.copy(self.scratch + "/" + k, save_dir + "/" + k)
        if self.mpi is not None:
            self.mpi.barrier()

