#!/usr/bin/env python
"""
TODO: Parallel I/O for polarizability matrix


Written by W. Kim  Apr. 20. 2023
woochang_kim@berkeley.edu
"""
import os
import numpy as np
import h5py

class Polarizability:
    def __init__(self
        ,fn_chimat = None
        ,dirname   = None
        ,chimat_h5 = None
        ,alat = None
        ,blat = None
        ,avec_alat = None
        ,bvec_blat = None
        ,avec_bohr    = None
        ,bvec_bohr    = None
        ,components = None
        ,nq   = None
        ,qpts_crys = None
        ,qpts_bohr = None
        ,nmtx = None
        ,gind_eps2rho = None
        ,matrix = None
        ,matrix_diag = None):
        self.fn_chimat =         fn_chimat
        self.dirname   =         dirname
        self.chimat_h5 =         chimat_h5
        self.alat =              alat
        self.blat =              blat
        self.avec_alat =         avec_alat
        self.bvec_blat =         bvec_blat
        self.avec_bohr    =      avec_bohr
        self.bvec_bohr    =      bvec_bohr
        self.components =        components
        self.nq   =              nq
        self.qpts_crys =         qpts_crys
        self.qpts_bohr =         qpts_bohr
        self.nmtx =              nmtx
        self.gind_eps2rho =      gind_eps2rho
        self.matrix =            matrix
        self.matrix_diag =       matrix_diag
        return None

    @classmethod
    def from_hdf5(cls, fn_chimat):
        """
        --INPUT--
        fn_chimat : str
            hdf5 file name

        --OUTPUT--
        Polarizability
        """
        ### Reading from hdf5 file ###
        chimat_h5 = h5py.File(fn_chimat, 'r')

        ### Reading mf_header part ###
        alat = chimat_h5['/mf_header/crystal/alat'][()]
        blat = chimat_h5['/mf_header/crystal/blat'][()]
        avec_alat = chimat_h5['/mf_header/crystal/avec'][:] # in alat
        bvec_blat = chimat_h5['/mf_header/crystal/bvec'][:] # in blat
        avec_bohr = alat*avec_alat
        bvec_bohr = blat*bvec_blat
        components = chimat_h5['/mf_header/gspace/components'][:]

        ### Reading eps_header part ###
        nq        = chimat_h5['/eps_header/qpoints/nq'][()]
        qpts_crys = chimat_h5['/eps_header/qpoints/qpts'][:]
        qpts_bohr = qpts_crys@bvec_bohr

        nmtx = chimat_h5['/eps_header/gspace/nmtx'][()]
        gind_eps2rho = chimat_h5['/eps_header/gspace/gind_eps2rho'][:]
        ### We do not load full matrix in this step ###
        matrix = None
        matrix_diag = None
        chimat_h5.close()
        return cls(fn_chimat=fn_chimat
        #    ,chimat_h5 = chimat_h5
            ,dirname = None
            ,alat = alat
            ,blat = blat
            ,avec_alat = avec_alat
            ,bvec_blat = bvec_blat
            ,avec_bohr    = avec_bohr
            ,bvec_bohr    = bvec_bohr
            ,components = components
            ,nq   = nq
            ,qpts_crys = qpts_crys
            ,qpts_bohr = qpts_bohr
            ,nmtx = nmtx
            ,gind_eps2rho = gind_eps2rho
            ,matrix = matrix
            ,matrix_diag = matrix_diag)


    @classmethod
    def from_split(cls, dirname, qpts_crys_inp):
        """
        Read header of chimat file from a splited calculation

        --INPUT--
        dirname : str
            path of directory

        qpts_crys_inp: float64(nq, 3)
            array of q-points (=/= q0) in crystal coordinate

        --OUTPUT--
        Polarizability
        """
        ### Reading from hdf5 file ###
        fn_chimat = None
        dirname   = dirname
        chimat_h5 = None
        alat = None
        blat = None
        avec_alat = None
        bvec_blat = None
        avec_bohr = None
        bvec_bohr = None
        components = None
        nq   = None
        qpts_crys = None
        qpts_bohr = None
        nmtx = None
        gind_eps2rho = None
        matrix = None
        matrix_diag = None

        for iq, qpt_crys in enumerate(qpts_crys_inp):
            fn = dirname + f'./Eps_{iq+1}/chimat.h5'
            if iq == 0:
                chimat_h5 = h5py.File(fn, 'r')

                ### Reading mf_header part ###
                alat = chimat_h5['/mf_header/crystal/alat'][()]
                blat = chimat_h5['/mf_header/crystal/blat'][()]
                avec_alat = chimat_h5['/mf_header/crystal/avec'][:] # in alat
                bvec_blat = chimat_h5['/mf_header/crystal/bvec'][:] # in blat
                avec_bohr = alat*avec_alat
                bvec_bohr = blat*bvec_blat
                components = chimat_h5['/mf_header/gspace/components'][:]

                ### Reading eps_header part ###
                nq        = chimat_h5['/eps_header/qpoints/nq'][()]
                qpts_crys = chimat_h5['/eps_header/qpoints/qpts'][:]
                qpts_bohr = qpts_crys@bvec_bohr

                nmtx = chimat_h5['/eps_header/gspace/nmtx'][()]
                gind_eps2rho = chimat_h5['/eps_header/gspace/gind_eps2rho'][:]
                ### We do not load full matrix in this step ###
                matrix = None
                matrix_diag = None
                chimat_h5.close()
            else:
                chimat_h5 = h5py.File(fn, 'r')

                ### Reading mf_header part ###
                alat_t = chimat_h5['/mf_header/crystal/alat'][()]
                blat_t = chimat_h5['/mf_header/crystal/blat'][()]
                avec_alat_t = chimat_h5['/mf_header/crystal/avec'][:] # in alat
                bvec_blat_t = chimat_h5['/mf_header/crystal/bvec'][:] # in blat
                avec_bohr_t = alat*avec_alat
                bvec_bohr_t = blat*bvec_blat
                #components_t = chimat_h5['/mf_header/gspace/components'][:]
                #if not ( (alat_t == alat):# and (blat_t == blat) and \
                   #      (avec_alat_t == avec_alat) and (bvec_blat_t == bvec_blat) and \
                   #      (avec_bohr_t == avec_bohr) and (bvec_bohr_t == bvec_bohr) ):
                #         (components_t  components) ):
                if not (alat_t == alat):# and (blat_t == blat) and \
                    print('Inconsistency in header')
                    Error
                else:
                    ### update attr ###
                    alat =        alat_t
                    blat =        blat_t
                    avec_alat =   avec_alat_t
                    bvec_blat =   bvec_blat_t
                    avec_bohr =   avec_bohr_t
                    bvec_bohr =   bvec_bohr_t
                    #components =  components_t

                ### Reading eps_header part ###
                nq_t        = chimat_h5['/eps_header/qpoints/nq'][()]
                qpts_crys_t = chimat_h5['/eps_header/qpoints/qpts'][:]
                qpts_bohr_t = qpts_crys_t@bvec_bohr_t
                nmtx_t = chimat_h5['/eps_header/gspace/nmtx'][()]
                gind_eps2rho_t = chimat_h5['/eps_header/gspace/gind_eps2rho'][:]

                ### Check consistency of qpts ###
                if not nq_t == 1:
                    print(f'nq should be 1 in split reading mode')
                    Error

                if not np.allclose(qpts_crys_t[0], qpt_crys):
                    print(f'Inconsistency in {iq+1}')
                    Error

                ### update attr ###
                nq += nq_t
                qpts_crys = np.concatenate((qpts_crys, qpts_crys_t), axis=0)
                qpts_bohr = np.concatenate((qpts_bohr, qpts_bohr_t), axis=0)
                nmtx      = np.concatenate((nmtx, nmtx_t), axis=0)
                gind_eps2rho  = np.concatenate((gind_eps2rho, gind_eps2rho_t), axis=0)

                ### We do not load full matrix in this step ###
                chimat_h5.close()

        return cls(fn_chimat=fn_chimat
        #    ,chimat_h5 = chimat_h5
            ,dirname = dirname
            ,alat = alat
            ,blat = blat
            ,avec_alat = avec_alat
            ,bvec_blat = bvec_blat
            ,avec_bohr    = avec_bohr
            ,bvec_bohr    = bvec_bohr
            ,components = components
            ,nq   = nq
            ,qpts_crys = qpts_crys
            ,qpts_bohr = qpts_bohr
            ,nmtx = nmtx
            ,gind_eps2rho = gind_eps2rho
            ,matrix = matrix
            ,matrix_diag = matrix_diag)

    def get_mat_iq_par(self, iq, comm):
        """
        Get 'iq'th polarizability matrix from the hdf5 file.
        Note! We only use this method to read uc chimat and do not use this method to read sc chimat!

        --INPUT--
        iq : int
            q-point index in python convention within given chimat.h5 file.
            Note! we have separate file (chi0mat.h5) for q0
            so the first q-point in chimat.h5 is not the global first q-point

        comm : MPI instance


        --OUTPUT--
        mat_iq : complex128(ng,ng)

        """
        #WK: we need
        #    1. make complex array
        #    2. transpose row and col bcz of C-ordering
        chimat_h5 = h5py.File(self.fn_chimat, 'r', driver='mpio', comm=comm)
        matrix    = (chimat_h5['/mats/matrix'][iq,0,0,:,:,0] + \
             1j*chimat_h5['/mats/matrix'][iq,0,0,:,:,1]).T
        chimat_h5.close()

        return matrix

    def get_mat_iq(self, iq):
        """
        Get 'iq'th polarizability matrix from the hdf5 file.
        Note! We only use this method to read uc chimat and do not use this method to read sc chimat!

        --INPUT--
        iq : int
            q-point index in python convention within given chimat.h5 file.
            Note! we have separate file (chi0mat.h5) for q0
            so the first q-point in chimat.h5 is not the global first q-point


        --OUTPUT--
        mat_iq : complex128(ng,ng)

        """
        #WK: we need
        #    1. make complex array
        #    2. transpose row and col bcz of C-ordering
        chimat_h5 = h5py.File(self.fn_chimat, 'r')
        matrix    = (chimat_h5['/mats/matrix'][iq,0,0,:,:,0] + \
             1j*chimat_h5['/mats/matrix'][iq,0,0,:,:,1]).T
        chimat_h5.close()

        return matrix

    def close_h5(self):
        """
        Close hdf5 instance

        """
        self.chimat_h5.close()
        return None
