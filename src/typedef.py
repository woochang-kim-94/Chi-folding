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
        self.chimat_h5 =         chimat_h5
        self.alat =              alat
        self.blat =              blat
        self.avec_alat =              avec_alat
        self.bvec_blat =              bvec_blat
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
    def header_from_hdf5(cls, fn_chimat):
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
        avec_bohr     = alat*avec_alat
        bvec_bohr     = blat*bvec_blat
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
        #chimat_h5.close()
        return cls(chimat_h5 = chimat_h5
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


    def get_mat_iq(self, iq):
        """
        Get 'iq'th polarizability matrix from the hdf5 file.

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
        #    2. transpose row and col
        return (self.chimat_h5['/mats/matrix'][iq,0,0,:,:,0] + \
             1j*self.chimat_h5['/mats/matrix'][iq,0,0,:,:,1]).T

    def close_h5(self):
        """
        Close hdf5 instance

        """
        self.chimat_h5.close()
        return None
