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
        ,alat = None
        ,blat = None
        ,avec = None
        ,bvec = None
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
        ### Reading mf_header part ###
        self.alat =              alat
        self.blat =              blat
        self.avec =              avec
        self.bvec =              bvec
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
        ### Reading from hdf5 file ###
        self.chimat_h5 = h5py.File(fn_chimat, 'r')

        ### Reading mf_header part ###
        self.alat = self.chimat_h5['/mf_header/crystal/alat'][()]
        self.blat = self.chimat_h5['/mf_header/crystal/blat'][()]
        self.avec = self.chimat_h5['/mf_header/crystal/avec'][:] # in alat
        self.bvec = self.chimat_h5['/mf_header/crystal/bvec'][:] # in blat
        self.avec_bohr     = self.alat*self.avec
        self.bvec_bohr     = self.blat*self.bvec
        self.components = self.chimat_h5['/mf_header/gspace/components'][:]

        ### Reading eps_header part ###
        self.nq        = self.chimat_h5['/eps_header/qpoints/nq'][()]
        self.qpts_crys = self.chimat_h5['/eps_header/qpoints/qpts'][:]
        self.qpts_bohr = self.qpts_crys@self.bvec_bohr

        ### We do not load full matrix in this step ###
        self.nmtx = self.chimat_h5['/eps_header/gspace/nmtx'][()]
        self.gind_eps2rho = self.chimat_h5['/eps_header/gspace/gind_eps2rho'][:]
        self.matrix = None
        return None

    def get_mat_iq(self, iq):
        """
        Get mapping indices from quc+G to qsc+g
        dat
        real(nG_atquc,ng_atqsc)

        --INPUT--
        iq : int
            q-point index in python convention within given chimat.h5 file.
            Note! we have separate file (chi0mat.h5) for q0
            so the first q-point in chimat.h5 is not the global first q-point


        --OUTPUT--
        mat_iq : complex128(ng,ng)

        """

        return self.chimat_h5['/mats/matrix'][iq,0,0,:,:,:].view(dtype=np.complex128)

