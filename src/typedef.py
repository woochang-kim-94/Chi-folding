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
    def __init__(self):
        ### Reading mf_header part ###
        self.alat = None
        self.blat = None
        self.avec = None
        self.bvec = None
        self.avec_bohr    = None
        self.bvec_bohr    = None
        self.components = None
        self.nq   = None
        self.qpts_crys = None
        self.qpts_bohr = None
        self.nmtx = None
        self.gind_eps2rho = None
        self.matrix = None
        return None

    def from_hdf5(self, fn_chimat):
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

