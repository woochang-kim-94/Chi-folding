#!/usr/bin/env python
"""
Generate Chi_sc(qsc,g,gp) from Chi_uc(quc,G,Gp)

Note! Still under development!
Note! For q-points representation, we use Bohr unit.

Written by W. Kim  Apr. 20. 2023
woochang_kim@berkeley.edu
"""
import os
import numpy as np
import h5py


class Polarizability:
    def __init__(self, fn_chimat):
        ### Reading from hdf5 file ###
        self.chimat_h5 = h5py.File(fn_chimat, 'r')

        ### Reading mf_header part ###
        self.alat = self.chimat_h5['/mf_header/crystal/alat'][()]
        self.blat = self.chimat_h5['/mf_header/crystal/blat'][()]
        self.avec = self.chimat_h5['/mf_header/crystal/avec'][:] # in alat
        self.bvec = self.chimat_h5['/mf_header/crystal/bvec'][:] # in blat
        self.avec_bohr     = self.alat*self.avec
        self.bvec_bohrinv  = self.blat*self.bvec
        self.components = self.chimat_h5['/mf_header/gspace/components'][:]

        ### Reading eps_header part ###
        self.nq   = self.chimat_h5['/eps_header/qpoints/nq'][()]
        self.qpts = self.chimat_h5['/eps_header/qpoints/qpts'][:]

        ### We do not load full matrix in this step ###
        self.nmtx = self.chimat_h5['/eps_header/gspace/nmtx'][()]
        self.gind_eps2rho = self.chimat_h5['/eps_header/gspace/gind_eps2rho'][:]
        self.matrix = None

        return
