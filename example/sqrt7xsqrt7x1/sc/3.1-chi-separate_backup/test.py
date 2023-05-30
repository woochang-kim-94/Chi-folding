#!/usr/bin/env python
"""


"""
import os
os.sys.path.append("../../../../src/")
import numpy as np
import h5py
from scipy.spatial import KDTree
from typedef import Polarizability

qpts_crys = np.loadtxt('qpts', usecols=(0,1,2),dtype=np.float64)
qpts_crys = qpts_crys[1:,:]
chi_sc = Polarizability.from_split(dirname='./', qpts_crys_inp=qpts_crys)
chi_sc_ref = Polarizability.from_hdf5(fn_chimat='../3.1-chi/chimat.h5')
print(chi_sc.fn_chimat)
print(chi_sc.alat)
print(chi_sc.blat)
print(chi_sc.avec_alat)
print(chi_sc.bvec_blat)
print(chi_sc.avec_bohr)
print(chi_sc.bvec_bohr)
#print(chi_sc.components)
print(chi_sc.nq) #
print(chi_sc.qpts_crys) #
print(chi_sc.qpts_bohr) #
print(chi_sc.nmtx)      #
print(chi_sc.gind_eps2rho) #
print(chi_sc_ref.gind_eps2rho) #
print(chi_sc.matrix)
print(chi_sc.matrix_diag)
breakpoint()
