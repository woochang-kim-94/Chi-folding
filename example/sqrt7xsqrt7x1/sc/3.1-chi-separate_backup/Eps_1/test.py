#!/usr/bin/env python
"""
# Chi-folding
Folding polarizability matrix from unit-cell representation to supercell representation

Generate Chi_sc(qsc,gsc,gscp) from Chi_uc(quc,guc,gucp)

*Comments on notation & naming convention*
    1. For real/reciprocal space, we use Bohr/Bohr^{-1} unit.
    2. Be carefull about distinguishing crystal and Bohr coordinate.
       Unlike q_bohr, physical meaning of q_crys depends on the cell size
    3. We use 'quc' for representing q-vectors in unit-cell BZ.
       We use 'qsc' for representing q-vectors in supercell BZ.
       We use 'gsc' and/or 'hklsc' for representing G-vectors in supercell representation.
       We use 'guc' and/or 'hkluc' for representing G-vectors in unit-cell representation.

*Comments on workflow.*
    Currently we assume that calculations of chi(eps)mat.h5 for supercell are done seperately for each q-points. It might be the case since size of chi_sc for each q-points is huge.


*TODO*
    1. input option for chi_uc.nmtx so that can change epscut for uc
    2. reading header part from seperate 'qsc'


Written by W. Kim  Apr. 20. 2023
woochang_kim@berkeley.edu
"""
import os
os.sys.path.append("../../../../../src/")
import numpy as np
import h5py
from scipy.spatial import KDTree
from typedef import Polarizability

def main():
    #sc_eps_path = '../sc/3.1-chi/'
    uc_eps_path = '../uc/3.1-chi/'
    chi_sc = Polarizability.header_from_hdf5(fn_chimat='./chimat.h5')
    print(chi_sc.fn_chimat)
    print(chi_sc.chimat_h5)
    print(chi_sc.alat)
    print(chi_sc.blat)
    print(chi_sc.avec_alat)
    print(chi_sc.bvec_blat)
    print(chi_sc.avec_bohr)
    print(chi_sc.bvec_bohr)
    print(chi_sc.components)
    print(chi_sc.nq) #
    print(chi_sc.qpts_crys) #
    print(chi_sc.qpts_bohr) #
    print(chi_sc.nmtx)      #
    print(chi_sc.gind_eps2rho) #
    print(chi_sc.matrix)
    print(chi_sc.matrix_diag)
    chi_sc.close_h5()


    return None


if __name__ == '__main__':
    main()
