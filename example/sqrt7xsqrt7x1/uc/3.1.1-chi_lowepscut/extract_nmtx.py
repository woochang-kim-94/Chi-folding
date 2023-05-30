#!/usr/bin/env python
"""
# Chi-folding
extract nmtx from a epsmat.h5 and save it in ASCII format

Written by W. Kim  Apr. 20. 2023
woochang_kim@berkeley.edu
"""
import os
import numpy as np
import h5py


def main():
    fn_chimat = './chi0mat.h5'
    chimat_h5 = h5py.File(fn_chimat, 'r')

    nmtx = chimat_h5['/eps_header/gspace/nmtx'][:]
    np.savetxt('chi0mat.nmtx.txt', nmtx)

    chimat_h5.close()

    fn_chimat = './chimat.h5'
    chimat_h5 = h5py.File(fn_chimat, 'r')

    nmtx = chimat_h5['/eps_header/gspace/nmtx'][:]
    np.savetxt('chimat.nmtx.txt', nmtx)

    chimat_h5.close()

    return None


if __name__ == "__main__":
    main()
