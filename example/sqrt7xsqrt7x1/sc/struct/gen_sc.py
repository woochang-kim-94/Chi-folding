"""
From the pristine unitcell structures
making a new commensurate structure with differential tensile strain

"""
from numpy import *
from pymatgen.core.structure import Structure
from pymatgen.core.units import bohr_to_ang


def main():
    # 0) Reading the atomic structure of top and bottom layer of the AB stacked bilayer MoSe2
    struct = Structure.from_file('./uc_top.POSCAR')

    # Make 1 x sqrt(3) rectangular lattice which is the new unitcell of the system

    i = 1
    l, m = i , i + 1
    sc_mat = array([[ l,  m, 0],        # [ l,   m, 0]
                    [-m,l+m, 0],        # [-m, l+m, 0]
                    [ 0,  0, 1]])

    struct.make_supercell(sc_mat)

    struct.to(filename='sc_unrelaxed.cif')
    struct.to(filename='sc_unrelaxed.POSCAR')

    return None

if __name__=='__main__':
    main()
