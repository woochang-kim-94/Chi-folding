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
