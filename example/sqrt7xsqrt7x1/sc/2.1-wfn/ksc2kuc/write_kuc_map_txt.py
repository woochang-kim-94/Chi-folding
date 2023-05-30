import numpy as np

def main():
    _, A_uc, _, _, _, _, = read_inp('./ksc2kuc.inp')
    B_uc = np.linalg.inv(A_uc)
    B_uc = np.transpose(B_uc)
    kuc_map = np.load("kuc_map.npy")
    kuc_map_crys = tpbatocrys(kuc_map, B_uc)


    kpts_tpba = []
    for mapping in kuc_map_crys:
        for kp in mapping:
            k_tpba = kp[0]*B_uc[0] + kp[1]*B_uc[1]
            kpts_tpba.append(k_tpba)

    kpts_tpba = np.array(kpts_tpba)
    np.savetxt('kpts_tpba', kpts_tpba)


    fn  = 'kuc_map_crys.txt'
    f = open(fn, 'w')
    for mapping in kuc_map_crys:
        for kp in mapping:
            f.write(f'{kp[0]:15.10f}  {kp[1]:15.10f}  {kp[2]:15.10f}  1.0\n')

    f.close()


    fnq = 'kuc_map_crys_q.txt'
    qshift = np.array([0.001, 0.0, 0.0])
    f = open(fnq, 'w')
    for mapping in kuc_map_crys:
        for kp in mapping:
            kp = kp + qshift
            f.write(f'{kp[0]:15.10f}  {kp[1]:15.10f}  {kp[2]:15.10f}  1.0\n')

    f.close()

def read_kpt(f_name,B_sc):
    # Read f_name file and return in tpba format
    fp = open(f_name)
    fmt = fp.readline().split()
    nkpts = eval(fp.readline().split()[0])
    kpts = np.loadtxt(f_name, skiprows = 2)
    if fmt[0] == 'tpba':
        return kpts
    if fmt[0] == 'crystal':
        kpts_tpba = np.zeros((nkpts, 3))
        for ik in range(nkpts):
            if nkpts == 1:
                kpts_tpba[ik] = kpts[0]*B_sc[0] + kpts[1]*B_sc[1]
            else:
                kpts_tpba[ik] = kpts[ik, 0]*B_sc[0] + kpts[ik, 1]*B_sc[1]
        return kpts_tpba

def read_inp(f_name):
    fp = open(f_name)
    lines = fp.readlines()
    A_uc = np.zeros((3,3))
    sc = np.zeros((2,2), dtype = np.int16)
    nv_sc = None
    nc_sc = None
    nF_sc = None
    for i in range(len(lines)):
        if "nv_sc" in lines[i]:
            w = lines[i+1].split()
            nv_sc, nc_sc, nF_sc = eval(w[0]), eval(w[1]), eval(w[2])
        if "alat" in lines[i]:
            w = lines[i+1].split()
            alat = eval(w[0])
        if "Unit-cell vectors" in lines[i]:
            for j in range(3):
                w = lines[i+j + 1].split()
                A_uc[j] = np.array([eval(w[0]), eval(w[1]), eval(w[2])])
        if "Super-cell vectors" in lines[i]:
            for j in range(2):
                w = lines[i+j + 1].split()
                sc[j] = np.array([eval(w[0]), eval(w[1])])
    return None, A_uc, None, None, None, None


def tpbatocrys_one(kpt_tpba, B):
    # Input: kpoints in tpba format
    # B: Reciprocal lattice vectors in tpba units
    kpt_tpba = np.array(kpt_tpba)
    A = np.linalg.inv(B)
    k_c = np.dot(kpt_tpba, A)
    return np.array(k_c)

def tpbatocrys(kpt_tpba, B):
    # Input: kpoints in tpba format
    # B: Reciprocal lattice vectors in tpba units
    kpt_tpba = np.array(kpt_tpba)
    kpt_c = []
    A = np.linalg.inv(B)
    for j in range(len(kpt_tpba)):
        tmp = []
        for i in range(len(kpt_tpba[0])):
            k_c = np.dot(kpt_tpba[j,i], A)
            tmp.append(k_c)
        kpt_c.append(tmp)
    return np.array(kpt_c)



if __name__=='__main__':
    main()
