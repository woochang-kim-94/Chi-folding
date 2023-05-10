import numpy as np

def main():
    _, A_uc, _, _, _, _, = read_inp('./bfold.inp')
    B_uc = np.linalg.inv(A_uc)
    B_uc = np.transpose(B_uc)
    kuc_map_crys = np.load('./kuc_map_crys.npy')

    fn  = 'kuc_map_crys.txt'
    f = open(fn, 'w')
    for mapping in kuc_map_crys:
        for kp in mapping:
            f.write(f'{kp[0]:1.10f}  {kp[1]:1.10f}  {kp[2]:1.10f}  1.0\n')

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
	sc = np.zeros((2,2), dtype = np.integer)
	for i in range(len(lines)):
		if "npools" in lines[i]:
			w = lines[i+1].split()
			npools = int(eval(w[0]))
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
		if "QE_command" in lines[i]:
			QE_cmd = lines[i+1].rstrip()
		if "QE_input_file" in lines[i]:
			QE_inp = lines[i+1].rstrip()
	return alat, A_uc, sc, npools, QE_cmd, QE_inp

if __name__=='__main__':
    main()
