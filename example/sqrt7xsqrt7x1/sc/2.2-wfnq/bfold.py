import numpy as np
import sys, os
import matplotlib.pyplot as plt
import matplotlib.path as mpath

def read_qeout(out_file):
	fp = open(out_file,"r")
	lines = fp.readlines()
	fp.close()

	Eig = []
	for i in range(len(lines)):
		if "number of k points=" in lines[i]:
			w = lines[i].split()
			nk = eval(w[4])
		if "number of Kohn-Sham states" in lines[i]:
			w = lines[i].split()
			nbnd = int(eval(w[4]))
			if nbnd%8 == 0:
				nlines = int(nbnd/8)
			else:
				nlines = int(nbnd/8) + 1
		if "bands (ev):" in lines[i] or "band energies (ev)" in lines[i]:
			tmp = []
			for j in range(nlines):
				w = lines[i+j+2].split()
				tmp = tmp + w
			for it in range(len(tmp)):
				tmp[it]  = eval(tmp[it])
			Eig.append(tmp)
	return Eig,nk,nbnd


def fetchEnk(kpts_uc,npools, QE_cmd, QE_inp):
	E_map = []
	#os.system("module load espresso/7.0-libxc-5.2.2")
	for i in range(len(kpts_uc)):
		os.system("mkdir UCset" + str(i+1))
		#os.system("cp *.upf UCset" + str(i+1))
		os.system("cp " + QE_inp +  " UCset" + str(i+1))
		os.system("cp -r *.save UCset" + str(i+1))
		os.chdir("UCset" + str(i+1))
		fp = open(QE_inp, 'a')
		fp.write("%d\n"%( len(kpts_uc[i])))
		for j in range(len(kpts_uc[i])):
			fp.write("%f %f %f 1.0\n"%(kpts_uc[i,j,0],kpts_uc[i,j,1], kpts_uc[i,j,2]))
		fp.close()
		print("Running QE for supercell ik:", i)
		os.system(QE_cmd + "  -npools " +  str(npools) + " < " + QE_inp + " > out")
		Eig,nk,nbnd = read_qeout("out")
		np.save("Eig" + str(i), Eig)
		E_map.append(Eig)
		os.chdir("../")
	return E_map


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


def read_kpt_sc(f_name,B_sc):
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

def mapk(kpts_sc, B_sc, B_uc, mrk):
	k_uc = []
	k_uc_all = []
	for ik in range(len(kpts_sc)):
		kpG, kpG_all = findkuc(kpts_sc[ik], B_sc[0], B_sc[1],B_uc[0],B_uc[1], mrk)
		np.savetxt("kpG_k" + str(ik+1) + ".dat", kpG)
		k_uc.append(kpG)
		k_uc_all.append(kpG_all)
	return np.array(k_uc), np.array(k_uc_all)


def check_umk(kpG, kpG_list, b1_uc, b2_uc):
	for ik in range(len(kpG_list)):
		umk1 = kpG + b1_uc
		diff1 = np.linalg.norm(umk1 - kpG_list[ik])
		umk2 = kpG + b2_uc
		diff2 = np.linalg.norm(umk2 - kpG_list[ik])
		umk3 = kpG + b1_uc + b2_uc
		diff3 = np.linalg.norm(umk3 - kpG_list[ik])
		umk4 = kpG - b1_uc
		diff4 = np.linalg.norm(umk4 - kpG_list[ik])
		umk5 = kpG - b2_uc
		diff5 = np.linalg.norm(umk5 - kpG_list[ik])
		umk6 = kpG - b1_uc - b2_uc
		diff6 = np.linalg.norm(umk6 - kpG_list[ik])
		umk7 = kpG - b1_uc + b2_uc
		diff7 = np.linalg.norm(umk7 - kpG_list[ik])
		umk8 = kpG + b1_uc - b2_uc
		diff8 = np.linalg.norm(umk8 - kpG_list[ik])
		if diff1 < 1e-6 or diff2 < 1e-6 or diff3 < 1e-6 or diff4 < 1e-6 or diff5 < 1e-6 or diff6 < 1e-6 or diff7 < 1e-6 or diff8 < 1e-6 :
			return False
	return True

def filterkpG(kpG, b1_uc,b2_uc):
	kpG_new = []
	for ik in range(len(kpG)):
		di = False
		for jk in range(len(kpG)):
			diff = kpG[ik] - kpG[jk]
			add = b1_uc + b2_uc
			if diff.all() == b1_uc.all() or diff.all() == b2_uc.all()	or diff.all() == add.all():
				di == True
		if not di:
			kpG_new.append(kpG[ik])
	return np.array(kpG_new)

def findkuc(kpt, b1, b2, b1_uc, b2_uc, mrk):
	add = b1_uc + b2_uc
	delta1 = 1e-2*b1_uc/np.linalg.norm(b1_uc)
	delta2 = 1e-2*b2_uc/np.linalg.norm(b2_uc)
	b1_npd = b1_uc + delta1 - delta2
	b2_npd = b2_uc + delta2 - delta1
	b2_nmd = b2_uc - delta2 + delta1
	add_pd = b1_uc + delta1 + b2_uc + delta2
	add_md = b1_uc - delta1 + b2_uc - delta2
	m1m2 = -1*delta1 -1*delta2
	V = np.array([ [m1m2[0],m1m2[1]],[b1_npd[0],b1_npd[1]], [add_pd[0], add_pd[1]], [b2_npd[0], b2_npd[1]] ])
	bbPath = mpath.Path(V)
	kpG = []
	kpG_all = []
	for n1 in range(-50,51):
		for n2 in range(-50,51):
           #kpG.append(kpt + n1*b1 + n2*b2)
			tmp = kpt + n1*b1 + n2*b2
			kpG_all.append(tmp)
			if bbPath.contains_point(tmp):
#               plt.scatter(tmp[0], tmp[1], marker = mrk, s = 60, color = 'k')
				if check_umk(tmp, kpG, b1_uc, b2_uc):
					kpG.append(tmp)

	kpG_f = filterkpG(np.array(kpG), b1_uc, b2_uc)
	return kpG_f, kpG_all

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
# Unit-cell dimensions:
alat, A_uc, sc, npools, QE_cmd, QE_inp = read_inp("bfold.inp")

B_uc = np.linalg.inv(A_uc)
B_uc = np.transpose(B_uc)

# SC dimensions: 3x3
A_sc = np.zeros((3,3))
A_sc[0] = A_uc[0]*sc[0,0] + A_uc[1]*sc[0,1]
A_sc[1] = A_uc[0]*sc[1,0] + A_uc[1]*sc[1,1]
A_sc[2] = A_uc[2]

# 21.8: A1 = 1*a1 + 2*a2 ; A2 = -2*a1 + 3*a2
print("Super-cell vectors\n",A_sc)
B_sc = np.linalg.inv(A_sc)
B_sc = np.transpose(B_sc)

mrk = '^'
kpts_sc = read_kpt_sc("kpts_sc",B_sc)
print("Number of super-cell k-points:", len(kpts_sc))
kuc_map, kuc_all = mapk(kpts_sc, B_sc, B_uc, mrk)
print("Mapped k-points written to file kuc_map.npy")
print("Shape of kuc_map:", np.shape(kuc_map))
np.save("kuc_map", kuc_map)
np.save("kuc_all", kuc_all)
kuc_map_crys = tpbatocrys(kuc_map, B_uc)
np.save("kuc_map_crys.npy", kuc_map_crys)
