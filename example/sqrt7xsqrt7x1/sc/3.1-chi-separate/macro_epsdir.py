import os, sys
import numpy as np

fqpts = open("qpts", 'r')
lines = fqpts.readlines()
fqpts.close()
nq    = len(lines)
qstep = 1
for iq in range(0,nq):
    dirname = "Eps_{0}".format(iq)
    if os.path.exists(dirname) and os.path.isdir(dirname):
        print(f"The directory '{dirname}' exists.")
    else:
        print(f"The directory '{dirname}' does not exist.")
        print(f"make directory '{dirname}'")
        os.system('mkdir ' + dirname)
    os.chdir(dirname)
    os.system("rm ./epsilon.inp")
    os.system("cp ../epsilon.inp .")
    f = open("epsilon.inp", 'a')
    for j in range(0, qstep):
        f.write(lines[iq + j])
    f.write("end")
    os.system("cp ../job.backup.sh .")
    os.system("sed -i 's/JOBNAME/chi_iq{0}/g' ./job.backup.sh".format(iq+1))
    os.system("sed -i 's/Eps_iq/Eps_{0}/g' ./job.backup.sh".format(iq+1))
    os.system("ln -s ../WFN")
    f.close()
#    os.system("sbatch  ./job.backup.sh")
    os.chdir("../")



