import numpy as np


# N_particles runtime Q0 Q1 kT2 kH2 kS2 

Np = int(100)
runtime = 500
Q0 = 0.3
Q1 = 0.7
kT2 = 1.0
kH2 = 1.0
kS2 = 0.01
iftaxis = "true"

fname = 'parameters.csv'
with open(fname, 'w') as f:
    f.write('# N_particles runtime Q0 Q1 kT2 kH2 kS2 iftaxis ifkinesis\n')
    for kT2 in [0.1, 1.0]:
        for kH2 in [ 0.1, 1.0]:
            for kS2 in [0.1]:
                for ifkinesis in ["false","true"]:
                    params = "{Np:d},{runtime:d},{Q0},{Q1},{kT2},{kH2},{kS2},{iftaxis},{ifkinesis}".format(Np=Np,runtime=runtime,
                        Q0=Q0,Q1=Q1,kT2=kT2,kH2=kH2,kS2=kS2,iftaxis=iftaxis,ifkinesis=ifkinesis)
                    # print(params)
                    f.write(params)
                    f.write("\n")
            
