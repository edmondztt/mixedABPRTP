import numpy as np


# N_particles runtime Q0 Q1 kT2 kH2 kS2 

Np = int(100)
runtime = 600
Q0 = 0.5
kT20 = 1.0/30
kH20 = 1.0/30
kS20 = 1.0/60
iftaxis = "true"

fname = 'parameters.csv'
with open(fname, 'w') as f:
    f.write('# N_particles runtime Q0 Q1 kT2 kH2 kS2 iftaxis ifkinesis\n')
    for kT2 in np.array([2.0, 4.0])*kT20:
        for kH2 in np.array([2.0, 4.0])*kH20:
            for kS2 in np.array([1.0])*kS20:
                for Q1 in [0.9]:
                    for ifkinesis in ["false","true"]:
                        params = "{Np:d},{runtime:d},{Q0:.2f},{Q1:.2f},{kT2:.2f},{kH2:.2f},{kS2:.2f},{iftaxis},{ifkinesis}".format(Np=Np,runtime=runtime,
                            Q0=Q0,Q1=Q1,kT2=kT2,kH2=kH2,kS2=kS2,iftaxis=iftaxis,ifkinesis=ifkinesis)
                        # print(params)
                        f.write(params)
                        f.write("\n")
            
