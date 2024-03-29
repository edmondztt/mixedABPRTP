import numpy as np


# N_particles runtime Q0 Q1 kT2 kH2 kS2 

Np = int(100)
runtime = 600
Q0 = 0.5
Q1 = 0.9
kT20 = 1.0/30
kH20 = 1.0/15
kS20 = 1.0/60
gamma0_inv = 15
iftaxis = "true"
ifkk = "true"
ifok = "true"

kS2 = kS20
fname = 'parameters.csv'
with open(fname, 'w') as f:
    f.write('# N_particles runtime gamma0_inv Q0 Q1 kT2 kH2 kS2 iftaxis ifkk ifok iflarge\n')
    for kT2 in np.array([1.0, 2.0, 4.0])*kT20:
        for kH2 in np.array([1.0, 2.0, 4.0])*kH20:
            for if_large in ["false", "true"]:
                for ifok in ["false","true"]:
                    for ifkk in ["false","true"]:
                        params = "{Np:d},{runtime:d},{gamma0_inv:d},{Q0:.2f},{Q1:.2f},{kT2:.2f},{kH2:.2f},{kS2:.2f},{iftaxis},{ifkk},{ifok},{if_large}".format(
                            Np=Np,runtime=runtime,gamma0_inv=gamma0_inv,
                            Q0=Q0,Q1=Q1,kT2=kT2,kH2=kH2,kS2=kS2,
                            iftaxis=iftaxis,ifkk=ifkk,ifok=ifok,
                            if_large=if_large)
                        # print(params)
                        f.write(params)
                        f.write("\n")
            
