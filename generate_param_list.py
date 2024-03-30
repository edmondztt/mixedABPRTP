import numpy as np


# N_particles runtime Q0 Q1 kT2 kH2 kS2 

Np = int(100)
runtime = 1800
Q0 = 1.0
kHT20 = 1.0/15
# now use the same accumulation rate for H & T
gamma0_inv = 15
iftaxis = "true"
ifkk = "true"
ifok = "true"

fname = 'parameters.csv'
with open(fname, 'w') as f:
    f.write('# N_particles runtime Q0 kHT2 iftaxis ifkk ifok iflarge depth\n')
    for Np in [100, 10000]:
        for k2factor in np.array([1.0, 2.0, 4.0, 8.0]):
                kHT2 = kHT20 * k2factor
                for if_large in ["false", "true"]:
                    for ifok in ["false","true"]:
                        for ifkk in ["false","true"]:
                            for iftaxis in ["false","true"]:
                                for depth in ["8.5", "8.9"]:
                                    params = "{Np:d},{runtime:d},{Q0:.2f},{kHT2:.2f},{iftaxis},{ifkk},{ifok},{if_large},{depth}".format(
                                        Np=Np,runtime=runtime,
                                        Q0=Q0,kHT2=kHT2,
                                        iftaxis=iftaxis,ifkk=ifkk,ifok=ifok,
                                        if_large=if_large,
                                        depth=depth)
                                    # print(params)
                                    f.write(params)
                                    f.write("\n")
                        
