import numpy as np


# N_particles runtime Q0 Q1 kT2 kH2 kS2 

Np = int(100)
runtime = 1800
Q0 = 0.5
kHT20 = 1.0/30
# now use the same accumulation rate for H & T
gamma0_inv = 15
iftaxis = "true"
ifkk = "true"
ifok = "true"

fname = 'parameters.csv'
with open(fname, 'w') as f:
    f.write('# N_particles runtime Q0 kHT2 iftaxis ifkk ifok iflarge depth\n')
    # for Np in [100, 10000]:
    for Np in [100]:
        for Q0 in [0.5, 1.0, 0.2]:
            for k2factor in np.array([0.25, 1.0, 4.0, 16.0]):
                    kHT2 = kHT20 * k2factor
                    # for if_large in ["false", "true"]:
                    for if_large in ["false"]:
                        for ifok in ["false","true"]:
                            for ifkk in ["false","true"]:
                                # for iftaxis in ["false","true"]:
                                for iftaxis in ["true"]:
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
                            
