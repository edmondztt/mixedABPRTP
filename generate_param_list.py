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
    f.write('# N_particles runtime Q0 kHT2 iftaxis ifkk ifok iflarge\n')
    for k2factor in np.array([2.0, 4.0, 8.0]):
            kHT2 = kHT20 * k2factor
            # for if_large in ["false", "true"]:
            for if_large in ["false"]:
                for ifok in ["false","true"]:
                    for ifkk in ["false","true"]:
                        params = "{Np:d},{runtime:d},{Q0:.2f},{kHT2:.2f},{iftaxis},{ifkk},{ifok},{if_large}".format(
                            Np=Np,runtime=runtime,
                            Q0=Q0,kHT2=kHT2,
                            iftaxis=iftaxis,ifkk=ifkk,ifok=ifok,
                            if_large=if_large)
                        # print(params)
                        f.write(params)
                        f.write("\n")
            
