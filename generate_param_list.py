import numpy as np


# N_particles runtime Q0 Q1 kT2 kH2 kS2 

Np = int(100)
runtime = 1800
# Q0 = 1.0
tauHT1 = 60
# now use the same accumulation rate for H & T
gamma0_inv = 15
if_head = "true"
ifkk = "true"
ifok = "true"
if_large = "false"

fname = 'parameters.csv'
with open('logfinished.csv', 'w') as f:
    f.write('N_particles tauHT1 noise_Q kHT2 DR if_head ifkk ifok plate_condition iftail depth\n')
with open(fname, 'w') as f:
    f.write('N_particles tauHT1 noise_Q kHT2 DR if_head ifkk ifok plate_condition iftail depth\n')
    # for Np in [100, 10000]:
    for Np in [100]:
        # for Q0 in [0.5, 1.0, 0.2]:
        for noise_Q in [0.1, 1.0]:
        # for noise_Q in [0.1, 0.5, 1.0]:
            # for k2factor in np.array([ 1.0, 2.0, 5.0]):
            for k2factor in np.array([0.1, 1.0, 5.0]):
            # for k2factor in np.array([0.2, 1.0, 5.0]):
#                    kHT2 = kHT20 * k2factor
                    # for kklino in [1.0]:
                    for DR in [1/50]:
                        # for if_large in ["false", "true"]:
                        # for plate_condition in ["small", "large", "smalldilute"]:
                        # for plate_condition in ["small", "large"]:
                        for plate_condition in ["small", "large"]:
                            for iftail in ["true","false"]:
                            # for ifok in ["false","true"]:
                            # for ifok in ["true"]:
                                for ifkk in ["true"]:
                                    # for if_head in ["false"]:
                                    for if_head in ["true", "false"]:
                                        for depth in ["8.5"]:
                                        # for depth in ["8.5", "8.9"]:
                                            params = "{Np:d},{tauHT1:.1f},{noise_Q:.2f},{kHT2:.2f},{DR:.2f},{if_head},{ifkk},{ifok},{plate_condition},{iftail},{depth}".format(
                                                Np=Np,tauHT1=tauHT1,
                                                noise_Q=noise_Q,kHT2=k2factor, DR=DR,
                                                if_head=if_head,ifkk=ifkk,ifok=ifok,
                                                plate_condition=plate_condition,iftail=iftail,
                                                depth=depth)
                                            # print(params)
                                            f.write(params)
                                            f.write("\n")
                            
