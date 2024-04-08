import hoomd
# import matplotlib
import math
import gsd.hoomd
import numpy as np
from random import randint
from random import uniform
import datetime, time
import os
import sys


# matplotlib.style.use('ggplot')
# import matplotlib_inline
# matplotlib_inline.backend_inline.set_matplotlib_formats('svg')

class print_sim_state(hoomd.custom.Action):
    def act(self, timestep):
        global init_time, last_output
        try:
            sec_remaining = int((self._state._simulation.final_timestep - timestep) / self._state._simulation.tps)
        except ZeroDivisionError:
            sec_remaining = 0
        print(
            "Time", str(datetime.timedelta(seconds=int(time.time() - init_time))),
            "| Step:", timestep,
            "| TPS:", str(round(float(self._state._simulation.tps),3)),
            "| ETR:", str(datetime.timedelta(seconds=sec_remaining))
        )
        last_output = time.time()

class trigger_every_n_sec(hoomd.trigger.Trigger):
    def __init__(self):
        hoomd.trigger.Trigger.__init__(self)

    def compute(self, timestep):
        global last_output, update_every_n
        return ((time.time() - last_output) >=update_every_n)

def rand_unit_quaternion(N, threeD=False):
    '''creates a unit quaternion to describe a 2 or 3 dimensional object; returns [w, x, y, z]'''
    orientation = []
    for i in range(N):
        roll = 0 if threeD == 0 else np.pi * uniform(-1,1) # rotation around x-axis
        pitch = 0 if threeD == 0 else np.pi * uniform(-1,1)# rotation around y-axis
        yaw = np.pi * uniform(-1,1) # rotation around z-axis
        v = np.array(
            [
                np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2), #qw
                np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2), #qx
                np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2), #qy
                np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) #qz
            ]
        )
        v = v / np.linalg.norm(v)
        orientation.append(np.array(v))
    return orientation



dt = 1e-3
sigma_tumble = 0.2*np.pi
DR = 0
N_particles = 10
N_particles = int(N_particles)
runtime = 1800
Q0 = 1.00
noise_Q = 0.01*Q0
kHT2 = 1.00
kklino = 2
if_taxis = True
if_klinokinesis = True
if_orthokinesis = True
if_large = False
depth = 8.5
print("if_taxis=", if_taxis)
print("if_klinokinesis=", if_klinokinesis)
print("if_orthokinesis=", if_orthokinesis)
print("if_large=", if_large)
print("depth=", depth)

gamma0_inv = 15
gamma0 = 1 / gamma0_inv

# both head and tail memory timescale is measured by their effects on AVA motor.
# the AVA activity seems to correlate strongly with single sensory neuron in real time, so we take both head and tail confidence to be 10s memory. head timescale from Bargmann 2015 Fig.2B
kT1 = 1.0/10.0
kH1 = 1.0/10.0
U0 = 0.064
U1 = 0.03
# sigma_QT = 1.5 # from titration data: let's say 1x there is O(0.1) factor
sigma_QT = 2.0
sigma_QH = 6.0 # from Fig.2E of Bargmann 2015: 1000x dilution result in 0.25 factor
c0 = 1e-6
dc0 = 1e-6 / 1 # take this as the typical c change rate
# timescale_across_plate = 30 / U0
# dc0 = 1e-5 / timescale_across_plate # typical large concentration increase rate ~ 2e-8. above this QH stim term will saturate

L0 = 6 # initial spread since stimulation starts at 5 min = 300 s.

Q1 = 1.0 # now we do not use Q1 really. temporarily kept here for interface legacy
kS1 = 1/300 # now we do not use S really. temporarily kept here for interface legacy
kS2 = 0.0 # now we do not use S really. temporarily kept here for interface legacy
print("N=",N_particles, ", Q0=",Q0, ", kH2=kT2=",kHT2,", runtime=",runtime)

if if_large:
    rmax = 40 # 40 mm radius for large dist
    # X0 = 25 # for large dist symm setting case 2
    X0 = 0
    c_filename = "mylarge_dist_c_crosssection_agar"+str(depth)+"mm_30min.txt"
    root_path = "data/large/"
else:
    rmax = 30 # 30 mm radius for dilute
    X0 = 0
    c_filename = "mydilute_c_crosssection_agar"+str(depth)+"mm_30min.txt"
    root_path = "data/dilute/"

root_path += "agar"+str(depth)+"mm/"
if not os.path.exists(root_path):
    os.makedirs(root_path, exist_ok=True)

path = root_path
if if_klinokinesis:
    path += "kk_"
else:
    path += "kr_"
if if_orthokinesis:
    path += "ok_"
else:
    path += "or_"
if if_taxis:
    path += "taxis_"
else:
    path += "notaxis_"


gsd_filename = "testN{0}_runtime{1}_Q0{2:.2f}_kHT2{3:.2f}_noiseQ{4:.2f}_kklino{5:.1f}_depth{6}mm.gsd".format(N_particles, runtime, Q0, kHT2, noise_Q, kklino, depth)
print("gsd fname = ", gsd_filename)
fname_init = 'init.gsd'

cpu = hoomd.device.CPU()
simulation = hoomd.Simulation(device=cpu, seed=1)


flag_continue = False
if(not os.path.exists(gsd_filename)):
    flag_continue = False
    print(gsd_filename, " does not exist. try ", fname_init)
    if(not os.path.exists(fname_init)):
        print(fname_init, " does not exist. creating new config.")
        L = 2*rmax+1.0
        print('L=',L)
        X = L0*(np.random.rand(N_particles)-0.5)+X0
        Y = L0*(np.random.rand(N_particles)-0.5)
        Z = np.zeros_like(X)
        # X = np.array([2.3])
        # Y = np.array([25])
        # Z = np.zeros_like(X)
        position = np.stack((X,Y,Z),axis=-1)
        print(position)
        frame = gsd.hoomd.Frame()
        frame.particles.N = N_particles
        frame.particles.position = position[0:N_particles]
        print(np.shape(frame.particles.position))
        frame.particles.typeid = [0] * N_particles
        frame.configuration.box = [L, L, 0, 0, 0, 0]
        frame.particles.types = ['A']
        theta = np.pi/3
        frame.particles.orientation = rand_unit_quaternion(N_particles)
        # frame.particles.orientation = [np.cos(theta/2), 0, 0, np.sin(theta/2)]
        print("created {N:d} particles".format(N=len(frame.particles.position)))
        # simulation.timestep = 1459800
        simulation.create_state_from_snapshot(frame)
    else:
        simulation.create_state_from_gsd(
            filename=fname_init
        )
else:
    flag_continue = True
    print("continue run from ", gsd_filename)
    simulation.create_state_from_gsd(
        filename=gsd_filename,
        frame=14598
    )

integrator = hoomd.md.Integrator(dt=dt)

overdamped_viscous = hoomd.md.methods.OverdampedViscous(
    filter=hoomd.filter.All())
integrator.methods.append(overdamped_viscous)
# nvt = hoomd.md.methods.ConstantVolume(
#     filter=hoomd.filter.All(), thermostat=hoomd.md.methods.thermostats.Bussi(kT=1.5)
# )
# integrator.methods.append(nvt)


mixed_active = hoomd.md.force.MixedActive(filter=hoomd.filter.All(), L=rmax*2, 
                    is_klinokinesis=if_klinokinesis, is_orthokinesis=if_orthokinesis)
mixed_active.mixed_active_force['A'] = (1,0,0)
mixed_active.active_torque['A'] = (0,0,0)
mixed_active.params['A'] = dict(kT1=kT1, kT2=kHT2, kH1=kH1, kH2=kHT2,
        kS1 = kS1, kS2 = kS2, Q0=Q0, Q1=Q1, kklino=kklino, noise_Q = noise_Q, U0=U0, U1=U1, gamma0=gamma0, 
        c0_PHD=c0, dc0=dc0, sigma_QH=sigma_QH, sigma_QT=sigma_QT)
# mixed_active.kT1['A'] = 1.0 / 600 # Q tail decays in 10 min.
# mixed_active.kT2['A'] = 1
# mixed_active.kH1['A'] = 0.1 # Q head decays in 10 s.
# mixed_active.kH2['A'] = 1
# mixed_active.kS1['A'] = 1.0 / 30 # S decays in 30 s.??
# mixed_active.kS2['A'] = 0.1 
# mixed_active.Q0['A'] = 0.3 # below this will tumble
# mixed_active.Q1['A'] = 0.7 # above this will accelerate
# mixed_active.noise_Q['A'] = 0.03 
# mixed_active.U0['A'] = 20 # base velocity ~ 20 um/s
# mixed_active.U1['A'] = 10 # faster ~ +10 um/s
# mixed_active.gamma0['A'] = 1 / 10.0 # tumble about every 10 s. ??
# mixed_active.c0_PHD['A'] = 0.1e-5 # the concentration level that PHD will detect

if not if_klinokinesis:
    sigma_tumble = -1 # < 0 means no kinesis only random turning with unform angle distribution
rotational_diffusion_tumble_updater = mixed_active.create_diffusion_tumble_updater(
    trigger=10, rotational_diffusion=DR, tumble_angle_gauss_spread=sigma_tumble, iftaxis=if_taxis)
simulation.operations += rotational_diffusion_tumble_updater
integrator.forces.append(mixed_active)

# active = hoomd.md.force.Active(filter=hoomd.filter.All())
# active.active_force['A'] = (1,0,0)
# active.active_torque['A'] = (0,0,0)
# rotational_diffusion_updater = active.create_diffusion_updater(
#     trigger=10, rotational_diffusion=DR)
# simulation.operations += rotational_diffusion_updater
# integrator.forces.append(active)

simulation.operations.integrator = integrator


update_every_n = 10
simulation.operations.writers.append(
    hoomd.write.CustomWriter(
        action = print_sim_state(),
        trigger = trigger_every_n_sec()
    )
)

gsd_writer = hoomd.write.GSD(trigger=hoomd.trigger.Periodic(100),
                filename=gsd_filename,
                dynamic=['property','momentum'])
simulation.operations.writers.append(gsd_writer)

walls = [hoomd.wall.Cylinder(radius=rmax, axis=(0,0,1), inside=True)]
shifted_lj_wall = hoomd.md.external.wall.ForceShiftedLJ(
    walls=walls)
shifted_lj_wall.params['A'] = {
    "epsilon": 10.0, "sigma": 1.0, "r_cut": 1.0*2**(1.0/6.0)}
integrator.forces.append(shifted_lj_wall)

init_time = time.time()
last_output = init_time


simulation.run(0) # have to first do run(0) so that the mixed_active cpp_obj is attached and then we can call cpp methods of it

print("now read from c file")
mixed_active.set_concentration_field_file(c_filename)

nsteps = int(runtime/dt)
print("run for {0} timesteps".format(nsteps))
simulation.run(nsteps, write_at_start=True)

print("finished!")
