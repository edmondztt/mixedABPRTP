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
DR = 0.1
N_particles = sys.argv[1]
N_particles = int(N_particles)
runtime = float(sys.argv[2])
Q0 = float(sys.argv[3])
Q1 = float(sys.argv[4])
kT2 = float(sys.argv[5])
kH2 = float(sys.argv[6])
kS2 = float(sys.argv[7])
if_taxis = (sys.argv[8]=="true")
print("if_taxis=", if_taxis)

# root_path = "/mnt/c/Users/wanxu/pheromone-modeling/"
root_path = ""
if if_taxis:
    path = root_path+"data-both/"
else:
    path = root_path+"data-kinesis/"

noise_Q = 0.01
print("N=",N_particles,", Q0=",Q0,", Q1=",Q1,", kT2=",kT2,", kH2=",kH2,", kS2=",kS2,", runtime=",runtime)

gsd_filename = path + "kinesis_N{0}_runtime{1}_Q0{2}_Q1{3}_kT2{4}_kH2{5}_kS2{6}_iftaxis{7}.gsd".format(N_particles, runtime,
    Q0, Q1, kT2, kH2, kS2, if_taxis)
print("gsd fname = ", gsd_filename)
fname_init = 'init.gsd'

cpu = hoomd.device.CPU()
simulation = hoomd.Simulation(device=cpu, seed=1)

dr = 0.1
dtheta = np.pi/180
rmax = 30 # 30 mm radius for dilute

X0 = 7.5 # initial position

flag_continue = False
if(not os.path.exists(gsd_filename)):
    flag_continue = False
    print(gsd_filename, " does not exist. try ", fname_init)
    if(not os.path.exists(fname_init)):
        print(fname_init, " does not exist. creating new config.")
        L = 2*rmax+1.0
        print('L=',L)
        X = np.random.rand(N_particles)+X0
        Y = np.random.rand(N_particles)
        Z = np.zeros_like(X)
        position = np.stack((X,Y,Z),axis=-1)
        frame = gsd.hoomd.Frame()
        frame.particles.N = N_particles
        frame.particles.position = position[0:N_particles]
        frame.particles.typeid = [0] * N_particles
        frame.configuration.box = [L, L, 0, 0, 0, 0]
        frame.particles.types = ['A']
        frame.particles.orientation = rand_unit_quaternion(N_particles)
        print("created {N:d} particles".format(N=len(frame.particles.position)))
        simulation.create_state_from_snapshot(frame)
    else:
        simulation.create_state_from_gsd(
            filename=fname_init
        )
else:
    flag_continue = True
    print("continue run from ", gsd_filename)
    simulation.create_state_from_gsd(
        filename=gsd_filename
    )

integrator = hoomd.md.Integrator(dt=dt)

overdamped_viscous = hoomd.md.methods.OverdampedViscous(
    filter=hoomd.filter.All())
integrator.methods.append(overdamped_viscous)
# nvt = hoomd.md.methods.ConstantVolume(
#     filter=hoomd.filter.All(), thermostat=hoomd.md.methods.thermostats.Bussi(kT=1.5)
# )
# integrator.methods.append(nvt)


mixed_active = hoomd.md.force.MixedActive(filter=hoomd.filter.All(), L=30*2)
mixed_active.mixed_active_force['A'] = (1,0,0)
mixed_active.active_torque['A'] = (0,0,0)
mixed_active.params['A'] = dict(kT1=1.0/600, kT2=kT2, kH1 = 1.0/60.0, kH2=kH2,
        kS1 = 1.0/30, kS2 = kS2, Q0=Q0, Q1=Q1, noise_Q = noise_Q, U0=0.2, U1=0.1, gamma0=1 / 30.0, c0_PHD = 1e-6, sigma_QC=2.5)
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


update_every_n = 30
simulation.operations.writers.append(
    hoomd.write.CustomWriter(
        action = print_sim_state(),
        trigger = trigger_every_n_sec()
    )
)

gsd_writer = hoomd.write.GSD(trigger=hoomd.trigger.Periodic(100),
                filename=gsd_filename,
                dynamic=['property'])
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

c_filename = "new_dilute_c_crosssection_agar.txt"
print("now read from c file")
mixed_active.set_concentration_field_file(c_filename)

nsteps = int(runtime/dt)
print("run for {0} timesteps".format(nsteps))
simulation.run(nsteps)

print("finished!")
