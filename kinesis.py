import hoomd
# import matplotlib
import math
import gsd.hoomd
import numpy as np
from random import randint
from random import uniform
import datetime, time
import os


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
DR = 0.5
runtime = 0.01

gsd_filename = 'test.gsd'
fname_init = 'init.gsd'

cpu = hoomd.device.CPU()
simulation = hoomd.Simulation(device=cpu, seed=1)

if(not os.path.exists(gsd_filename)):
    if(not os.path.exists(fname_init)):
        m = 4
        N_particles = 4 * m**2
        spacing = 10
        K = math.ceil(N_particles ** (1 / 2))
        print('K=',K)
        L = K * spacing
        print('L=',L)
        x = np.linspace(-L / 2, L / 2, K, endpoint=False)
        X, Y = np.meshgrid(x,x)
        X = X.flatten()
        Y = Y.flatten()
        Z = np.zeros_like(X)
        position = np.stack((X,Y,Z),axis=-1)
        frame = gsd.hoomd.Frame()
        frame.particles.N = N_particles
        frame.particles.position = position[0:N_particles]
        frame.particles.typeid = [0] * N_particles
        frame.configuration.box = [L, L, 0, 0, 0, 0]
        frame.particles.types = ['A']
        frame.particles.orientation = rand_unit_quaternion(N_particles)
        with gsd.hoomd.open(name=fname_init, mode='x') as f:
            f.append(frame)
    simulation.create_state_from_gsd(
        filename=fname_init
    )
else:
    simulation.create_state_from_gsd(
        filename=gsd_filename
    )

integrator = hoomd.md.Integrator(dt=dt)
cell = hoomd.md.nlist.Cell(buffer=0.4)
lj = hoomd.md.pair.LJ(nlist=cell)
lj.params[('A', 'A')] = dict(epsilon=0, sigma=0.1)
lj.r_cut[('A', 'A')] = 0.1
integrator.forces.append(lj)
overdamped_viscous = hoomd.md.methods.OverdampedViscous(
    filter=hoomd.filter.All())
integrator.methods.append(overdamped_viscous)
# nvt = hoomd.md.methods.ConstantVolume(
#     filter=hoomd.filter.All(), thermostat=hoomd.md.methods.thermostats.Bussi(kT=1.5)
# )
# integrator.methods.append(nvt)


mixed_active = hoomd.md.force.MixedActive(filter=hoomd.filter.All())
mixed_active.mixed_active_force['A'] = (1,0,0)
mixed_active.active_torque['A'] = (0,0,0)
mixed_active.params['A'] = dict(kT1=1.0/600, kT2=1, kH1 = 0.1, kH2=1.0,
        kS1 = 1.0/30, kS2 = 0.1, Q0=0.3, Q1=0.7, noise_Q = 0.03, U0=20, U1=10, gamma0=1 / 10.0, c0_PHD = 0.1e-5)
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
    trigger=10, rotational_diffusion=DR, tumble_angle_gauss_spread=sigma_tumble)
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

gsd_writer = hoomd.write.GSD(trigger=hoomd.trigger.Periodic(1_00),
                      filename=gsd_filename)
simulation.operations.writers.append(gsd_writer)


init_time = time.time()
last_output = init_time

simulation.run(0)

c_filename = "dilute_c_crosssection_agar.txt"
dr = 0.1
dtheta = np.pi/180
rmax = 100
mixed_active.set_grid_size(dr, dtheta, rmax)
mixed_active.set_concentration_field_file(c_filename)

simulation.run(int(runtime/dt))
