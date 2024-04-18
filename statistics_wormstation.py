import numpy as np
import gsd.hoomd
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as mcolors
import re
import os


fname_frac_succ = 'frac_succ_alldata.txt'
frac_succ_alldata = []
frac_fail_alldata = []

def process_data(gsd_filename, N_particles, figname_vmag_hist, figname_Nfound, 
                 figname_VRVT, figname_traj_succ, figname_traj_fail, figname_retention):
    time_unit = 0.1 # s per frame
    ####################### now start reading data ####################
    traj=gsd.hoomd.open(gsd_filename,'r')
    Ntotal = N_particles
    box = traj[0].configuration.box
    print(box)
    L = box[0]
    R1 = R2 = L/2/(np.sqrt(2)+1)
    center1 = np.array([-L/2+R1,0,0])
    center2 = np.array([L/2-R2,0,0])
    center_source = np.array([-15,0,0])
    R_source = 2
    ####################### now gather statistics and plot ##################
    # now gather statistics and plot
    U_all = []
    succ_count = np.zeros((Ntotal,1))
    VR = []
    VT = []
    # Nframes = 6000
    Nframes = len(traj)
    print(Nframes)
    time = time_unit * np.linspace(0,Nframes-1,Nframes)
    Nfound_all = np.zeros_like(time)
    inds_still_searching_alltime = []
    frame_found = Nframes*2*np.ones((Ntotal,))
    dist_to_source = []
    for frame in range(int(Nframes)):
    # for frame in range(3):
        # print(frame)
        data = traj[frame]
        p = data.particles
        t = p.tumble
        pos = p.position
        r = pos-center_source
        tmp_succ = np.linalg.norm(r,axis=1)<R_source
        U = t[:,1]
        U_all.append(U)
        v = p.velocity
        rr = np.linalg.norm(r,axis=1)
        dist_to_source.append(rr)
        r_hat = r / rr[:, np.newaxis]
        vr = v * r_hat
        vr = np.sum(vr, axis=1)
        VR.append(vr)
        VT.append(U)
        if tmp_succ.any():
            succ_count[tmp_succ] += 1
        # now record worms who are still searching
        inds_still_searching = np.where(succ_count==0)[0]
        inds_still_searching_alltime.append(inds_still_searching)
        inds_curr_found = np.where(succ_count>0)[0]
        Nfound_all[frame] = len(inds_curr_found)
        # print(inds_curr_found)
        frame_found[inds_curr_found] = np.minimum(frame_found[inds_curr_found], frame*np.ones((len(inds_curr_found),)))
        # N1 = np.sum(np.linalg.norm(pos - center1,axis=1)<R1)
        # N2 = np.sum(np.linalg.norm(pos - center2,axis=1)<R2)
        # N_region1_all.append(N1)
        # N_region2_all.append(N2)
    U_all = np.array(U_all)
    VR = np.array(VR)
    VT = np.array(VT)
    dist_to_source = np.array(dist_to_source)
    ################### process the vmag hist & VRVT scatter data ##############
    ind_succ = np.where(succ_count>0)[0]
    ind_fail = np.where(succ_count==0)[0]
    Nsucc = len(ind_succ)
    Nfail = len(ind_fail)
    VR_succ = []; VR_fail = [];
    VT_succ = []; VT_fail = [];
    for i in range(Nsucc):
        pid = ind_succ[i]
        curr_frame_found = int(frame_found[pid])
        # print(curr_frame_found)
        VR_succ.append(VR[:curr_frame_found,pid])
        VT_succ.append(VT[:curr_frame_found,pid])
    for i in range(Nfail):
        pid = ind_fail[i]
        VR_fail.append(VR[:,pid])
        VT_fail.append(VT[:,pid])
    frac_succ_alldata.append(Nsucc/N_particles)
    frac_fail_alldata.append(Nfail/N_particles)

    if Nfail>0:
        VR_fail_flat = np.concatenate(VR_fail)
        VT_fail_flat = np.concatenate(VT_fail)
        U_fail_flat = VT_fail_flat
        NVtotal_fail = len(VR_fail_flat)
        mean_VT_fail = np.mean(VT_fail_flat)
        max_VR_fail = np.max(VR_fail_flat)

        n1_fail = np.sum(np.logical_and(VR_fail_flat>=0, VT_fail_flat>=mean_VT_fail)) / NVtotal_fail
        n2_fail = np.sum(np.logical_and(VR_fail_flat<0, VT_fail_flat>=mean_VT_fail)) / NVtotal_fail
        n3_fail = np.sum(np.logical_and(VR_fail_flat<0, VT_fail_flat<mean_VT_fail)) / NVtotal_fail
        n4_fail = np.sum(np.logical_and(VR_fail_flat>=0, VT_fail_flat<mean_VT_fail)) / NVtotal_fail
        n1_fail_str = "n1={:.3f}".format(n1_fail)
        n2_fail_str = "n2={:.3f}".format(n2_fail)
        n3_fail_str = "n3={:.3f}".format(n3_fail)
        n4_fail_str = "n4={:.3f}".format(n4_fail)

    if Nsucc>0:
        VR_succ_flat = np.concatenate(VR_succ)
        VT_succ_flat = np.concatenate(VT_succ)
        U_succ_flat = VT_succ_flat
        NVtotal_succ = len(VR_succ_flat)
        mean_VT_succ = np.mean(VT_succ_flat)
        max_VR_succ = np.max(VR_succ_flat)

        n1_succ = np.sum(np.logical_and(VR_succ_flat>=0, VT_succ_flat>=mean_VT_succ)) / NVtotal_succ
        n2_succ = np.sum(np.logical_and(VR_succ_flat<0, VT_succ_flat>=mean_VT_succ)) / NVtotal_succ
        n3_succ = np.sum(np.logical_and(VR_succ_flat<0, VT_succ_flat<mean_VT_succ)) / NVtotal_succ
        n4_succ = np.sum(np.logical_and(VR_succ_flat>=0, VT_succ_flat<mean_VT_succ)) / NVtotal_succ
        n1_succ_str = "n1={:.3f}".format(n1_succ)
        n2_succ_str = "n2={:.3f}".format(n2_succ)
        n3_succ_str = "n3={:.3f}".format(n3_succ)
        n4_succ_str = "n4={:.3f}".format(n4_succ)
    ########### plot typical trajectories for succ. if any ######
    if Nsucc>0:
        pid = ind_succ[0]
        curr_frame_found = int(frame_found[pid])
        traj_succ = []
        for frame in range(curr_frame_found):
            data = traj[frame]
            p = data.particles
            pos = p.position
            traj_succ.append(pos[pid, :2])
        traj_succ = np.array(traj_succ)
        plt.figure(figsize=(3,3))
        plt.scatter(traj_succ[:,0], traj_succ[:,1],c=np.arange(len(traj_succ)), cmap='viridis', norm=mcolors.Normalize(vmin=0, vmax=int(Nframes/2)))
        cbar = plt.colorbar()
        ticks = np.linspace(0, Nframes/2, num=6)  # Choose appropriate number of ticks
        tick_labels = [f"{int(tick/10)}s" for tick in ticks]
        # Set ticks and labels to the colorbar
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(tick_labels)
        plt.axis('equal')
        plt.savefig(figname_traj_succ, dpi=100,bbox_inches='tight')
    ########### plot typical trajectories for fail. if any
    if Nfail>0:
        dist_to_source_fail = dist_to_source[:,ind_fail]
        ind_farthest = np.unravel_index(np.argmax(dist_to_source_fail), dist_to_source_fail.shape)[1]
        print(ind_farthest)
        traj_fail = []
        for frame in range(Nframes):
            data = traj[frame]
            p = data.particles
            pos = p.position
            traj_fail.append(pos[ind_farthest, :2])
        traj_fail = np.array(traj_fail)
        plt.figure(figsize=(3,3))
        plt.scatter(traj_fail[:,0], traj_fail[:,1],c=np.arange(len(traj_fail)), cmap='viridis', norm=mcolors.Normalize(vmin=0, vmax=int(Nframes)))
        cbar = plt.colorbar()
        ticks = np.linspace(0, Nframes, num=6)  # Choose appropriate number of ticks
        tick_labels = [f"{int(tick/10)}s" for tick in ticks]
        # Set ticks and labels to the colorbar
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(tick_labels)
        plt.axis('equal')
        plt.savefig(figname_traj_fail, dpi=100,bbox_inches='tight')
    ############################## now make the plots #######################
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    if Nsucc>0:
        nU_succ, edges, patches = plt.hist(U_succ_flat[U_succ_flat>0],100,density=True);
        max_nU_succ = np.max(nU_succ)
    else:
        max_nU_succ = 0
    plt.title("Vmag (Nsucc={:d})".format(Nsucc))
    plt.ylim([0,50])
    plt.subplot(1,2,2)
    if Nfail>0:
        nU_fail, edges, patches = plt.hist(U_fail_flat[U_fail_flat>0],100,density=True);
        max_nU_fail = np.max(nU_fail)
    else:
        max_nU_fail = 0
    max_nU_both = max(max_nU_fail, max_nU_succ)
    # plt.ylim([0,max_nU_both*1.2])
    plt.ylim([0,50])
    plt.title("Vmag (Nfail={:d})".format(Nfail))
    plt.savefig(figname_vmag_hist, dpi=100,bbox_inches='tight')

    plt.figure(figsize=(10,5))
    plt.plot(time, Nfound_all/Ntotal, label='fraction found')
    plt.xlabel('time [s]')
    plt.title('fraction of found so far')
    plt.savefig(figname_Nfound, dpi=100,bbox_inches='tight')

    myscattersize = 0.001

    plt.figure(figsize=(10,5))
    factor_shift_posx = 0.6
    plt.subplot(1,2,1)
    plt.title('Vtotal vs VR for succ')
    if Nsucc>0:
        for i in range(Nsucc):
            plt.scatter(VR_succ[i], VT_succ[i], s=myscattersize)
        plt.text(max_VR_succ*factor_shift_posx,mean_VT_succ*1.05,n1_succ_str,fontweight='bold')
        plt.text(-max_VR_succ,mean_VT_succ*1.05,n2_succ_str,fontweight='bold')
        plt.text(-max_VR_succ,mean_VT_succ*.95,n3_succ_str,fontweight='bold')
        plt.text(max_VR_succ*factor_shift_posx,mean_VT_succ*.95,n4_succ_str,fontweight='bold')
    plt.subplot(1,2,2)
    plt.title('Vtotal vs VR for fail')
    if Nfail>0:
        for i in range(Nfail):
            plt.scatter(VR_fail[i], VT_fail[i], s=myscattersize)
        plt.text(max_VR_fail*factor_shift_posx,mean_VT_fail*1.05,n1_fail_str,fontweight='bold')
        plt.text(-max_VR_fail,mean_VT_fail*1.05,n2_fail_str,fontweight='bold')
        plt.text(-max_VR_fail,mean_VT_fail*.95,n3_fail_str,fontweight='bold')
        plt.text(max_VR_fail*factor_shift_posx,mean_VT_fail*.95,n4_fail_str,fontweight='bold')
    plt.savefig(figname_VRVT, dpi=100,bbox_inches='tight')


parameter_file_path = 'logfinished-batch1.csv'
fig_dir = "data-production2/figures/"
root_path = "data-production2/"

tauHT1 = 60

mode = 1
if(mode==1):
    df = pd.read_csv(parameter_file_path, comment='#', delimiter=' ', dtype={
        'N_particles': int,
        'runtime': float,
        'noise_Q': float,
        'kHT2': float,
        'DR': float,
        'if_head': str,
        'iftail': str,
        'iftaxis': str,
        'ifkk': str,
        'ifok': str,
        'plate_condition': str,
        'depth': float
    })
    df.columns = ['N_particles', 'runtime', 'noise_Q', 'kHT2', 'DR', 'if_head', 'iftail', 'iftaxis', 'ifkk', 'ifok', 'plate_condition', 'depth']
    for index, row in df.iterrows():
        N_particles = row['N_particles']
        runtime = row['runtime']
        noise_Q = row['noise_Q']
        kHT2 = row['kHT2']
        DR = row['DR']
        if_head = (row['if_head']=="true")
        if_tail = (row['iftail']=="true")
        iftaxis = (row['iftaxis']=="true")
        ifkk = (row['ifkk']=="true")
        ifok = (row['ifok']=="true")
        plate_condition = row['plate_condition']
        depth = row['depth']
        if plate_condition == "smalldilute":
            folder = "dilute"
        else:
            folder = plate_condition
        if if_tail:
            kT2 = kHT2
        else:
            kT2 = 0.0
        if if_head:
            kH2 = kHT2
        else:
            kH2 = 0.0
        path=root_path
        path += folder+"/"
        path += "agar"+str(depth)+"mm/"
        prefix = ""
        if if_head:
            prefix += "head_"
        else:
            prefix += "nohead_"
        if if_tail:
            prefix += "tail_"
        else:
            prefix += "notail_"
        if iftaxis:
            prefix += "taxis_"
        if ifkk:
            prefix += "kk_"
        else:
            prefix += "kr_"
        if ifok:
            prefix += "ok_"
        else:
            prefix += "or_"

        gsd_filename = path + prefix + "N{0}_tauHT1{2:.1f}_kHT2{3:.2f}_noiseQ{4:.2f}_DR{5:.2f}_depth{6}mm.gsd".format(N_particles, runtime, tauHT1, kHT2, noise_Q, DR,depth)
        print("gsd fname = ", gsd_filename)
        suffix = "N{0}_tauHT1{2:.1f}_kHT2{3:.2f}_noiseQ{4:.2f}_DR{5:.2f}_depth{6}mm.png".format(N_particles, runtime, tauHT1, kHT2, noise_Q, DR,depth).format(N_particles, runtime, kH2, noise_Q, DR,depth)
        figname_vmag_hist = fig_dir + folder + "/" + "vmaghist_" + prefix + suffix
        figname_Nfound = fig_dir + folder + "/" + "Nfound_" + prefix + suffix
        figname_retention = fig_dir + folder + "/" + "retention_" + prefix + suffix
        figname_VRVT = fig_dir + folder + "/" + "VRVT" + prefix + suffix
        figname_traj_succ = fig_dir + folder + "/" + "traj_succ" + prefix + suffix
        figname_traj_fail = fig_dir + folder + "/" + "traj_fail" + prefix + suffix
        process_data(gsd_filename, N_particles, figname_vmag_hist, figname_Nfound, 
                 figname_VRVT, figname_traj_succ, figname_traj_fail, figname_retention)
        print("finished proc file : ", gsd_filename)
        plt.close()
else:
    with open(parameter_file_path, 'r') as file:
        for line in file:
            gsd_filename = line.strip()
            png_path = os.path.basename(gsd_filename).replace('.gsd', '.png')
            print(png_path)
            path_parts = gsd_filename.split(os.sep)
            data_index = path_parts.index('data')
            folder = path_parts[data_index + 1]
            print("Folder :", folder)
            figname_vmag_hist = fig_dir + folder + "/" + "vmaghist_" + png_path
            figname_Nfound = fig_dir + folder + "/" + "Nfound_" + png_path
            figname_retention = fig_dir + folder + "/" + "retention_" + png_path
            figname_VRVT = fig_dir + folder + "/" + "VRVT" + png_path
            figname_traj_succ = fig_dir + folder + "/" + "traj_succ" + png_path
            figname_traj_fail = fig_dir + folder + "/" + "traj_fail" + png_path
            print(gsd_filename)
            print(figname_vmag_hist)
            match = re.search(r'N(\d+)', gsd_filename)
            if match:
                N_particles = int(match.group(1))
                print(f"Np = {N_particles}")
            else:
                print("No number found after 'N'")
            process_data(gsd_filename, N_particles, figname_vmag_hist, figname_Nfound, 
                    figname_VRVT, figname_traj_succ, figname_traj_fail, figname_retention)
            print("finished proc file : ", gsd_filename)
            plt.close()
        

np.savetxt(fname_frac_succ, frac_succ_alldata, fmt='%.2f', delimiter=',')