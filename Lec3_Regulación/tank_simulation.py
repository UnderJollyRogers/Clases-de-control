import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
import numpy as np
import math
import control as ctl

# Time
t0=0 # [s]
t_end=32000 # [s]
dt=100
t=np.arange(t0,t_end+dt,dt)

# Zero arrays for the tanks' volumes
volume_Tank1=np.zeros(len(t))
 
# Constants
r1 = 3
k = 0.01
Fin = 0.5
# Create volumes for
tfTank1 = ctl.tf([1], [np.pi*r1**2, k])
_, volume_Tank1 = ctl.step_response(tfTank1*0.5, T=t)

############################ ANIMATION #################################
frame_amount=len(t)

# Create the watertanks
radius=5 # [m]
volume_i=0 # [m^3]
volume_f=100 # [m^3]
dVol=10

def update_plot(num):
    # Tank 1
    tank_1.set_data([-radius,radius],
        [volume_Tank1[num],volume_Tank1[num]])

    tank_12.set_height(volume_Tank1[num])

    tnk_1.set_data(t[0:num],volume_Tank1[0:num])
    

    return tank_12,tank_1,tnk_1

# Set up your figure properties
fig=plt.figure(figsize=(10,9),dpi=120,facecolor=(0.8,0.8,0.8))
gs=gridspec.GridSpec(2,1)

ax0=fig.add_subplot(gs[0,0],facecolor=(0.9,0.9,0.9))
tank_1,=ax0.plot([],[],'r',linewidth=4)
tank_12 = plt.Rectangle([-5,0], 10, 0, facecolor="royalblue")
ax0.add_patch(tank_12)

plt.xlim(-radius,radius)
plt.ylim(volume_i,volume_f)
plt.xticks(np.arange(-radius,radius+1,radius))
plt.yticks(np.arange(volume_i,volume_f+dVol,dVol))
plt.ylabel('tank volume [m^3]')
plt.title('Tank 1')
# copyright=ax0.text(-radius,(volume_f+10)*3.2/3,'© Mark Misin Engineering',size=12)

# Create volume function
ax1=fig.add_subplot(gs[1,0], facecolor=(0.9,0.9,0.9))
tnk_1,=ax1.plot([],[],'blue',linewidth=3,label='Tank 1')
plt.xlim(t0,t_end)
plt.ylim(volume_i,volume_f)
plt.yticks(np.arange(volume_i,volume_f+1,dVol))
plt.xlabel('time [s]')
plt.ylabel('tank volume [m^3]')
plt.grid(True)
plt.legend(loc='upper right',fontsize='small')

plane_ani=animation.FuncAnimation(fig,update_plot, frames=frame_amount,interval=20,repeat=False,blit=True)
plt.show()
