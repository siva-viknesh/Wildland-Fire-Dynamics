# ************************ WILD FIRE SOLVER - 2D UNSTEADY PDE SOLVER - POST PROCESS  ********************************* #
#   Author  : SIVA VIKNESH S.,
#   Address : SCI INSTITUTE, UNIVERSITY OF UTAH, SALT LAKE CITY, UTAH, USA
# ******************************************************************************************************************** #
import os
import numpy as np
#import cupy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
import matplotlib.cm as cm
import matplotlib.animation as animation
from scipy.linalg import circulant, toeplitz, inv
import h5py
# -------------------------------------------------------------------------------------------------------------------- #

print ("*"*85)
print ("\n WILD FIRE SIMULATION DATA ARE EXTRACT TO DECIPHER THE DYNAMICS......... \n")

xmin  =  0.0
xmax  =  2.0
Nx    =  256
dx    = (xmax - xmin) / Nx

ymin  =  0.0
ymax  =  1.0                                          
Ny    =  128
dy    = (ymax - ymin) / Ny

dt    = 1e-8                                               # TIME STEP
Nt    = int(5.0/dt)                                        # NO. OF TIME STEPS
NIT   = 1e4                                               # TIME INTERVAL FOR SVAING THE DATA
Nskip = 1
 
x = np.linspace(xmin, xmax, num=Nx)
y = np.linspace(ymin, ymax, num=Ny)
X, Y = np.meshgrid(x, y)

# -------------------------------------------------------------------------------------------------------------------- #

data_file = []
for file in os.listdir(os.getcwd() + '/'):
    if file.endswith('.h5'):
        data_file.append(file)

Nfiles   = len(data_file) -1                                # NO. OF FILES TO READ

print ("NO. OF DATA FILES =", Nfiles)
Nfiles   = int(Nfiles /Nskip)
print ("NO. OF DATA FILES TO READ =", Nfiles)

# -------------------------------------------------------------------------------------------------------------------- #
result = []  # Fuel Burning rate

right_front  = []
left_front   = []
top_front    = []
bottom_front = []

for n in range (1, Nfiles):
    file_name = "Data_" + str(int(n*Nskip*NIT)) + ".h5"

    print ("*"*85)
    print ('\nREADING THE DATA FILE: ', file_name)
    print ('\n')

    with h5py.File(file_name, 'r') as hf:
        Fuel = np.array(hf['Fuel'][()])
        hf.close()
    cs = plt.contour(X, Y, Fuel, [0.99])
    p1 = cs.collections[0].get_paths()[0]  # grab the 1st path
    xy = p1.vertices
    del cs, p1

    result.append(np.trapz(xy[:,1], x = xy[:, 0]))
    right_front.append(np.max(xy[:, 0]))
    left_front.append(np.min(xy[:, 0]))
    top_front.append(np.max(xy[:, 1]))
    bottom_front.append(np.min(xy[:, 1]))

time = np.linspace(1, Nfiles-1, Nfiles-1)
fuel_burn_rate = np.array(result)/ np.max(np.array(result))
right_front    = np.array(right_front)
left_front     = np.array(left_front)
top_front      = np.array(top_front)
bottom_front   = np.array(bottom_front)

data = np.vstack((time, fuel_burn_rate , right_front, left_front, top_front, bottom_front))

np.savetxt("flame_location.dat", np.transpose(data), fmt = '%10.6f')
plt.close('all')
plt.subplot(2, 2, 1)
plt.plot(time, np.array(right_front),  '-o', label ='Right front')
plt.xlabel('t') 
plt.ylabel('x') 
plt.grid()
plt.legend()


plt.subplot(2, 2, 2)
plt.plot(time, np.array(left_front), '-o', label ='Left front')
plt.xlabel('t') 
plt.ylabel('x') 
plt.grid()
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(time, np.array(top_front), '-o', label ='Top front')
plt.xlabel('t') 
plt.ylabel('y') 
plt.grid()
plt.legend()


plt.subplot(2, 2, 4)
plt.plot(time, np.array(bottom_front), '-o', label ='Bottom front')
plt.xlabel('t') 
plt.ylabel('y') 
plt.grid()
plt.legend()

plt.tight_layout()

plt.savefig("fig.png")

tolerance = 1

vel_right   = np.diff (right_front)  / (dt*NIT)
vel_left    = np.abs(np.diff (left_front))   / (dt*NIT)
vel_top     = np.diff (top_front)    / (dt*NIT)
vel_bottom  = np.abs(np.diff (bottom_front)) / (dt*NIT)


vel_right_mean = np.mean(vel_right [vel_right >= tolerance])
time_right     = np.size((vel_right [vel_right >= tolerance]))* (dt*NIT)

vel_left_mean = np.mean(vel_left  [vel_left >= tolerance])
time_left     = np.size((vel_left [vel_left >= tolerance]))* (dt*NIT)

vel_top_mean = np.mean(vel_top  [vel_top >= tolerance])
time_top     = np.size((vel_top [vel_top >= tolerance]))* (dt*NIT)

vel_bot_mean = np.mean(vel_bottom  [vel_bottom >= tolerance])
time_bot     = np.size((vel_bottom [vel_bottom >= tolerance]))* (dt*NIT)

print(time_right, time_left, time_top, time_bot)
print(vel_right_mean, vel_left_mean, vel_top_mean, vel_bot_mean)
