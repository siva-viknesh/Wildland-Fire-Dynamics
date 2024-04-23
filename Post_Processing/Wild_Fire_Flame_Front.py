# ************************ WILD FIRE SOLVER - 2D UNSTEADY PDE SOLVER - POST PROCESS  ********************************* #
# Author  : SIVA VIKNESH
# Email   : siva.viknesh@sci.utah.edu / sivaviknesh14@gmail.com
# Address : SCI INSTITUTE, UNIVERSITY OF UTAH, SALT LAKE CITY, UTAH, USA
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

xmin  = -2.0
xmax  =  2.0
Nx    =  256
dx    = (xmax - xmin) / Nx

ymin  = -2.0
ymax  =  2.0                                          
Ny    =  256
dy    = (ymax - ymin) / Ny

dt    = 1e-7                                               # TIME STEP
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

Nfiles   = 70#len(data_file) -1                                # NO. OF FILES TO READ

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

fig = plt.figure()
ax = plt.axes()
plt.plot(time, np.array(right_front),  '-o', label ='Right front')
#plt.plot(time, np.array(left_front), '-o', label ='Left front')
#plt.gca().invert_xaxis()
plt.grid()
plt.xlabel("Time")
plt.ylabel("Flame Front location - X")
plt.legend()
plt.savefig("fig.png")
    




