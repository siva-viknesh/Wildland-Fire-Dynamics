# ******************************* WILD FIRE SOLVER - 2D UNSTEADY PDE SOLVER   **************************************** #
# Author  : SIVA VIKNESH
# Email   : siva.viknesh@sci.utah.edu / sivaviknesh14@gmail.com
# Address : SCI INSTITUTE, UNIVERSITY OF UTAH, SALT LAKE CITY, UTAH, USA
# ******************************************************************************************************************** #
#import numpy as np
import cupy as np
from scipy.linalg import circulant, toeplitz, inv
import h5py

# ******************************************************************************************************************** #

def PHASE_CHANGE_FUN (u, Upc):                                          # PHASE CHANGE FUNCTION       
	#S = np.zeros_like(u)
	#S[u >= Upc] = 1.0
	S = 1.0/(1.0+np.exp(-50.0*(u-Upc)))
	return S
# -------------------------------------------------------------------------------------------------------------------- #    
	
def DIFFUSION_COEFFICIENT (u, kap, eps):                                # DIFFUSION COEFFICIENT

	return kap * (1.0 + eps * u) ** 3 + 1.0

def DIFFUSION_DERIVATIVE (u, kap, eps):                                 # DIFFUSION COEFFICIENT - DERIVATIVE

	return 3* eps* kap * (1.0 + eps * u) ** 2

# -------------------------------------------------------------------------------------------------------------------- #    

def FUEL_SOURCE (u, ustiff, Upc, beta, eps, alpha):                    # F : TEMPERATURE FUEL-REACTION FUNCTION

	return PHASE_CHANGE_FUN(u, Upc) * beta * np.exp(u / (1.0 + eps * u)) - alpha * ustiff

# ------------------------------------- RHS OF FUEL DECAY PDE EQUATION ----------------------------------------------- #    

def FUEL_MASS_PDE (u, Upc, beta, eps, q):                          # G:  RHS OF MASS FRACTION PDE

	return -1.0* PHASE_CHANGE_FUN(u, Upc) * (eps / q) * beta * np.exp(u / (1.0 + eps * u))

# --------------------------------------- RHS OF ENERGY PDE EQUATION -------------------------------------------------- #  

def RHS_FUNCTION (Dx, Dxx, Dy, Dyy, U, Ustiff, Vx, Vy, Upc, beta, eps, alpha, kap):

	Ux,  Uy     = np.einsum('ij, kj -> ki', Dx,  Vx*U), np.einsum('ij, jk -> ik', Dy,  Vy*U)
	USx, USy    = np.einsum('ij, kj -> ki', Dxx, Ustiff), np.einsum('ij, jk -> ik', Dyy,  Ustiff)
	K           = DIFFUSION_COEFFICIENT (U, kap, eps)
	USxx, USyy  = np.einsum('ij, kj -> ki', Dxx, K*USx), np.einsum('ij, jk -> ik', Dyy, K*USy)

	FUEL        = FUEL_SOURCE (U, Ustiff, Upc, beta, eps, alpha)
	DIFFUSION   = USxx + USyy
	ADVECTION   = Ux + Uy

	RHS_PDE     = DIFFUSION + FUEL - ADVECTION 

	return RHS_PDE

# ------------------------------------- WIND VELOCITY TOPOLOGY CALCULATION ------------------------------------------- #  

def DOUBLE_GYRE_VELOCITY (Nx, Ny, t, w, Vmag, epsilon) :

	X = np.linspace(0.0, 2.0, num=Nx)
	Y = np.linspace(0.0, 1.0, num=Ny)

	x, y = np.meshgrid(X, Y)

	Fxt = epsilon*np.sin(w*t)*(x**2) + (1.0-2.0*epsilon*np.sin(w*t))*x
	dfdx = 2.0*epsilon*np.sin(w*t)*x + (1.0-2.0*epsilon*np.sin(w*t))

	Vx = -Vmag*np.sin(np.pi*Fxt)*np.cos(np.pi*y)
	Vy =  Vmag*np.cos(np.pi*Fxt)*np.sin(np.pi*y)*dfdx
	
	return Vx, Vy

def FREESTREAM_VELOCITY (Nx, Ny, Vmag, alpha):

	x = np.linspace(0.0, 1.0, num=Nx)
	y = np.linspace(0.0, 1.0, num=Ny)

	X, Y = np.meshgrid(x, y)
 
	Vx = Vmag* np.cos(alpha*np.pi/180.0)*np.ones_like(X)
	Vy = Vmag* np.sin(alpha*np.pi/180.0)*np.ones_like(X)
	
	return Vx, Vy


# ---------------------------------- BOUNDARY CONDITION OF TEMPERATURE & FUEL ---------------------------------------- #

def NEUMANN_BOUNDARY_CONDITION (U):                                                 # NEUMANN BOUNDARY CONDITION

	# TEMPERATURE BOUNDARIES
	Utop    = np.copy(U [ 1:3,  :])
	Ubottom = np.copy(U [-3:-1,  :])
	Uleft   = np.copy(U [  :, 1:3])
	Uright  = np.copy(U [  :,-3:-1])

	U [0,  :] = (4.0*Utop[0, :] - Utop[1, :])/3.0                       # TOP BOUNDARY
	U [-1, :] = (Ubottom[0, :] - 4.0*Ubottom[1, :])/3.0                 # BOTTOM BOUNDARY
	U [:,  0] = (4.0*Uleft[:, 0] - Uleft[:, 1])/3.0                     # LEFT BOUNDARY
	U [:, -1] = (Uright[:, 0] - 4.0*Uright[:, 1])/3.0                   # RIGHT BOUNDARY
	U [U < 0.0] = 0.0
	return U
	

def ROBIN_BOUNDARY_CONDITION (U, Vx, Vy, dx, dy, kap, eps):   
	
	Utop     = np.copy(U [ 1:3 ,  :])
	Ubottom  = np.copy(U [-3:-1,  :])
	Uleft    = np.copy(U [ :,   1:3])
	Uright   = np.copy(U [ :, -3:-1])

	tolerance = 1e-8

	#--------------------------------------------- LEFT BOUNDARY ----------------------------------------------------- #

	dT = 1.0
	Un = np.copy (Uleft[:, 0])
	Vn = np.copy (Vx [:, 0])
	
	while dT > tolerance:
		dUdx      = (-3.0*Un + 4.0*Uleft[:, 0]  - Uleft[:, 1])/ (2.0*dx)
		DIFFUSION = DIFFUSION_COEFFICIENT (Un, kap, eps) * dUdx
		ADVECTION = Vn * Un

		G = ADVECTION - DIFFUSION
		J = Vn - dUdx* DIFFUSION_DERIVATIVE (Un, kap, eps) + DIFFUSION_COEFFICIENT (Un, kap, eps)*(1.50/dx)

		ULB = Un - G/J
		dT  = np.mean(np.absolute(ULB - Un))                             # TOLERANCE
		Un  = np.copy(ULB)

	#-------------------------------------------- RIGHT BOUNDARY ----------------------------------------------------- #

	dT = 1.0
	Un = np.copy (Uright[:, -1]) 
	Vn = np.copy (Vx [:, -1])
	
	while dT > tolerance:
		dUdx      = (Uright[:, -2] - 4.0*Uright[:, -1] + 3.0*Un)  / (2.0*dx)
		DIFFUSION = DIFFUSION_COEFFICIENT (Un, kap, eps) * dUdx
		ADVECTION = Vn * Un

		G = ADVECTION - DIFFUSION
		J = Vn - dUdx * DIFFUSION_DERIVATIVE (Un, kap, eps) - DIFFUSION_COEFFICIENT (Un, kap, eps)*(1.50/dx)

		URB = Un - G/J
		dT  = np.mean(np.absolute(URB - Un))                             # TOLERANCE
		Un  = np.copy(URB)

	#--------------------------------------------- TOP BOUNDARY ----------------------------------------------------- #

	dT = 1.0
	Un = np.copy (Utop[0, :])
	Vn = np.copy (Vy  [0, :])
	
	while dT > tolerance:
		dUdy      = (-3.0*Un + 4.0*Utop[0, :] - Utop[1, :]) / (2.0*dy) 
		DIFFUSION = DIFFUSION_COEFFICIENT (Un, kap, eps) * dUdy
		ADVECTION = Vn * Un

		G = ADVECTION - DIFFUSION
		J = Vn - dUdy* DIFFUSION_DERIVATIVE (Un, kap, eps) + DIFFUSION_COEFFICIENT (Un, kap, eps)*(1.50/dy)

		UTB = Un - G/J
		dT  = np.mean(np.absolute(UTB - Un))                             # TOLERANCE
		Un  = np.copy(UTB)

	#------------------------------------------- BOTTOM BOUNDARY ----------------------------------------------------- #

	dT = 1.0
	Un = np.copy (Ubottom [-1, :])
	Vn = np.copy (Vy  [-1, :])
	
	while dT > tolerance:
		dUdy      = (Ubottom[-2, :]  - 4.0*Ubottom[-1, :] + 3.0*Un) / (2.0*dy)
		DIFFUSION = DIFFUSION_COEFFICIENT (Un, kap, eps) * dUdy
		ADVECTION = Vn * Un

		G = ADVECTION - DIFFUSION
		J = Vn - dUdy* DIFFUSION_DERIVATIVE (Un, kap, eps) - DIFFUSION_COEFFICIENT (Un, kap, eps)*(1.50/dy)

		UBB = Un - G/J
		dT  = np.mean(np.absolute(UBB - Un))                             # TOLERANCE
		Un  = np.copy(UBB)

	U [0,  :] = UTB                                                 # TOP BOUNDARY
	U [-1, :] = UBB                                                 # BOTTOM BOUNDARY
	U [:,  0] = ULB                                                 # LEFT BOUNDARY
	U [:, -1] = URB                                                 # RIGHT BOUNDARY

	return U

# -------------------------------- INITIAL DISTRIBUTION OF VEGETATION OVER SPACE ------------------------------------- #  

def BETA_DISTRIBUTION (Nx, Ny, mean, std):

	size = (Ny, Nx)

	#dis = np.random.normal(loc = mean, scale = std, size = size)
	#dis = np.random.uniform (low = mean-std, high = mean, size = size)
	dis = np.ones((size))

	return dis*0.60

# ----------------------------------- INITIAL LOCATION OF FLAME SOURCE ----------------------------------------------- #  

def PATCH_FLAME_LOCATION (Nx, Ny, Xc, Yc, xmin, xmax, ymin, ymax, UFlame, R):

	x = np.linspace(xmin, xmax, num=Nx)
	y = np.linspace(ymin, ymax, num=Ny)
	X, Y = np.meshgrid(x, y)
	
	U = np.zeros_like(X)
	U[((Xc-R) <= X) & (X <= (Xc+R)) & ((Yc-R) <= Y) & (Y <= (Yc+R))] = UFlame

	return U

# ----------------------------------- WRITING THE TIME SERIES DATA --------------------------------------------------- #  

def WRITE_DATA_H5PY (N, beta, Vx, Vy, Temperature):

	file_name = "Data_" + str(N) + ".h5"

	with h5py.File(file_name, 'w') as hf:
		hf.create_dataset('Fuel', data = np.asnumpy(beta))
		hf.create_dataset('Vx',   data = np.asnumpy(Vx))
		hf.create_dataset('Vy',   data = np.asnumpy(Vy))
		hf.create_dataset('Temperature', data = np.asnumpy(Temperature))
		hf.close()

# -------------------------------------------------------------------------------------------------------------------- #  

# ******************************************************************************************************************** #

# ********************************************* NUMERICAL SCHEMES **************************************************** #

# ******************************************************************************************************************** #

# ------------------------------------ FIRST DERIVATIVE CENTRAL DIFFERENCE SCHEME ------------------------------------ #  

def CD2 (N, h):
	a11, a12, a13 =  1.0, 0.0, -1.0 
	d = np.zeros(Nx)
	d[-1], d[0], d[1] = a11, a12, a13
	D = circulant (np.asnumpy(d))/(2.0*h)

	D [0,  :]  = 0.0
	D [0,   :3] = np.asnumpy([-3.0, 4.0, -1.0])/(2.0*h)

	D [-1,  :]  = 0.0
	D [-1, -3:] = np.asnumpy([1.0, -4.0, 3.0])/(2.0*h)
	return D

# ---------------------------------------------- UPWIND COMPACT SCHEME  ---------------------------------------------- #

def OUCS2 (N, h):
	d = np.zeros(N)
	
	a  = -40.0

	p33 = 36.0
	p32 = p33/3.0 - a/12.0
	p34 = p33/3.0 + a/12.0
	
	d[1]  = p32
	d[0]  = p33
	d[-1] = p34

	D1 = circulant(np.asnumpy(d))

	d     = np.zeros(N)

	q31 = -p33/36.0 + a/72.0
	q32 = -7.0*p33/9.0 + a/9.0
	q33 = -a/4.0
	q34 = +7.0*p33/9.0 + a/9.0
	q35 = +p33/36.0 + a/72.0

	beta1 = 0.020
	beta2 = 0.090 

	d[-2]  = q35
	d[-1]  = q34
	d[0]   = q33
	d[1]   = q32
	d[2]   = q31

	D2 = circulant(np.asnumpy(d))/h

	out = inv(D1) @ D2

	out [0:2,  :] = 0.0
	out [0,   :3] = np.asnumpy([-3.0, 4.0, -1.0])/(2.0*h)
	out [1,   :5] = np.asnumpy([((2.0*beta1)/3.0-1.0/3.0), -((8.0*beta1)/3.0 + 0.50), 4.0*beta1 +1.0, -(8.0*beta1/3.0 + 1.0/6.0), 2.0*beta1/3.0])

	out [-2:,  :]  = 0.0
	out [-2, -5:] = np.asnumpy([(-2.0*beta2/3.0,  (8.0*beta2/3.0 + 1.0/6.0), -(4.0*beta2 +1.0), ((8.0*beta2)/3.0 + 0.50),  -((2.0*beta2)/3.0-1.0/3.0))])
	out [-1, -3:] = np.asnumpy([1.0, -4.0, 3.0])/(2.0*h)

	return out 

def OUCS3 (N, h):
	d = np.zeros(N)
	D = 0.3793894912
	E = 0.183205192
	F = 1.57557379
	G = -2.0

	p32 = D - G / 60.0
	p33 = 1.0
	p34 = D + G / 60.0

	d[1]  = p32
	d[0]  = p33
	d[-1] = p34

	D1 = circulant(np.asnumpy(d))

	d     = np.zeros(N)

	q31 = -F/4.0  +  G/300.0
	q32 = -E/2.0  +  G/30.0
	q33 = -11.0   *  G/150.0
	q34 =  E/2.0  +  G/30.0
	q35 =  F/4.0  +  G/300.0

	beta1 = -0.025
	beta2 = 0.090

	d[-2]  = q35
	d[-1]  = q34
	d[0]   = q33
	d[1]   = q32
	d[2]   = q31

	D2 = circulant(np.asnumpy(d))/h

	out = inv(D1) @ D2

	out [0:2,  :] = 0.0
	out [0,   :3] = np.asnumpy([-3.0, 4.0, -1.0])/(2.0*h)
	out [1,   :5] = np.asnumpy([((2.0*beta1)/3.0-1.0/3.0), -((8.0*beta1)/3.0 + 0.50), 4.0*beta1 +1.0, -(8.0*beta1/3.0 + 1.0/6.0), 2.0*beta1/3.0])/h

	out [-2:,  :]  = 0.0
	out [-2, -5:] = np.asnumpy([(-2.0*beta2/3.0,  (8.0*beta2/3.0 + 1.0/6.0), -(4.0*beta2 +1.0), ((8.0*beta2)/3.0 + 0.50),  -((2.0*beta2)/3.0-1.0/3.0))])/h
	out [-1, -3:] = np.asnumpy([1.0, -4.0, 3.0])/(2.0*h)

	return out

# --------------------------------------------- COMBINED COMPACT SCHEME  --------------------------------------------- # 

def NCCD (N, h):
	f1 = 7.0  / 16.0
	f2 = -h / 16.0
	f3 = 15.0 / (16.0*h)
	f4 = 9.0  / (8.0*h)
	f5 = -1.0 / 8.0
	f6 = 3.0 / (h**2)
	
	# Declaration of Matrix A1
	p31, p32, p33, p34, p35 = 0, f1, 1.0, f1, 0
	A1 = toeplitz([p33, p32] + [0] * (N - 2), [p33, p34] + [0] * (N - 2))
	A1[0, N-1] = p32
	A1[N-1, 0] = p34
	A1 = np.array(A1)

	# Declaration of Matrix B1
	p31, p32, p33, p34, p35 = 0, -f2, 0, f2, 0
	B1 = toeplitz([p33, p32] + [0] * (N - 2), [p33, p34] + [0] * (N - 2))
	B1[0, N-1] = p32
	B1[N-1, 0] = p34
	B1 = np.array(B1)

	# Declaration of Matrix C1
	q31, q32, q33, q34, q35 = 0, -f3, 0, f3, 0
	C1 = toeplitz([q33, q32] + [0] * (N - 2), [q33, q34] + [0] * (N - 2))
	C1[0, N-1] = q32
	C1[N-1, 0] = q34
	C1 = np.array(C1)

	# Declaration of Matrix A2
	p31, p32, p33, p34, p35 = 0, -f4, 0, f4, 0
	A2 = toeplitz([p33, p32] + [0] * (N - 2), [p33, p34] + [0] * (N - 2))
	A2[0, N-1] = p32
	A2[N-1, 0] = p34
	A2 = np.array(A2)

	# Declaration of Matrix B2
	p31, p32, p33, p34, p35 = 0, f5, 1.0, f5, 0
	B2 = toeplitz([p33, p32] + [0] * (N - 2), [p33, p34] + [0] * (N - 2))
	B2[0, N-1] = p32
	B2[N-1, 0] = p34
	B2 = np.array(B2)

	# Declaration of Matrix C2
	q31, q32, q33, q34, q35 = 0, f6, -2*f6, f6, 0
	C2 = toeplitz([q33, q32] + [0] * (N - 2), [q33, q34] + [0] * (N - 2))
	C2[0, N-1] = q32
	C2[N-1, 0] = q34
	C2 = np.array(C2)

	# Declaration of Matrix D1 & D2
	D1 = np.linalg.inv(A1 - B1 @ np.linalg.inv(B2) @ A2) @ (C1 - B1 @ np.linalg.inv(B2) @ C2)
	D2 = np.linalg.inv(B2 - A2 @ np.linalg.inv(A1) @ B1) @ (C2 - A2 @ np.linalg.inv(A1) @ C1)

	# EXPLICIT BOUNDARY CLOSURES
	beta1 = -0.025
	beta2 = 0.090

	D1 [0:2,  :] = 0.0
	D1 [0,   :3] = np.array([-3.0, 4.0, -1.0])/(2.0*h)
	D1 [1,   :5] = np.array([((2.0*beta1)/3.0-1.0/3.0), -((8.0*beta1)/3.0 + 0.50), 4.0*beta1 +1.0, -(8.0*beta1/3.0 + 1.0/6.0), 2.0*beta1/3.0]) / h
	
	D1 [-2:, :]  = 0.0
	D1 [-2, -5:] = np.array([(-2.0*beta2/3.0,  (8.0*beta2/3.0 + 1.0/6.0), -(4.0*beta2 +1.0), ((8.0*beta2)/3.0 + 0.50),  -((2.0*beta2)/3.0-1.0/3.0))])/h
	D1 [-1, -3:] = np.array([1.0, -4.0, 3.0])/(2.0*h)
	
	D2 [:2,  :]  = 0.0
	D2 [1,   :3] = np.array([1.0, -2.0, 1.0])/ (h**2)
	#D2 [0,   :3] = np.array([1.0, -2.0, 1.0])/ (h**2)
	D2 [0,   :4] = np.array([2.0, -5.0, 4.0, -1.0])/ (h**3)

	D2 [-2:,  :]  = 0.0
	D2 [-2, -3:]  = np.array([1.0, -2.0, 1.0])/ (h**2)
	#D2 [-1, -3:]  = np.array([1.0, -2.0, 1.0])/ (h**2)
	D2 [-1, -4:]  = np.array([-1.0, 4.0, -5.0, 2.0])/ (h**3)

	return D1, D2

# ----------------------------------------- EXPLICIT RK-4 TIME INTEGRATION   ----------------------------------------- # 

def RK4 (dt, Dx, Dy, Vx, Vy, Upc, beta, eps, alpha, kap, U):     # EXPLICIT RK4 TIME INTEGRATION

	a21 = 0.50
	a32 = 0.50
	a43 = 1.00

	W1  = 1.0/6.0
	W2  = 1.0/3.0
	W3  = 1.0/3.0
	W4  = 1.0/6.0

	# TEMPERATURE TIME INTEGRATION
	Un  = np.copy(U)

	K1  = RHS_FUNCTION(Dx, Dy, Un,                Vx, Vy, Upc, beta, eps, alpha, kap) 
	K2  = RHS_FUNCTION(Dx, Dy, Un + a21 * dt* K1, Vx, Vy, Upc, beta, eps, alpha, kap) 
	K3  = RHS_FUNCTION(Dx, Dy, Un + a32 * dt* K2, Vx, Vy, Upc, beta, eps, alpha, kap) 
	K4  = RHS_FUNCTION(Dx, Dy, Un + a43 * dt* K3, Vx, Vy, Upc, beta, eps, alpha, kap) 

	Unew = Un + dt * (W1*K1 + W2*K2 + W3*K3 + W4*K4)

	# FUEL TIME INTEGRATION
	bn = np.copy(beta)

	B1 = FUEL_MASS_FRACTION (Un, Upc, bn,                eps, Q)
	B2 = FUEL_MASS_FRACTION (Un, Upc, bn + a21 * dt* B1, eps, Q)
	B3 = FUEL_MASS_FRACTION (Un, Upc, bn + a32 * dt* B2, eps, Q)
	B4 = FUEL_MASS_FRACTION (Un, Upc, bn + a43 * dt* B3, eps, Q)

	bnew = bn + dt * (W1*B1 + W2*B2 + W3*B3 + W4*B4)

	return Unew, bnew

# ------------------------------- IMPLICIT-EXPLICIT (IMEX) RK-4 TIME INTEGRATION ------------------------------------- # 

def NEWTON_METHOD (dt, abc, Vx, Vy, Dx, Dxx, Dy, Dyy, eps, kap, alpha, Upc, U, Ustiff, beta):

	tolerance = 1e-7    
	dy = 1.0
	yn = RHS_FUNCTION(Dx, Dxx, Dy, Dyy, U, Ustiff, Vx, Vy, Upc, beta, eps, alpha, kap) # INITIAL VALUES
	J  = np.ones_like(U)*(1.0 + alpha*dt*abc)                                          # JACOBIAN
	
	while dy > tolerance:
		K  = RHS_FUNCTION(Dx, Dxx, Dy, Dyy, U, Ustiff + dt*abc*yn, Vx, Vy, Upc, beta, eps, alpha, kap)
		yN = yn - (yn - K)/J
		dy = np.mean(np.absolute(yN - yn))                                              # TOLERANCE
		yn = np.copy(yN)

	return yN


def IMEX_RK4 (U, dt, Dx, Dxx, Dy, Dyy, Vx, Vy, Upc, beta, eps, alpha, kap, q):

	a  = 0.24169426078821
	b  = a/4.0
	n  = 0.12915286960590

	a11                =  a
	a21, a22           = -a, a
	a32, a33           =  1.0 - a, a
	a41, a42, a43, a44 =  b, n, 0.5-b-n-a, a

	A32       =  1.0
	A42, A43  =  0.25, 0.25

	W1  = 0.0
	W2  = 1.0/6.0
	W3  = 1.0/6.0
	W4  = 2.0/3.0

	Un = np.copy(U)
	Bn = np.copy(beta)

	#------------------------------------------------- STAGE I --------------------------------------------------------- #
	# STAGE VARIABLES
	U1, US1, B1  = Un, Un, Bn

	# STAGE VALUES
	KB1 = FUEL_MASS_PDE (U1, Upc, B1, eps, q)/ (1.0 + (eps/q)*dt*a11*PHASE_CHANGE_FUN (U1, Upc))
	KU1 = NEWTON_METHOD (dt, a11, Vx, Vy, Dx, Dxx, Dy, Dyy, eps, kap, alpha, Upc, U1, US1, B1+dt*a11*np.copy(KB1))

	#------------------------------------------------ STAGE II --------------------------------------------------------- #
	# STAGE VARIABLES
	U2, US2, B2  = Un, Un + dt*a21*KU1, Bn + dt*a21*KB1

	# STAGE VALUES
	KB2 = FUEL_MASS_PDE (U2, Upc, B2, eps, q)/ (1.0 + (eps/q)*dt*a22*PHASE_CHANGE_FUN (U2, Upc))
	KU2 = NEWTON_METHOD (dt, a22, Vx, Vy, Dx, Dxx, Dy, Dyy, eps, kap, alpha, Upc, U2, US2, B2+dt*a22*np.copy(KB2))

	#------------------------------------------------ STAGE III -------------------------------------------------------- #
	# STAGE VARIABLES
	U3, US3, B3  = Un + dt*A32*KU2, Un + dt*a32*KU2, Bn + dt*a32*KB2

	# STAGE VALUES
	KB3 = FUEL_MASS_PDE (U3, Upc, B3, eps, q)/ (1.0 + (eps/q)*dt*a33*PHASE_CHANGE_FUN (U3, Upc))
	KU3 = NEWTON_METHOD (dt, a33, Vx, Vy, Dx, Dxx, Dy, Dyy, eps, kap, alpha, Upc, U3, US3, B3+dt*a33*np.copy(KB3))

	#------------------------------------------------ STAGE IV --------------------------------------------------------- #
	# STAGE VARIABLES
	U4, US4, B4  = Un + dt*(A42*KU2 + A43*KU3), Un + dt*(a41*KU1 + a42*KU2 + a43*KU3), Bn + dt*(a41*KB1 + a42*KB2 + a43*KB3)

	# STAGE VALUES
	KB4 = FUEL_MASS_PDE (U4, Upc, B4, eps, q)/ (1.0 + (eps/q)*dt*a44*PHASE_CHANGE_FUN (U4, Upc))
	KU4 = NEWTON_METHOD (dt, a44, Vx, Vy, Dx, Dxx, Dy, Dyy, eps, kap, alpha, Upc, U4, US4, B4+dt*a44*np.copy(KB4))

	#-------------------------------------------- MARCHING OVER TIME --------------------------------------------------- #

	Unew = Un + dt * (W1*KU1 + W2*KU2 + W3*KU3 + W4*KU4)
	Bnew = Bn + dt * (W1*KB1 + W2*KB2 + W3*KB3 + W4*KB4)

	return Unew, Bnew

def IMEX_RK3 (U, dt, Dx, Dxx, Dy, Dyy, Vx, Vy, Upc, beta, eps, alpha, kap, q):
	a = 1.0 - 1.0/np.sqrt(2)

	a11            =  a
	a21, a22       =  1.0 - 2.0*a, a
	a31, a32, a33  = 0.50-a, 0.0 , a
	
	A21       =  1.0
	A31, A32  =  0.25, 0.25

	W1  = 1.0/6.0
	W2  = 1.0/6.0
	W3  = 2.0/3.0

	Un = np.copy(U)
	Bn = np.copy(beta)

	#------------------------------------------------- STAGE I --------------------------------------------------------- #
	# STAGE VARIABLES
	U1, US1, B1  = Un, Un, Bn

	# STAGE VALUES
	KB1 = FUEL_MASS_PDE (U1, Upc, B1, eps, q)/ (1.0 + (eps/q)*dt*a11*PHASE_CHANGE_FUN (U1, Upc))
	KU1 = NEWTON_METHOD (dt, a11, Vx, Vy, Dx, Dxx, Dy, Dyy, eps, kap, alpha, Upc, U1, US1, B1+dt*a11*np.copy(KB1))

	#------------------------------------------------ STAGE II --------------------------------------------------------- #
	# STAGE VARIABLES
	U2, US2, B2  = Un + dt*A21*KU1, Un + dt*a21*KU1, Bn + dt*a21*KB1

	# STAGE VALUES
	KB2 = FUEL_MASS_PDE (U2, Upc, B2, eps, q)/ (1.0 + (eps/q)*dt*a22*PHASE_CHANGE_FUN (U2, Upc))
	KU2 = NEWTON_METHOD (dt, a11, Vx, Vy, Dx, Dxx, Dy, Dyy, eps, kap, alpha, Upc, U2, US2, B2+dt*a22*np.copy(KB2))

	#------------------------------------------------ STAGE III -------------------------------------------------------- #
	# STAGE VARIABLES
	U3, US3, B3  = Un + dt*(A31*KU1+A32*KU2), Un + dt*(a31*KU1+a32*KU2), Bn + dt*(a31*KB1+a32*KB2)

	# STAGE VALUES
	KB3 = FUEL_MASS_PDE (U3, Upc, B3, eps, q)/ (1.0 + (eps/q)*dt*a33*PHASE_CHANGE_FUN (U3, Upc))
	KU3 = NEWTON_METHOD (dt, a11, Vx, Vy, Dx, Dxx, Dy, Dyy, eps, kap, alpha, Upc, U3, US3, B3+dt*a33*np.copy(KB3))

	#-------------------------------------------- MARCHING OVER TIME --------------------------------------------------- #

	Unew = Un + dt * (W1*KU1 + W2*KU2 + W3*KU3)
	Bnew = Bn + dt * (W1*KB1 + W2*KB2 + W3*KB3)

	return Unew, Bnew


def IMEX_RK2 (U, dt, Dx, Dxx, Dy, Dyy, Vx, Vy, Upc, beta, eps, alpha, kap, q):

	g         =  (3.0 + np.sqrt(3))/6.0
	a11       =  g
	a21, a22  =  1.0 - 2.0*g, g
	
	A21       =  1.0

	W1  = 0.50
	W2  = 0.50

	Un = np.copy(U)
	Bn = np.copy(beta)

	#------------------------------------------------- STAGE I --------------------------------------------------------- #
	# STAGE VARIABLES
	U1, US1, B1  = Un, Un, Bn

	# STAGE VALUES
	KB1 = FUEL_MASS_PDE (U1, Upc, B1, eps, q)/ (1.0 + (eps/q)*dt*a11*PHASE_CHANGE_FUN (U1, Upc))
	KU1 = NEWTON_METHOD (dt, a11, Vx, Vy, Dx, Dxx, Dy, Dyy, eps, kap, alpha, Upc, U1, US1, B1+dt*a11*np.copy(KB1))

	#------------------------------------------------ STAGE II --------------------------------------------------------- #
	# STAGE VARIABLES
	U2, US2, B2  = Un+ dt*A21*KU1, Un + dt*a21*KU1, Bn + dt*a21*KB1

	# STAGE VALUES
	KB2 = FUEL_MASS_PDE (U2, Upc, B2, eps, q)/ (1.0 + (eps/q)*dt*a22*PHASE_CHANGE_FUN (U2, Upc))
	KU2 = NEWTON_METHOD (dt, a22, Vx, Vy, Dx, Dxx, Dy, Dyy, eps, kap, alpha, Upc, U2, US2, B2+dt*a22*np.copy(KB2))

	#-------------------------------------------- MARCHING OVER TIME --------------------------------------------------- #

	Unew = Un + dt * (W1*KU1 + W2*KU2)
	Bnew = Bn + dt * (W1*KB1 + W2*KB2)

	return Unew, Bnew

# ******************************************************************************************************************** #

# *************************************** MAIN PROGRAM STARTS HERE *************************************************** #

# ******************************************************************************************************************** #

""" ********************************* GEOMETRIC PARAMETERS OF THE DOMAIN ******************************************* """

print ("*"*85)
print ("\nWILD FIRE SIMULATION HAS BEEN STARTED......... \n")

xmin = 0.0
xmax = 50.0
Nx   = 256
dx   = (xmax - xmin) / Nx

ymin = 0.0
ymax = 50.0                                          
Ny   = 256
dy   = (ymax - ymin) / Ny

dt   = 1e-7                                                # TIME STEP
Nt   = int(2.0/dt)                                        # NO. OF TIME STEPS
NIT  = int(1/(1e2*dt))                                     # TIME INTERVAL FOR SVAING THE DATA

""" ****************************************** RESTART DATA FILES ************************************************** """

Restart  = False
NRestart = 500000

""" ***************************************** PDE SOLVER PARAMETERS ************************************************ """

# NON DIMENSIONAL PARAMTERS OF THE PDE
eps    = 3e-2                                                # INVERSE OF ACTIVATION ENERGY
Q      = 1.00                                                # NON-DIMENSIONAL REACTION HEAT
alpha  = 1e-3                                                # NON-DIMENSIONAL NATURAL CONVECTION COEFFICENT
kap    = 1e-1                                                # DIFFUSION PARAMETER
Upc    = 3.0                                                 # NON-DIMENSIONAL PHASE CHANGE TEMPERATURE
UFlame = 31.0                                                # NON-DIMENSIONAL TEMPERATURE AT FLAME SOURCE

""" ************************************ FUEL DISTRIBUTION PARAMETERS ********************************************** """

mean = 1.00                                                  # MEAN - FUEL VEGETATION
std  = 0.05                                                  # STANDARD DEVIATION (LOW) - FUEL VEGETATION

""" ***************************************** INITIAL CONDITION  *************************************************** """
if (Restart):
	file_name = "Data_" + str(NRestart) + ".h5"

	print ("*"*85)
	print ('\nREADING THE DATA FILE: ', file_name)
	print ('\n')

	with h5py.File(file_name, 'r') as hf:
		beta = np.array(hf['Fuel'][()])
		Vx   = np.array(hf['Vx'][()])
		Vy   = np.array(hf['Vy'][()])
		U    = np.array(hf['Temperature'][()])

else:
	NRestart = 0
	beta = BETA_DISTRIBUTION (Nx, Ny, mean, std)                # FUEL VEGETATION DISTRIBUTION

	# INITIAL FLAME LOCATION

	xo, yo  = 0.20, 0.20
	s       = 0.02
	
	#U = GAUSSIAN_FLAME_LOCATION (Nx, Ny, xo, yo, s, UFlame)

	Xc, Yc = 8.0, 8.0
	R      = 4.0

	U    = PATCH_FLAME_LOCATION (Nx, Ny, Xc, Yc, xmin, xmax, ymin, ymax, UFlame, R)

	# WIND TOPOLOGY
	Vmag    = 425.0                                           # VELOCITY MAGNITUDE
	AoA     = 45.0
	Vx, Vy = FREESTREAM_VELOCITY (Nx, Ny, Vmag, AoA)             # DOUBLE GYRE FLOW

print ("*"*85)
print ("\nVEGETATION DISTRIBUTION & WIND VELOCITY HAS BEEN INITIALISED ......... \n")


""" ***************************************** ZERO DIFFUSION MODEL ************************************************* """

Umax = (Q/eps) *(np.max(BETA_DISTRIBUTION (Nx, Ny, mean, std)) + (eps/Q)*UFlame)

""" ************************************ FINITE DIFFERENCE OPERATORS *********************************************** """

Dx  = OUCS2 (Nx, dx)
Dy  = OUCS2 (Ny, dy)

Dxx = CD2 (Nx, dx)
Dyy = CD2 (Ny, dy)

""" *************************************** COMPUTATION OVER TIME ************************************************** """

for i in range (Nt - NRestart):

	#U = ROBIN_BOUNDARY_CONDITION (U, Vx, Vy, dx, dy, kap, eps)
	U = NEUMANN_BOUNDARY_CONDITION (U)	
	print ("*"*65)
	print ("ITERATION = ", i + NRestart)
	print ("MAX TEMPERATURE   = ", np.max (U))
	print ("POSSIBLE MAX TEMP = ", Umax)
	print ("AVERAGE FUEL      = ", np.mean (beta))

	# WRITE THE TMPERATURE AND BETA DATA
	if i % NIT == 0:
		WRITE_DATA_H5PY (i + NRestart, beta, Vx, Vy, U)
		print ("*"*65)
		print ("\n FIELD DATA HAS BEEN STORED ......... \n")
		#dt = ADAPTIVE_TIME_STEPPING (dt, Umax, U, Dx, Dy, Vx, Vy, Upc, beta, eps, alpha, kap, Q)
	
	# COMPUTE THE UNSTEADY WIND VELOCITY FIELD 
	#Vx, Vy  = VELOCITY (i*dt, X/Xn, Y/Yn, w, Vmag, eps)

	# MARCH THE TEMPERATURE AND BETA OVER TIME
	U, beta  = IMEX_RK3 (U, dt,  Dx, Dxx, Dy, Dyy, Vx, Vy, Upc, beta, eps, alpha, kap, Q)


# ******************************************************************************************************************** #

# **************************************** MAIN PROGRAM ENDS HERE **************************************************** #

# ******************************************************************************************************************** #
