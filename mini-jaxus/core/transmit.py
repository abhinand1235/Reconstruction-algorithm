from probe import *
from medium import *
import jax.numpy as jnp
jnp.set_printoptions(threshold=float('inf'))
jnp.set_printoptions(
    threshold=float('inf'),
    linewidth=200
)
import jax
import matplotlib.pyplot as plt
#u(r,t)=Acos(kr−ωt+ϕ): plane wave equation in 2d
A = 1000000 #1 MPa
angular_freq = center_freq * 2 * 3.14 # 2*pi*f
phi = 0 # wave strating at maximum aplitude: intial phase 
time_onecycle = 1/center_freq
t = jnp.arange(0,5*time_onecycle,time_onecycle/20)  # here one clock cycle time is time_onecycle
k = (2 * 3.14)/wavelength # (2 * pi)/lambda
r = jnp.arange(0,0.04,1e-5) #wave travels 4 cm stsring from 0 
R,T = jnp.meshgrid(r,t)
plane_wave = A * jnp.cos((k*R)-(angular_freq*T)+phi)

#Create a square region of ovarian tissue(epithilum) assume the dimension to be 40 * 40
tissue_matrix = jnp.ones((40,40))
M,N = (40-1)/2,(40-1)/2
radius = 8

#the cyst shape equation (x-M)^2 + (y-N)^2 <=R^2
for i in range(len(tissue_matrix)):
    for j in range(len(tissue_matrix)):
        dis = (i-M)**2 + (j-N)**2
        if(dis<=radius**2):
            tissue_matrix=tissue_matrix.at[i,j].set(0)

end_t = 0.04/speed #time at which the waves reaches the 4cm long tissue
wave_t = jnp.arange(0,end_t,time_onecycle/20)
wave_pos = speed * wave_t

# --- ONLY EXECUTES WHEN RUNNING transmit.py DIRECTLY ---
if __name__ == "__main__":
    plt.figure(figsize=(12,4))
    plt.plot(
        r,
        plane_wave[len(t)//2],
        linewidth=3,
        antialiased=True,
        solid_capstyle="round",
        solid_joinstyle="round"
    )
    plt.xlim(0, 0.004)       # Zoom only
    plt.ylim(-A, A)
    plt.grid(True)
    plt.show(block=False)
    plt.pause(2)
    plt.close()

    print("The tissue matrix without cyst: ")
    print(jnp.ones((40,40)))
    print(M,N)
    print("Updated")
    print(tissue_matrix)
    print("The time from transducers to tissue cyst with each clock cycle 20 samples")
    print(wave_t)
    print("The wave pos from transducers to tissue cyst with each clock cycle 20 samples")
    print(wave_pos)
