from wave_propagation import *
import jax.numpy as jnp
#The echo equation is Echo= Reflection Weight× Attenuation
#reflection weight: How much of the wave does this material send back
#Attenuation: How much does the echo lose strength while traveling?
reflectw_tissue = 1.0
relectw_cyst = 0.05
