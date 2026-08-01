from wave_propagation import *
import jax.numpy as jnp
import sys
sys.path.append("..")
sys.path.append("../core")
from core.transmit import *
#The echo equation is Echo = Reflection Weight× Attenuation
#reflection weight: How much of the wave does this material send back
#Attenuation:How much does the echo lose strength while traveling eq: Attenuation=e^−αd
attentation_coefficient = 0.5 # means for 1 Mhz wave tarvelling 1 cm distance 0.5 decibal of sound energy is lost 
#convert the coefficient in Np/m
# 0.5 dB/cm/MHz × 5 MHz =  2.5 dB/cm as the wave frequency is 5Mhz
# 2.5 db/cm * 100 cm/m = 250 db/m
#1 Np = 8.686 dB, so 250/8.68 = 28.8 Np/m
tissue_weight = 1.0
cyst_weight = 0.5
coe = 28.8
cell_index = cell_index = jnp.round(cell_near / cell_width).astype(jnp.int32)
attenuation = jnp.exp(-coe*dist)
echo = []
for i,j in cell_index:
 if tissue_matrix[i][j] == 1:
  echo.append(tissue_weight * attenuation[i*column+j])
 else:
  echo.append(cyst_weight * attenuation[i*column+j])
echo = jnp.array(echo)
if __name__ == "__main__":
  print("attenuation is: ")
  print(attenuation)
  print(jnp.min(dist))
  print(jnp.max(dist))
  print("The echo generated is: ")
  print(echo) #echo amplitude: how strong the waves that reached the transducer
