import sys
sys.path.append("..")
sys.path.append("../core")
from core.transmit import *
from core.probe import *
import jax 
import jax.numpy as jnp
cell_width = 0.00003 # width is 30 micrometer which is 30*10^-6m
cell=[]
for i in range(len(tissue_matrix)):
  for j in range(len(tissue_matrix)):
   cell.append((i+cell_width,j+cell_width))
cell_pos = jnp.array(cell)
#x->latral distance y->depth
x = cell_pos[:, 0]
y = cell_pos[:, 1]
distance_between = y[:, None] - wave_pos[None, :] #to determine how close the wave is to the cells use cell_pos - wave_pos
val = jnp.any(distance_between <= 0, axis=1)
cell_near = cell_pos[val] #cells that interacted with wave
#find the distance form tissue to tranducers
x = cell_near[:,0]
y = cell_near[:,1]
time_delay = jnp.sqrt((x[:,None]-0)**2 + (y[:,None]-elem_pos[None,:])**2)/1540
if __name__ == "__main__":
  print("The cell poistion is")
  print(cell_pos)
  print("The cells near")
  print(cell_near)
  print("The time delay is: ")
  print(time_delay)
