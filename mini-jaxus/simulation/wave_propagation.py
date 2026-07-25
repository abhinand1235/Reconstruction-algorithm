import sys
sys.path.append("..")
sys.path.append("../core")
from core.transmit import *
import jax
cell_width = 0.00003 # width is 30 micrometer which is 30*10^-6m 
import jax.numpy as jnp
cell_pos=[]
for i in range(len(tissue_matrix)):
  for j in range(len(tissue_matrix)):
   cell_pos.append((i+0.00003,j+0.00003))
#to determine how close the wave is to the cells use cell_pos - wave_pos
dist_btw=[]
for j in range(len(cell_pos)):
  x,y = cell_pos[j]
  dist_btw.append(x-wave_pos)
print("The distance between cell and wavefront")
print(dist_btw)
cells_near=[]
for i in range(len(dist_btw)):
 for d in dist_btw[i]:
  if d<=0:
   cell_near.append(cell_pos[i])
print(cell near)
