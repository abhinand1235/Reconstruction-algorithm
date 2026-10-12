import jax.numpy as jnp
row = jnp.arange(0,0.00003*40,0.00003)
column = jnp.arange(0,0.00003*40,0.00003)
R1,C1 = jnp.meshgrid(row,column)
if __name__ == "__main__":
 print(R1.shape)
 print(C1.shape)
