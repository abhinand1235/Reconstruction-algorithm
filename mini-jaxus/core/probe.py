import jax
import jax.numpy as jnp
array_elem = 128
elem_id = jnp.arange(1,array_elem+1) #tranducer element id
pitch = 0.0003
mid_point = (1 + array_elem)/2
elem_pos = (elem_id - mid_point)*pitch
width = 0.00027  # width of transducer < pitch 
center_freq = 4e6 #the frequncy at which tranducer emits sound and recieve it effeciently
aperature = (array_elem - 1)*pitch
if __name__ == "__main__":
 print(elem_id)
 print(elem_pos)
 print("The aperature is: ",aperature) #total width
