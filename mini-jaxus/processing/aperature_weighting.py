#aperature_weighting.py

# Aperture weighting: multiply each transducer's analytic signal by its weight before delay-and-sum beamforming.
#hans weighting: assigning transducer elements different weights. Middle elements contribute strongly
#The edge ones contributes less
#used to control how much each transducer contributes. prevents side lobs(unwanted smaller beams beside the main beam)
#equation is wn​=1/2*(1−cos(2πn/127)) where ​n=0,…,127
from analytic_signal import *
import sys
sys.path[0] = sys.path[0].replace("processing","core")
from probe import *
import jax.numpy as jnp
elem = jnp.arange(128)
w = 0.5*(1-jnp.cos(
        (2*jnp.pi*elem)/(array_elem-1)
    )
)
ap_weighted = w[:,None] * val
if __name__ == "__main__":
 print(k.shape)
