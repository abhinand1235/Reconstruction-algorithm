# Mini-JAXus

A mini ultrasound image reconstruction algorithm implemented using Python and JAX.

This project is a simplified implementation of an ultrasound simulation and reconstruction pipeline. The goal is to understand and build the core steps involved in ultrasound image formation, from wave propagation through tissue to future image reconstruction using delay-based methods.

---

# Project Overview

Ultrasound imaging works by transmitting acoustic waves into tissue and measuring the returning echoes. The received signals contain information about tissue structures, and reconstruction algorithms use this information to generate an image.

Mini-JAXus currently focuses on implementing the early stages of this pipeline:

1. Creating a simulated tissue environment.
2. Representing tissue as individual spatial cells.
3. Simulating wave interaction with tissue.
4. Identifying cells reached by the propagating wave.
5. Calculating travel time between tissue cells and transducer elements.

The project uses JAX to convert numerical operations into array-based computations suitable for GPU acceleration.

---

# Current Implementation

## 1. Tissue Model

A simulated tissue region has been created.

The tissue is represented as a grid of small cells.

Each cell contains a spatial coordinate:


(x, y)


where:

- `x` represents lateral position.
- `y` represents depth.

The cell spacing is defined using:


cell_width = 30 micrometers


The tissue grid is converted into coordinate positions:


cell_pos


Example representation:


[
[x1, y1],
[x2, y2],
[x3, y3],
...
]


These coordinates are later used for distance and propagation calculations.

---

# 2. Wave-Tissue Interaction

The simulation determines which tissue cells interact with the propagating wave.

The wave position is compared with the depth of each tissue cell.

Distance calculation:


distance = cell_depth - wave_position


A cell is considered affected when:


distance <= 0


The cells satisfying this condition are stored as:


cell_near


These are the tissue cells that have been reached by the wave.

---

# 3. JAX Conversion

The initial implementation used Python loops.

The computation was converted into JAX operations.

Implemented JAX operations:

- JAX arrays for storing tissue coordinates.
- Vectorized distance calculations.
- Boolean masking for selecting cells.
- Matrix-based time delay calculation.

This allows the simulation to scale better for larger tissue sizes and enables GPU execution through JAX.

---

# 4. Cell Selection Using Wave Position

The distance between every cell and wave position is calculated.

The resulting distance matrix represents:


cell × wave_position


Each cell is checked to determine whether the wave has reached it.

The selected cells are stored in:


cell_near


which contains the coordinates of interacting tissue cells.

---

# 5. Time Delay Calculation

After finding the interacting cells, the travel distance between each cell and transducer element is calculated.

The Euclidean distance formula is used:

\[
d = \sqrt{(x_2-x_1)^2+(y_2-y_1)^2}
\]

The speed of sound in tissue is:


1540 m/s


Time delay:


time_delay = distance / speed_of_sound


The output is a delay matrix:


(number_of_cells, number_of_transducer_elements)


Each value represents:


time required for the wave to travel between a tissue cell and a transducer element


---

# Current Pipeline

The implemented pipeline:


Create Tissue Grid
|
v
Generate Cell Coordinates
|
v
Simulate Wave Position
|
v
Find Cells Reached By Wave
|
v
Calculate Cell-Transducer Distance
|
v
Calculate Time Delay
