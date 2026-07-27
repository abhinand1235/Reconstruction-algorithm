# Code Flow (Read in This Order)

## 1. `core/probe.py`

**Input:**

* Number of transducer elements
* Element spacing (pitch)
* Center frequency

**What it does:**

* Creates the ultrasound probe geometry.
* Computes the position of every transducer element.

**Output:**

* `elem_pos`
* `center_freq`
* Other probe parameters used by later files.

---

## 2. `core/medium.py`

**Input:**

* Probe parameters from `probe.py`

**What it does:**

* Defines the acoustic properties of the imaging medium.

**Output:**

* Speed of sound
* Density
* Acoustic impedance
* Wavelength

These values are used when simulating wave propagation.

---

## 3. `core/transmit.py`

**Input:**

* Probe geometry
* Medium properties

**What it does:**

* Generates a plane ultrasound wave.
* Creates a simple 40×40 tissue model.
* Places a circular cyst inside the tissue.
* Computes the wave position as it travels through the tissue.

**Output:**

* `tissue_matrix`
* `wave_pos`

These are passed to the propagation stage.

---

## 4. `simulation/wave_propagation.py`

**Input:**

* `tissue_matrix`
* `wave_pos`
* `elem_pos`

**What it does:**

* Converts every tissue cell into its physical `(x, y)` coordinate.
* Determines which tissue cells have been reached by the transmitted wave.
* Computes the distance from every illuminated cell to every transducer element.
* Calculates the travel time (time delay).

**Output:**

* `cell_pos`
* `cell_near`
* `dist`
* `time_delay`

These values are used during echo simulation.

---

## 5. `simulation/scatter.py` (Current Work)

**Input:**

* `cell_near`
* `dist`
* Tissue information from `tissue_matrix`

**What it does:**

* Converts illuminated cell coordinates back into tissue indices.
* Computes attenuation for the returning echoes.
* (Currently under development) Calculates the echo strength produced by each illuminated tissue cell.

**Planned Output:**

* Echo amplitudes for every illuminated cell.

These echoes will later become the input for the receive and beamforming stages.
