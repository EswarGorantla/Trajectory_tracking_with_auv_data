# Project Methodology
Mathematical details behind the dynamics modeling, system identification, neural networks, and receding-horizon control.

### AUV State Representation
The state vector is defined as:
$$\mathbf{x} = [x, y, z, \phi, \theta, \psi, U]^T$$

Where:
* $x, y$: North-East Cartesian relative position (m)
* $z$: Depth (m)
* $\phi, \theta, \psi$: Roll, Pitch, and Heading Euler angles (rad)
* $U$: Surge speed (m/s)

### System Identification
The drag parameter $K_d$ was identified using least squares regression:
$$K_d = 0.022498$$
