# SPH Fluid Simulation with Terrain

This project implements a 3D Smoothed Particle Hydrodynamics (SPH) fluid simulation with an interactive terrain using Python, Pygame, and OpenGL.

## Features

- 3D fluid simulation using SPH method
- Interactive terrain with crater-like heightmap
- Real-time rendering using OpenGL
- Camera rotation for better view control
- Earthquake simulation with spacebar press

## Installation

1. Ensure you have Python 3.x installed on your system.
2. Install the required dependencies:

```
poetry install
poetry shell
```

## Usage

Run the simulation by executing the main script:

```
python main.py
```

## Controls

- Left mouse button: Click and drag to rotate the camera
- Spacebar: Trigger an earthquake in the simulation

## How it works

The simulation uses the SPH method to model fluid dynamics. Key components include:

1. Particle System: Represents the fluid as a collection of particles.
2. Grid: Optimizes neighbor search for particle interactions.
3. Terrain: Generates a crater-like landscape for the fluid to interact with.
4. Forces: Computes pressure, viscosity, and gravity forces acting on particles.
5. Rendering: Visualizes particles and terrain using OpenGL.

## Customization

You can adjust various parameters in the code to change the simulation behavior:

- `PARTICLE_MASS`: Affects the mass of each particle
- `H`: Smoothing length for SPH calculations
- `GAS_CONSTANT`: Influences pressure calculations
- `VISCOSITY`: Controls the fluid's resistance to flow
- `GRAVITY`: Adjusts the strength and direction of gravity

## Performance Note

The simulation is computationally intensive. If you experience low framerates, try reducing the number of particles in the `initialize_particles` method.

## License

This project is open-source and available under the MIT License.