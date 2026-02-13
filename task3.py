"""
N-Body Gravitational Simulation using Leapfrog Integration
============================================================
This code simulates the gravitational interaction of N bodies in 3D space.

"""

import math
import csv
from nbody_visualizer import draw_gui



# ============================================================================
# BODY CLASS
# ============================================================================
class Body:
    """Represents a celestial body with mass, position, velocity, and acceleration"""
    
    def __init__(self, mass, x, y, z, vx, vy, vz):
        """
        Initialize a body with its physical properties.
        
        Args:
            mass: mass of the body
            x, y, z: initial position coordinates
            vx, vy, vz: initial velocity components
        """
        self.mass = mass
        
        # Position (3D coordinates)
        self.x = x
        self.y = y
        self.z = z
        
        # Velocity (3D components)
        self.vx = vx
        self.vy = vy
        self.vz = vz
        
        # Acceleration (will be calculated by force computation)
        self.ax = 0.0
        self.ay = 0.0
        self.az = 0.0
    
    def __str__(self):
        """String representation for easy printing"""
        return (f"Body(mass={self.mass:.2e}, "
                f"pos=({self.x:.2e}, {self.y:.2e}, {self.z:.2e}), "
                f"vel=({self.vx:.2e}, {self.vy:.2e}, {self.vz:.2e}))")


# ============================================================================
# FORCE CALCULATION
# ============================================================================
def calculate_forces(bodies, G, epsilon):
    """
    Calculate gravitational forces between all pairs of bodies and update accelerations.
    
    This function implements Newton's law of universal gravitation:
        F = G * m1 * m2 / r^2
    
    With softening to prevent numerical instability when bodies are very close:
        F = G * m1 * m2 / (r^2 + epsilon^2)^(3/2)
    
    Args:
        bodies: List of Body objects
        G: Gravitational constant
        epsilon: Softening parameter (prevents division by zero)
    """
    # Step 1: Reset all accelerations to zero
    # (We'll accumulate forces from all other bodies)
    for body in bodies:
        body.ax = 0.0
        body.ay = 0.0
        body.az = 0.0
    
    # Step 2: Loop through all unique pairs of bodies
    # We use i and j indices where j > i to avoid calculating the same pair twice
    for i in range(len(bodies)):
        for j in range(i + 1, len(bodies)):
            body1 = bodies[i]
            body2 = bodies[j]
            
            # Calculate displacement vector from body1 to body2
            dx = body2.x - body1.x
            dy = body2.y - body1.y
            dz = body2.z - body1.z
            
            # Calculate distance between the two bodies
            r = math.sqrt(dx**2 + dy**2 + dz**2)
            
            # Apply softening: prevents numerical explosion when r is very small
            # r_soft = sqrt(r^2 + epsilon^2)
            r_soft = math.sqrt(r**2 + epsilon**2)
            
            # Calculate force magnitude
            # We use r_soft^3 instead of r_soft^2 because we need to normalize
            # the direction vector (dx, dy, dz) by dividing by r
            # So: F_vec = (G * m1 * m2 / r_soft^2) * (dx, dy, dz) / r
            #           = (G * m1 * m2 / r_soft^3) * (dx, dy, dz)
            force_magnitude = G * body1.mass * body2.mass / (r_soft**3)
            
            # Calculate force components (force is along the line connecting the bodies)
            fx = force_magnitude * dx
            fy = force_magnitude * dy
            fz = force_magnitude * dz
            
            # Update accelerations using Newton's 2nd law: a = F/m
            # Newton's 3rd law: force on body1 from body2 is equal and opposite
            # to force on body2 from body1
            
            # Body1 is attracted toward body2 (positive direction)
            body1.ax += fx / body1.mass
            body1.ay += fy / body1.mass
            body1.az += fz / body1.mass
            
            # Body2 is attracted toward body1 (negative direction)
            body2.ax -= fx / body2.mass
            body2.ay -= fy / body2.mass
            body2.az -= fz / body2.mass


# ============================================================================
# LEAPFROG INTEGRATION FUNCTIONS
# ============================================================================
def kick_half_step(bodies, dt):
    """
    Update velocities by half a timestep using current accelerations.
    
    This is the 'kick' part of the kick-drift-kick leapfrog method.
    Formula: v_new = v_old + a * (dt/2)
    
    Args:
        bodies: List of Body objects
        dt: Full timestep size (we use dt/2 in this function)
    """
    half_dt = dt / 2.0
    
    for body in bodies:
        body.vx += body.ax * half_dt
        body.vy += body.ay * half_dt
        body.vz += body.az * half_dt


def drift(bodies, dt):
    """
    Update positions by a full timestep using current velocities.
    
    This is the 'drift' part of the kick-drift-kick leapfrog method.
    Formula: x_new = x_old + v * dt
    
    Args:
        bodies: List of Body objects
        dt: Full timestep size
    """
    for body in bodies:
        body.x += body.vx * dt
        body.y += body.vy * dt
        body.z += body.vz * dt


# ============================================================================
# LOAD INITIAL CONDITIONS FROM CSV
# ============================================================================
def load_initial_conditions(filename):
    """
    Load initial masses, positions, and velocities from CSV file.
    
    Expected CSV format:
        mass,distanceX,distanceY,distanceZ,velocityX,velocityY,velocityZ
        [data rows...]
    
    Args:
        filename: Path to the CSV file
        
    Returns:
        List of Body objects initialized with data from file
    """
    bodies = []
    
    print(f"Loading initial conditions from: {filename}")
    
    with open(filename, 'r') as f:
        csv_reader = csv.DictReader(f)
        
        for row in csv_reader:
            # Parse values from CSV
            mass = float(row['mass'])
            x = float(row['distanceX'])
            y = float(row['distanceY'])
            z = float(row['distanceZ'])
            vx = float(row['velocityX'])
            vy = float(row['velocityY'])
            vz = float(row['velocityZ'])
            
            # Create Body object and add to list
            body = Body(mass, x, y, z, vx, vy, vz)
            bodies.append(body)
    
    print(f"Successfully loaded {len(bodies)} bodies")
    return bodies



# ============================================================================
# MAIN SIMULATION FUNCTION
# ============================================================================
def run_simulation(filename, num_steps, dt, G=6.67430e-11, epsilon=1e8,
                   visualization_interval=100):
    """
    Run the N-body gravitational simulation using leapfrog integration
    with GUI visualization.
    """

    print("\n" + "="*70)
    print("N-BODY GRAVITATIONAL SIMULATION")
    print("="*70)

    bodies = load_initial_conditions(filename)

    print(f"\nSimulation Parameters:")
    print(f"  Number of bodies: {len(bodies)}")
    print(f"  Number of steps: {num_steps}")
    print(f"  Timestep (dt): {dt} seconds")
    print(f"  Gravitational constant (G): {G}")
    print(f"  Softening parameter (epsilon): {epsilon}")

    print("\nCalculating initial forces...")
    calculate_forces(bodies, G, epsilon)

    print("\n" + "="*70)
    print("STARTING SIMULATION")
    print("="*70)

    for step in range(num_steps):

        # --- LEAPFROG INTEGRATION ---

        # 1. Kick (half-step)
        kick_half_step(bodies, dt)

        # 2. Drift (full step)
        drift(bodies, dt)

        # 3. Recalculate forces
        calculate_forces(bodies, G, epsilon)

        # 4. Kick (half-step)
        kick_half_step(bodies, dt)

        # --- GUI Visualization ---

        p_x = [body.x for body in bodies]
        p_y = [body.y for body in bodies]
        p_z = [body.z for body in bodies]

        if not draw_gui(p_x, p_y, p_z):
            print("Simulation stopped by user")
            break

        if step % visualization_interval == 0:
            print(f"Step {step}/{num_steps}")

    return bodies



# ============================================================================
# EXAMPLE USAGE
# ============================================================================
if __name__ == "__main__":
    """
    Example simulation using the solar100.csv data.
    
    ADJUSTABLE PARAMETERS:
    - INPUT_FILE: Path to your CSV file
    - NUM_STEPS: How many timesteps to simulate
    - DT: Size of each timestep (smaller = more accurate but slower)
    - G: Gravitational constant (SI units: 6.67430e-11)
    - EPSILON: Softening parameter (prevents numerical instability)
    - VIS_INTERVAL: How often to print status (every N steps)
    """
    
    # =================================================================
    # CONFIGURATION - ADJUST THESE PARAMETERS
    # =================================================================
    
    INPUT_FILE = "solar100.csv"        # Your data file
    NUM_STEPS = 1000                    # Number of simulation steps
    DT = 3600                           # Timestep: 1 hour = 3600 seconds
    G = 6.67430e-11                     # Gravitational constant (SI units)
    EPSILON = 1e8                       # Softening: 100,000 km
    VIS_INTERVAL = 100                  # Print status every 100 steps
   
    # =================================================================
    # RUN SIMULATION
    # =================================================================
    
    final_bodies = run_simulation(
        filename=INPUT_FILE,
        num_steps=NUM_STEPS,
        dt=DT,
        G=G,
        epsilon=EPSILON,
        visualization_interval=VIS_INTERVAL,
       
    )
    
    # Optional: Print final state of all bodies
    print("\n" + "="*70)
    print("FINAL STATE OF ALL BODIES")
    print("="*70)
    for i, body in enumerate(final_bodies[:10]):  # Show first 10
        print(body)
    if len(final_bodies) > 10:
        print(f"... and {len(final_bodies) - 10} more bodies")