"""
N-Body Gravitational Simulation using Leapfrog Integration
============================================================
This code simulates the gravitational interaction of N bodies in 3D space.
It now uses the Barnes-Hut algorithm to reduce complexity from O(N^2) to O(N log N)
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
# BARNES-HUT OCTREE CLASSES AND FUNCTIONS
# ============================================================================
class OctreeNode:
    """3D Octree node for Barnes-Hut"""
    def __init__(self, x_min, x_max, y_min, y_max, z_min, z_max): #when we insert a body, we check if its position is inside these bounds.
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        self.z_min, self.z_max = z_min, z_max
        
        self.body = None          # Reference to body if leaf
        self.children = [None]*8  # 8 child nodes. Each child is also an OctreeNode or None if that subcube is empty.
        self.mass = 0.0 #the total mass of all bodies inside this node
        self.com = (0.0, 0.0, 0.0) #centre of mass
        self.is_internal = False #false means leaf node

    def contains(self, body):
        """Check if body is inside this node"""
        return (self.x_min <= body.x <= self.x_max and
                self.y_min <= body.y <= self.y_max and
                self.z_min <= body.z <= self.z_max)

    def subdivide(self):
        """If more than one body falls inside the same region → split the region into 8 smaller regions."""
        xm = (self.x_min + self.x_max)/2
        ym = (self.y_min + self.y_max)/2
        zm = (self.z_min + self.z_max)/2
        self.children = [
            OctreeNode(self.x_min, xm, self.y_min, ym, self.z_min, zm),
            OctreeNode(xm, self.x_max, self.y_min, ym, self.z_min, zm),
            OctreeNode(self.x_min, xm, ym, self.y_max, self.z_min, zm),
            OctreeNode(xm, self.x_max, ym, self.y_max, self.z_min, zm),
            OctreeNode(self.x_min, xm, self.y_min, ym, zm, self.z_max),
            OctreeNode(xm, self.x_max, self.y_min, ym, zm, self.z_max),
            OctreeNode(self.x_min, xm, ym, self.y_max, zm, self.z_max),
            OctreeNode(xm, self.x_max, ym, self.y_max, zm, self.z_max)
        ]
        self.is_internal = True

    def insert(self, body):
        """Insert body into octree recursively"""
        if not self.contains(body):
            return False

        if self.body is None and not self.is_internal:
            # Empty leaf → store body
            self.body = body
            self.mass = body.mass
            self.com = (body.x, body.y, body.z)
            return True
        else:
            # Leaf has one body → subdivide
            if not self.is_internal:
                self.subdivide()
                existing_body = self.body
                self.body = None
                self.insert(existing_body)

            # Insert body into the correct child
            for child in self.children:
                if child.insert(body):
                    break

            # Update mass & center of mass
            total_mass = sum(c.mass for c in self.children if c)
            if total_mass > 0:
                x_com = sum(c.com[0]*c.mass for c in self.children if c)/total_mass
                y_com = sum(c.com[1]*c.mass for c in self.children if c)/total_mass
                z_com = sum(c.com[2]*c.mass for c in self.children if c)/total_mass
                self.mass = total_mass
                self.com = (x_com, y_com, z_com)

            return True

def build_tree(bodies):
    """Constructs the octree for the current timestep"""
    xs = [b.x for b in bodies]
    ys = [b.y for b in bodies]
    zs = [b.z for b in bodies]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    min_z, max_z = min(zs), max(zs)

    root = OctreeNode(min_x, max_x, min_y, max_y, min_z, max_z)
    for b in bodies:
        root.insert(b)
    return root

def calculate_force(body, node, G=6.67430e-11, theta=0.5, epsilon=1e8):
    """
    Recursively calculates acceleration on a body due to the Barnes-Hut octree.
    Returns (ax, ay, az)
    """
    ax = ay = az = 0.0

    if node.mass == 0 or (node.body is body and not node.is_internal):
        return 0.0, 0.0, 0.0

    dx = node.com[0] - body.x
    dy = node.com[1] - body.y
    dz = node.com[2] - body.z
    r = math.sqrt(dx*dx + dy*dy + dz*dz + epsilon*epsilon)

    s = max(node.x_max - node.x_min, node.y_max - node.y_min, node.z_max - node.z_min)

    if node.is_internal and s / r > theta:
        # Too close → open children
        for child in node.children:
            if child:
                c_ax, c_ay, c_az = calculate_force(body, child, G, theta, epsilon)
                ax += c_ax
                ay += c_ay
                az += c_az
    else:
        # Far away or leaf → approximate as one body
        force = G * node.mass / (r**3)
        ax += force * dx
        ay += force * dy
        az += force * dz

    return ax, ay, az

def update_accelerations(bodies, tree, G=6.67430e-11, theta=0.5, epsilon=1e8):
    """Updates accelerations of all bodies using the Barnes-Hut tree"""
    for b in bodies:
        b.ax, b.ay, b.az = calculate_force(b, tree, G, theta, epsilon)


# ============================================================================ 
# LEAPFROG INTEGRATION FUNCTIONS (UNCHANGED)
# ============================================================================
def kick_half_step(bodies, dt):
    """Update velocities by half a timestep using current accelerations"""
    half_dt = dt / 2.0
    for body in bodies:
        body.vx += body.ax * half_dt
        body.vy += body.ay * half_dt
        body.vz += body.az * half_dt

def drift(bodies, dt):
    """Update positions by a full timestep using current velocities"""
    for body in bodies:
        body.x += body.vx * dt
        body.y += body.vy * dt
        body.z += body.vz * dt


# ============================================================================ 
# LOAD INITIAL CONDITIONS FROM CSV (UNCHANGED)
# ============================================================================
def load_initial_conditions(filename):
    bodies = []
    print(f"Loading initial conditions from: {filename}")
    with open(filename, 'r') as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            mass = float(row['mass'])
            x = float(row['distanceX'])
            y = float(row['distanceY'])
            z = float(row['distanceZ'])
            vx = float(row['velocityX'])
            vy = float(row['velocityY'])
            vz = float(row['velocityZ'])
            body = Body(mass, x, y, z, vx, vy, vz)
            bodies.append(body)
    print(f"Successfully loaded {len(bodies)} bodies")
    return bodies


# ============================================================================ 
# MAIN SIMULATION FUNCTION (UPDATED TO USE BARNES-HUT)
# ============================================================================
def run_simulation(filename, num_steps, dt, G=6.67430e-11, epsilon=1e8,
                   visualization_interval=100, theta=0.5):
    """
    Run the N-body gravitational simulation using leapfrog integration
    with Barnes-Hut optimization.
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
    print(f"  Barnes-Hut theta: {theta}")

    # Initial accelerations
    print("\nCalculating initial forces using Barnes-Hut...")
    tree = build_tree(bodies)
    update_accelerations(bodies, tree, G, theta, epsilon)

    print("\n" + "="*70)
    print("STARTING SIMULATION")
    print("="*70)

    for step in range(num_steps):

        # --- LEAPFROG INTEGRATION (kick-drift-build_tree-acceleration-kick) ---

        # 1. Kick (half-step)
        kick_half_step(bodies, dt)

        # 2. Drift (full step)
        drift(bodies, dt)

        # 3. Build Barnes-Hut tree
        tree = build_tree(bodies)

        # 4. Update accelerations using Barnes-Hut
        update_accelerations(bodies, tree, G, theta, epsilon)

        # 5. Kick (half-step)
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
    
    INPUT_FILE = "solar300.csv"        # Your data file
    NUM_STEPS = 1000                    # Number of simulation steps...in each step, values updated
    DT = 3600                          # Timestep
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