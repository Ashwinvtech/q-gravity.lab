import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
from matplotlib.animation import FuncAnimation # type: ignore

# Constants
G = 1.0

# -----------------------------
# Physics core
# -----------------------------
class NBodySim:
    def __init__(self, pos, vel, mass, dt=0.01, softening=1e-2):
        self.pos = pos.astype(float)    # (N,2)
        self.vel = vel.astype(float)    # (N,2)
        self.m = mass.astype(float)     # (N,)
        self.dt = float(dt)
        self.eps2 = float(softening) ** 2
        self.N = self.pos.shape[0]

    def accel(self, pos):
        a = np.zeros_like(pos)
        # O(N^2) pairwise forces (fine for N<=200ish)
        for i in range(self.N):
            r = pos - pos[i]  # (N,2)
            dist2 = np.sum(r * r, axis=1) + self.eps2
            inv = 1.0 / np.sqrt(dist2 ** 3)
            inv[i] = 0.0
            a[i] = G * np.sum((self.m[:, None] * r) * inv[:, None], axis=0)
        return a

    def step_leapfrog(self, substeps=1):
        dt = self.dt / substeps
        for _ in range(substeps):
            a0 = self.accel(self.pos)
            self.vel += 0.5 * dt * a0
            self.pos += dt * self.vel
            a1 = self.accel(self.pos)
            self.vel += 0.5 * dt * a1

    def energy(self):
        # kinetic
        K = 0.5 * np.sum(self.m * np.sum(self.vel * self.vel, axis=1))
        # potential
        U = 0.0
        for i in range(self.N):
            for j in range(i + 1, self.N):
                r = self.pos[j] - self.pos[i]
                dist = np.sqrt(np.dot(r, r) + self.eps2)
                U -= G * self.m[i] * self.m[j] / dist
        return K + U

    def angular_momentum_z(self):
        # Lz = sum m (x vy - y vx)
        x, y = self.pos[:, 0], self.pos[:, 1]
        vx, vy = self.vel[:, 0], self.vel[:, 1]
        return np.sum(self.m * (x * vy - y * vx))

# -----------------------------
# Presets
# -----------------------------
def preset_figure8():
    # Classic equal-mass figure-8 initial condition (scaled)
    m = np.array([1.0, 1.0, 1.0])
    pos = np.array([
        [-0.97000436,  0.24308753],
        [ 0.97000436, -0.24308753],
        [ 0.0,         0.0       ],
    ])
    vel = np.array([
        [ 0.4662036850,  0.4323657300],
        [ 0.4662036850,  0.4323657300],
        [-0.93240737,   -0.86473146  ],
    ])
    return "Figure-8 (3-body)", pos, vel, m, None

def preset_binary_intruder():
    # Two heavy bodies orbiting + a lighter intruder
    m = np.array([2.0, 2.0, 0.2])
    pos = np.array([[-0.8, 0.0], [0.8, 0.0], [0.0, 2.2]])
    vel = np.array([[0.0, 0.45], [0.0, -0.45], [0.55, 0.0]])
    return "Binary + intruder", pos, vel, m, None

def preset_slingshot_probe():
    # A "probe" (index 2) approaches a heavy body (index 0) with a smaller companion (index 1)
    # We'll mark probe_index=2 so optimizer can tune probe velocity.
    m = np.array([5.0, 1.0, 0.01])
    pos = np.array([[0.0, 0.0], [1.2, 0.0], [-3.0, 1.0]])
    vel = np.array([[0.0, 0.0], [0.0, 1.4], [1.2, -0.15]])
    probe_index = 2
    return "Slingshot probe (optimizer-ready)", pos, vel, m, probe_index

def preset_chaos_triangle():
    m = np.array([1.0, 1.0, 1.0])
    pos = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, 0.9]])
    vel = np.array([[0.25, 0.35], [-0.25, 0.35], [0.0, -0.7]])
    return "Chaotic triangle", pos, vel, m, None

PRESETS = [preset_figure8, preset_binary_intruder, preset_slingshot_probe, preset_chaos_triangle
           ]

# =============================
# Auto-zoom helper functions
# =============================
def center_of_mass(pos, m):
    return np.sum(pos * m[:, None], axis=0) / np.sum(m)

def compute_view_bounds(pos, center, margin=1.5, min_size=2.0):
    d = np.linalg.norm(pos - center[None, :], axis=1)
    r = max(d.max() * margin, min_size)
    return center[0] - r, center[0] + r, center[1] - r, center[1] + r

# -----------------------------
# "Quantum-inspired" optimizer
# (simulated annealing-style search)
# -----------------------------
def evaluate_orbit_cost(sim_init, probe_index, target_r=1.6, steps=2500, substeps=3):
    """
    Lower is better.
    Goal: keep probe bound + hover around target radius from primary (body 0).
    """
    sim = NBodySim(sim_init.pos.copy(), sim_init.vel.copy(), sim_init.m.copy(),
                   dt=sim_init.dt, softening=np.sqrt(sim_init.eps2))
    e0 = sim.energy()
    # Track radius from primary and whether it escapes
    rs = []
    escaped_penalty = 0.0
    for _ in range(steps):
        sim.step_leapfrog(substeps=substeps)
        r = np.linalg.norm(sim.pos[probe_index] - sim.pos[0])
        rs.append(r)
        if r > 8.0:  # "escape" threshold in our scaled units
            escaped_penalty = 50.0
            break

    rs = np.array(rs)
    # prefer being bound: energy of probe+primary roughly negative (heuristic)
    # simple: penalize large radius drift + escaping
    radius_cost = np.mean((rs - target_r) ** 2) if len(rs) > 10 else 10.0
    drift_cost = abs(sim.energy() - e0) * 0.2  # encourage stability (not too strict)
    return radius_cost + drift_cost + escaped_penalty

def simulated_annealing_velocity(sim, probe_index, v0, iters=120, T0=0.6, alpha=0.93, step0=0.35):
    """
    Searches probe initial velocity near v0.
    Returns best velocity and best cost.
    """
    best_v = v0.copy()
    sim_best = NBodySim(sim.pos.copy(), sim.vel.copy(), sim.m.copy(), dt=sim.dt, softening=np.sqrt(sim.eps2))
    sim_best.vel[probe_index] = best_v
    best_cost = evaluate_orbit_cost(sim_best, probe_index)

    cur_v = best_v.copy()
    cur_cost = best_cost

    T = T0
    step = step0
    rng = np.random.default_rng()

    for k in range(iters):
        # propose
        proposal = cur_v + rng.normal(0.0, step, size=2)
        sim_try = NBodySim(sim.pos.copy(), sim.vel.copy(), sim.m.copy(), dt=sim.dt, softening=np.sqrt(sim.eps2))
        sim_try.vel[probe_index] = proposal
        cost = evaluate_orbit_cost(sim_try, probe_index)

        # accept
        if cost < cur_cost or rng.random() < np.exp(-(cost - cur_cost) / max(T, 1e-9)):
            cur_v, cur_cost = proposal, cost
            if cost < best_cost:
                best_v, best_cost = proposal, cost

        T *= alpha
        step *= 0.995

    return best_v, best_cost

# -----------------------------
# Visualization + UI
# -----------------------------
def main():
    preset_idx = 0

    def load_preset(idx):
        name, pos, vel, m, probe_index = PRESETS[idx]()
        sim = NBodySim(pos, vel, m, dt=0.01, softening=1e-2)
        return name, sim, probe_index

    name, sim, probe_index = load_preset(preset_idx)

    # Figure layout: main axes + optional diagnostics
    show_diag = True
    fig = plt.figure(figsize=(9, 6))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1])
    ax = fig.add_subplot(gs[0, :])
    axE = fig.add_subplot(gs[1, 0])
    axL = fig.add_subplot(gs[1, 1])

    def configure_axes():
        ax.set_aspect("equal")
        ax.set_xlim(-4, 4)
        ax.set_ylim(-3, 3)
        ax.set_title(f"Q-Gravity Lab — {name}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    configure_axes()



 # --- Camera state for auto-zoom ---
    camera = {
        "cx": 0.0,
        "cy": 0.0,
        "r": 4.0,
        "smooth": 0.08,  # lower = smoother camera
    }
    auto_zoom = True



    scat = ax.scatter(sim.pos[:, 0], sim.pos[:, 1], s=(sim.m / sim.m.max()) * 120 + 30)
    trails = [ax.plot([], [], lw=1)[0] for _ in range(sim.N)]
    history = [[] for _ in range(sim.N)]
    trail_cap_short = 250
    trail_cap_long = 1200
    trail_cap = trail_cap_long

    # diagnostics buffers
    E_hist, L_hist = [], []
    t_hist = []
    E0 = sim.energy()
    L0 = sim.angular_momentum_z()

    lineE, = axE.plot([], [])
    lineL, = axL.plot([], [])

    def setup_diag_axes():
        axE.set_title("Total Energy (Δ from start)")
        axL.set_title("Angular Momentum Lz (Δ from start)")
        axE.set_xlabel("step")
        axL.set_xlabel("step")
        axE.set_ylabel("ΔE")
        axL.set_ylabel("ΔLz")
    setup_diag_axes()

    paused = {"v": False}

    help_text = (
        "Controls: Space pause | r reset | n/p preset | +/- speed | t trails | e diag | o optimize(probe)"
    )
    text = ax.text(0.02, 0.98, help_text, transform=ax.transAxes, va="top", fontsize=9)

    def reset_buffers():
        nonlocal E0, L0
        for h in history:
            h.clear()
        E_hist.clear(); L_hist.clear(); t_hist.clear()
        E0 = sim.energy()
        L0 = sim.angular_momentum_z()

    def apply_preset(new_idx):
        nonlocal preset_idx, name, sim, probe_index
        preset_idx = new_idx % len(PRESETS)
        name, sim, probe_index = load_preset(preset_idx)
        configure_axes()
        scat.set_sizes((sim.m / sim.m.max()) * 120 + 30)
        reset_buffers()

    def on_key(event):
        nonlocal trail_cap, show_diag
        if event.key == " ":
            paused["v"] = not paused["v"]
        elif event.key == "r":
            apply_preset(preset_idx)
        elif event.key == "n":
            apply_preset(preset_idx + 1)
        elif event.key == "p":
            apply_preset(preset_idx - 1)
        elif event.key == "+":
            sim.dt *= 1.2
        elif event.key == "-":
            sim.dt /= 1.2
        elif event.key == "t":
            trail_cap = trail_cap_short if trail_cap == trail_cap_long else trail_cap_long
        elif event.key == "e":
            show_diag = not show_diag
            axE.set_visible(show_diag)
            axL.set_visible(show_diag)
            fig.canvas.draw_idle()
        elif event.key == "o":
            if probe_index is None:
                print("Optimizer: this preset has no probe_index. Switch to 'Slingshot probe'.")
                return
            print("Running simulated annealing optimizer for probe velocity...")
            v0 = sim.vel[probe_index].copy()
            best_v, best_cost = simulated_annealing_velocity(sim, probe_index, v0)
            sim.vel[probe_index] = best_v
            reset_buffers()
            print(f"Done. Best cost={best_cost:.4f}, v={best_v}")

    fig.canvas.mpl_connect("key_press_event", on_key)

    step_counter = {"k": 0}

    def update(_):
        if not paused["v"]:
            # substeps improves stability for close passes
            sim.step_leapfrog(substeps=3)


# --------------------------
        # AUTO-ZOOM CAMERA (PASTE HERE)
        # --------------------------
        if auto_zoom:
            com = center_of_mass(sim.pos, sim.m)
            x0, x1, y0, y1 = compute_view_bounds(sim.pos, com)

            camera["cx"] += camera["smooth"] * (com[0] - camera["cx"])
            camera["cy"] += camera["smooth"] * (com[1] - camera["cy"])

            target_r = max(x1 - x0, y1 - y0) / 2
            camera["r"] += camera["smooth"] * (target_r - camera["r"])

            ax.set_xlim(camera["cx"] - camera["r"], camera["cx"] + camera["r"])
            ax.set_ylim(camera["cy"] - camera["r"], camera["cy"] + camera["r"])

            
        # Update scatter
        scat.set_offsets(sim.pos)

        # Trails
        for i in range(sim.N):
            history[i].append(sim.pos[i].copy())
            if len(history[i]) > trail_cap:
                history[i].pop(0)
            h = np.array(history[i])
            trails[i].set_data(h[:, 0], h[:, 1])

        # Diagnostics
        step_counter["k"] += 1
        if show_diag and (step_counter["k"] % 2 == 0):
            t_hist.append(step_counter["k"])
            E_hist.append(sim.energy() - E0)
            L_hist.append(sim.angular_momentum_z() - L0)

            lineE.set_data(t_hist, E_hist)
            lineL.set_data(t_hist, L_hist)

            # autoscale
            axE.relim(); axE.autoscale_view()
            axL.relim(); axL.autoscale_view()

        return [scat, *trails, lineE, lineL, text]

    ani = FuncAnimation(fig, update, interval=16, blit=True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()