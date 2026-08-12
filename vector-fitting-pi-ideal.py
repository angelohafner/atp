import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib.animation import FuncAnimation

# =============================================================================
# 1. PHYSICAL PARAMETERS & SIMULATION SETUP
# =============================================================================
l = 100.0  # Line length [km]
L_km = 1e-3  # Inductance [H/km]
C_km = 10e-9  # Capacitance [F/km]
R_km = 0.02  # Resistance [Ohm/km] (Low resistance to see reflections)

# Derived parameters
v_phase = 1 / np.sqrt(L_km * C_km)  # Phase velocity [km/s]
tau = l / v_phase  # Travel time [s]
Zc = np.sqrt(L_km / C_km)  # Characteristic Impedance [Ohm]

# Simulation time settings
t_final = 2.5 * tau
t = np.linspace(0, t_final, 600)
dt = t[1] - t[0]

# Input Signal: Step at t = 0.1 * tau
u_step_t = np.where(t >= 0.1 * tau, 1.0, 0.0)

# =============================================================================
# 2. MODEL 1: IDEAL DISTRIBUTED (REFERENCE)
# =============================================================================
# For an open-ended line, the voltage doubles at the end (Reflection = 2.0 pu)
y_ideal_t = np.zeros_like(t)
delay_idx = np.searchsorted(t, 0.1 * tau + tau)
if delay_idx < len(t):
    y_ideal_t[delay_idx:] = 2.0

# =============================================================================
# 3. MODEL 2: 5-PI SECTIONS (ORDER 10 LUMPED MODEL)
# =============================================================================
N = 5  # Number of sections
Rn, Ln, Cn = (R_km * l) / N, (L_km * l) / N, (C_km * l) / N
num_states = 2 * N

# State-space construction: [i1, v1, i2, v2, ..., iN, vN]
A = np.zeros((num_states, num_states))
B = np.zeros((num_states, 1))

for i in range(N):
    A[2 * i, 2 * i] = -Rn / Ln
    if i == 0:
        B[2 * i, 0] = 1 / Ln
    else:
        A[2 * i, 2 * i - 1] = 1 / Ln
    A[2 * i, 2 * i + 1] = -1 / Ln
    A[2 * i + 1, 2 * i] = 1 / Cn
    if i < N - 1:
        A[2 * i + 1, 2 * i + 2] = -1 / Cn

# C matrices for temporal (last node voltage) and spatial (all node voltages)
C_temp = np.zeros((1, num_states))
C_temp[0, -1] = 1.0
C_spat = np.zeros((N, num_states))
for i in range(N):
    C_spat[i, 2 * i + 1] = 1.0

sys_pi_temp = signal.StateSpace(A, B, C_temp, np.zeros((1, 1)))
sys_pi_spat = signal.StateSpace(A, B, C_spat, np.zeros((N, 1)))

# Simulate Pi-sections
_, y_pi_t, _ = signal.lsim(sys_pi_temp, u_step_t, t)
_, y_pi_s_all, _ = signal.lsim(sys_pi_spat, np.ones_like(t), t)

# =============================================================================
# 4. MODEL 3: VECTOR FITTING (ORDER 10 RATIONAL APPROXIMATION)
# =============================================================================
# Simulates the EMT-style VF: Exact delay + 10th order shaping filter
fc = 6 / tau  # Cutoff frequency
nyquist = 0.5 / dt
wn = min(0.95, fc / nyquist)
b_vf, a_vf = signal.butter(10, wn)
y_vf_t = signal.lfilter(b_vf, a_vf, y_ideal_t)

# =============================================================================
# 5. ANIMATION SETUP
# =============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
plt.subplots_adjust(bottom=0.15, wspace=0.25)

# --- Subplot 1: Temporal (Oscilloscope at x = 100 km) ---
ax1.set_xlim(0, t_final * 1e3)
ax1.set_ylim(-0.2, 2.5)
ax1.set_title('Temporal Response at Line End (x = 100 km)', fontweight='bold')
ax1.set_xlabel('Time [ms]')
ax1.set_ylabel('Voltage [pu]')
ax1.grid(True, alpha=0.3)

l_ideal_t, = ax1.plot([], [], 'r-', lw=2, label='Ideal (Reference)')
l_pi_t, = ax1.plot([], [], 'g-', lw=1.5, label='5-Pi Sections (Order 10)')
l_vf_t, = ax1.plot([], [], 'm-', lw=2, label='Vector Fitting (Order 10)')
ax1.legend(loc='upper right', fontsize='small')

# --- Subplot 2: Spatial (Wave propagation along x) ---
ax2.set_xlim(0, l)
ax2.set_ylim(-0.2, 2.5)
ax2.set_title('Spatial Voltage Distribution V(x)', fontweight='bold')
ax2.set_xlabel('Distance [km]')
ax2.set_ylabel('Voltage [pu]')
ax2.grid(True, alpha=0.3)

x_fine = np.linspace(0, l, 400)
x_nodes = np.linspace(0, l, N + 1)
l_ideal_s, = ax2.plot([], [], 'r-', lw=2, label='Ideal')
l_pi_s, = ax2.plot([], [], 'g-o', markersize=5, lw=1, label='5-Pi')
l_vf_s, = ax2.plot([], [], 'm-', lw=2, label='VF')
ax2.legend(loc='upper right', fontsize='small')

time_text = fig.text(0.5, 0.05, '', ha='center', fontweight='bold', fontsize=12)


def update(frame):
    curr_t = t[frame]

    # Update Temporal
    idx = frame
    l_ideal_t.set_data(t[:idx] * 1e3, y_ideal_t[:idx])
    l_pi_t.set_data(t[:idx] * 1e3, y_pi_t[:idx])
    l_vf_t.set_data(t[:idx] * 1e3, y_vf_t[:idx])

    # Update Spatial
    v_inc = np.where(curr_t >= x_fine / v_phase, 1.0, 0.0)
    v_refl = np.where(curr_t >= (2 * l - x_fine) / v_phase, 1.0, 0.0)
    v_ideal_s = v_inc + v_refl
    l_ideal_s.set_data(x_fine, v_ideal_s)

    v_pi_s = np.concatenate(([1.0], y_pi_s_all[frame, :]))
    l_pi_s.set_data(x_nodes, v_pi_s)

    v_vf_s = signal.lfilter(b_vf, a_vf, v_ideal_s)
    l_vf_s.set_data(x_fine, v_vf_s)

    time_text.set_text(f'Simulation Time: {curr_t * 1e3:.3f} ms')
    return l_ideal_t, l_pi_t, l_vf_t, l_ideal_s, l_pi_s, l_vf_s, time_text


# Create animation
ani = FuncAnimation(fig, update, frames=range(0, len(t), 2), blit=True, interval=20)

# To save the animation, uncomment the line below:
ani.save('transmission_line_comparison.gif', writer='pillow', fps=30)

# plt.show()