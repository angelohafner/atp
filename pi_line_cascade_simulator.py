import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from tqdm import tqdm
from PIL import Image


# =========================
# User-editable parameters
# =========================
DEFAULT_CONFIG = {
    "line": {
        "length_m": 30_000.0,
        "r_per_m_ohm": 8.0e-5,
        "l_per_m_h": 1.0e-6,
        "c_per_m_f": 10.0e-12,
    },
    "source": {
        "kind": "step",  # Options: "step", "dc", "ramp", "sin"
        "amplitude_v": 10_000.0,
        "frequency_hz": 1.0e3,
        "t_start_s": 0.0e-3,
        "rise_time_s": 8.0e-6,
        "dc_offset_v": 0.0,
        "resistance_ohm": 0.0,  # Source resistance for Bergeron reflections
    },
    "load": {
        "kind": "resistive",  # Options: "open", "resistive"
        "resistance_ohm": 2.0 * 316.23,
    },
    "simulation": {
        "n_values": [1, 10, 100],
        "time_end_s": 2.0e-3,
        "dt_s": 1.0e-6,
        "solver_method": "RK45",
        "rtol": 1.0e-6,
        "atol": 1.0e-9,
        "max_step_s": 1.0e-6,
    },
    "animation": {
        "fps": 5,
        "frame_stride": 2,
        "spatial_points": 100,
        "figsize": (12, 6),
        "dpi": 140,
        "y_margin": 0.10,
        "max_bounces": 40,
        "gif_name": "comparison_pi_1_10_100_bergeron.gif",
    },
}


# =========================
# Source waveform
# =========================
def source_voltage(t: float, source_cfg: dict) -> float:
    kind = source_cfg["kind"].lower()
    amplitude = float(source_cfg["amplitude_v"])
    frequency = float(source_cfg["frequency_hz"])
    t_start = float(source_cfg["t_start_s"])
    rise_time = max(float(source_cfg["rise_time_s"]), 1.0e-12)
    dc_offset = float(source_cfg.get("dc_offset_v", 0.0))

    if t < t_start:
        return dc_offset

    tau = t - t_start

    if kind == "dc":
        return dc_offset + amplitude
    if kind == "step":
        return dc_offset + amplitude
    if kind == "ramp":
        if tau <= rise_time:
            return dc_offset + amplitude * (tau / rise_time)
        return dc_offset + amplitude
    if kind == "sin":
        return dc_offset + amplitude * math.sin(2.0 * math.pi * frequency * tau)

    raise ValueError(f"Unsupported source kind: {kind}")


def source_voltage_array(t: np.ndarray, source_cfg: dict) -> np.ndarray:
    kind = source_cfg["kind"].lower()
    amplitude = float(source_cfg["amplitude_v"])
    frequency = float(source_cfg["frequency_hz"])
    t_start = float(source_cfg["t_start_s"])
    rise_time = max(float(source_cfg["rise_time_s"]), 1.0e-12)
    dc_offset = float(source_cfg.get("dc_offset_v", 0.0))

    t_arr = np.asarray(t, dtype=float)
    y = np.full(t_arr.shape, dc_offset, dtype=float)

    active_mask = t_arr >= t_start
    if not np.any(active_mask):
        return y

    tau = t_arr[active_mask] - t_start

    if kind == "dc":
        y[active_mask] = dc_offset + amplitude
        return y

    if kind == "step":
        y[active_mask] = dc_offset + amplitude
        return y

    if kind == "ramp":
        ramp_mask = tau <= rise_time
        y_active = np.empty_like(tau)
        y_active[ramp_mask] = dc_offset + amplitude * (tau[ramp_mask] / rise_time)
        y_active[~ramp_mask] = dc_offset + amplitude
        y[active_mask] = y_active
        return y

    if kind == "sin":
        y[active_mask] = dc_offset + amplitude * np.sin(2.0 * math.pi * frequency * tau)
        return y

    raise ValueError(f"Unsupported source kind: {kind}")


# =========================
# Pi-model parameters
# =========================
def build_pi_parameters(n_sections: int, line_cfg: dict) -> dict:
    length_total = float(line_cfg["length_m"])
    r_per_m = float(line_cfg["r_per_m_ohm"])
    l_per_m = float(line_cfg["l_per_m_h"])
    c_per_m = float(line_cfg["c_per_m_f"])

    dx = length_total / n_sections
    r_sec = r_per_m * dx
    l_sec = l_per_m * dx
    c_sec = c_per_m * dx

    node_caps = np.full(n_sections, c_sec, dtype=float)
    node_caps[-1] = 0.5 * c_sec

    return {
        "dx_m": dx,
        "r_sec_ohm": r_sec,
        "l_sec_h": l_sec,
        "c_sec_f": c_sec,
        "node_caps_f": node_caps,
        "z0_est_ohm": math.sqrt(l_per_m / c_per_m),
        "velocity_est_m_per_s": 1.0 / math.sqrt(l_per_m * c_per_m),
        "travel_time_s": length_total * math.sqrt(l_per_m * c_per_m),
    }


# =========================
# Pi-model ODE
# =========================
def line_ode(
    t: float,
    x: np.ndarray,
    n_sections: int,
    params: dict,
    source_cfg: dict,
    load_cfg: dict,
) -> np.ndarray:
    currents = x[:n_sections]
    voltages = x[n_sections:]

    r_sec = params["r_sec_ohm"]
    l_sec = params["l_sec_h"]
    node_caps = params["node_caps_f"]

    v_send = source_voltage(t, source_cfg)

    di_dt = np.zeros_like(currents)
    dv_dt = np.zeros_like(voltages)

    for k in range(n_sections):
        v_left = v_send if k == 0 else voltages[k - 1]
        v_right = voltages[k]
        di_dt[k] = (v_left - v_right - r_sec * currents[k]) / l_sec

    for j in range(n_sections - 1):
        dv_dt[j] = (currents[j] - currents[j + 1]) / node_caps[j]

    load_kind = load_cfg["kind"].lower()
    if load_kind == "open":
        i_load = 0.0
    elif load_kind == "resistive":
        r_load = max(float(load_cfg["resistance_ohm"]), 1.0e-9)
        i_load = voltages[-1] / r_load
    else:
        raise ValueError(f"Unsupported load kind: {load_kind}")

    dv_dt[-1] = (currents[-1] - i_load) / node_caps[-1]

    return np.concatenate([di_dt, dv_dt])


# =========================
# Pi-model simulation
# =========================
def simulate_case(n_sections: int, config: dict) -> dict:
    line_cfg = config["line"]
    source_cfg = config["source"]
    load_cfg = config["load"]
    sim_cfg = config["simulation"]

    params = build_pi_parameters(n_sections, line_cfg)

    dt = float(sim_cfg["dt_s"])
    time_end = float(sim_cfg["time_end_s"])

    n_steps = int(np.floor(time_end / dt))
    t_eval = np.arange(n_steps + 1, dtype=float) * dt

    if t_eval[-1] < time_end:
        t_eval = np.append(t_eval, time_end)
    else:
        t_eval[-1] = time_end

    x0 = np.zeros(2 * n_sections, dtype=float)

    solution = solve_ivp(
        fun=lambda t, x: line_ode(t, x, n_sections, params, source_cfg, load_cfg),
        t_span=(0.0, time_end),
        y0=x0,
        t_eval=t_eval,
        method=sim_cfg["solver_method"],
        rtol=sim_cfg["rtol"],
        atol=sim_cfg["atol"],
        max_step=min(float(sim_cfg["max_step_s"]), dt),
    )

    if not solution.success:
        raise RuntimeError(f"Simulation failed for N={n_sections}: {solution.message}")

    currents = solution.y[:n_sections, :]
    voltages = solution.y[n_sections:, :]
    source_trace = source_voltage_array(solution.t, source_cfg)

    return {
        "model_name": f"Pi N={n_sections}",
        "N": n_sections,
        "t": solution.t,
        "currents": currents,
        "voltages": voltages,
        "v_send": source_trace,
        "v_recv": voltages[-1, :],
        "i_send": currents[0, :],
        "params": params,
    }


# =========================
# Bergeron helpers
# =========================
def reflection_coefficient(z_term: float, z0: float) -> float:
    if np.isinf(z_term):
        return 1.0
    denom = z_term + z0
    if abs(denom) < 1.0e-12:
        return -1.0
    return (z_term - z0) / denom


def get_load_impedance(load_cfg: dict) -> float:
    load_kind = load_cfg["kind"].lower()
    if load_kind == "open":
        return np.inf
    if load_kind == "resistive":
        return max(float(load_cfg["resistance_ohm"]), 1.0e-12)
    raise ValueError(f"Unsupported load kind: {load_kind}")


def build_bergeron_profiles(frame_times: np.ndarray, x_positions_m: np.ndarray, config: dict) -> np.ndarray:
    line_cfg = config["line"]
    source_cfg = config["source"]
    load_cfg = config["load"]
    anim_cfg = config["animation"]

    length_m = float(line_cfg["length_m"])
    l_per_m = float(line_cfg["l_per_m_h"])
    c_per_m = float(line_cfg["c_per_m_f"])

    z0 = math.sqrt(l_per_m / c_per_m)
    velocity = 1.0 / math.sqrt(l_per_m * c_per_m)

    z_source = max(float(source_cfg.get("resistance_ohm", 0.0)), 0.0)
    z_load = get_load_impedance(load_cfg)

    gamma_s = reflection_coefficient(z_source, z0)
    gamma_l = reflection_coefficient(z_load, z0)

    launch_coeff = z0 / (z0 + z_source) if (z0 + z_source) > 1.0e-12 else 1.0
    max_bounces = int(anim_cfg["max_bounces"])

    t_grid = frame_times[np.newaxis, :]
    x_grid = x_positions_m[:, np.newaxis]

    v_total = np.zeros((len(x_positions_m), len(frame_times)), dtype=float)

    for m_idx in tqdm(range(max_bounces), desc="Bergeron reconstruction"):
        amp_forward = launch_coeff * (gamma_s * gamma_l) ** m_idx
        time_forward = t_grid - (2.0 * m_idx * length_m + x_grid) / velocity
        v_total = v_total + amp_forward * source_voltage_array(time_forward, source_cfg)

        amp_backward = launch_coeff * gamma_l * (gamma_s * gamma_l) ** m_idx
        time_backward = t_grid - ((2.0 * m_idx + 2.0) * length_m - x_grid) / velocity
        v_total = v_total + amp_backward * source_voltage_array(time_backward, source_cfg)

    return v_total


# =========================
# Spatial reconstruction for Pi
# =========================
def reconstruct_pi_profiles_for_frames(
    case_data: dict,
    x_query_m: np.ndarray,
    frame_indices: np.ndarray,
) -> np.ndarray:
    node_positions_m = np.concatenate(
        (
            np.array([0.0]),
            np.linspace(
                case_data["params"]["dx_m"],
                case_data["params"]["dx_m"] * case_data["N"],
                case_data["N"],
            ),
        )
    )

    voltage_nodes = np.vstack([case_data["v_send"], case_data["voltages"]])
    voltage_nodes_frames = voltage_nodes[:, frame_indices]

    spatial_matrix = np.empty((len(x_query_m), len(frame_indices)), dtype=float)

    for col in tqdm(range(len(frame_indices)), desc=f"Reconstructing Pi N={case_data['N']}"):
        spatial_matrix[:, col] = np.interp(x_query_m, node_positions_m, voltage_nodes_frames[:, col])

    return spatial_matrix


# =========================
# GIF export with progress bar
# =========================
def figure_to_image(fig: plt.Figure) -> Image.Image:
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).copy()
    rgba = rgba.reshape((height, width, 4))
    return Image.fromarray(rgba, mode="RGBA").copy()


def save_gif_from_profiles(
    x_km: np.ndarray,
    frame_times_ms: np.ndarray,
    pi_profiles_by_n_kv: dict,
    bergeron_profiles_kv: np.ndarray,
    gif_path: Path,
    config: dict,
) -> list:
    anim_cfg = config["animation"]

    all_mins = []
    all_maxs = []

    for arr in pi_profiles_by_n_kv.values():
        all_mins.append(float(np.min(arr)))
        all_maxs.append(float(np.max(arr)))

    all_mins.append(float(np.min(bergeron_profiles_kv)))
    all_maxs.append(float(np.max(bergeron_profiles_kv)))

    y_min = min(all_mins)
    y_max = max(all_maxs)
    y_span = y_max - y_min
    y_margin = float(anim_cfg["y_margin"]) * max(y_span, 1.0)

    y_min = y_min - y_margin
    y_max = y_max + y_margin

    fig, ax = plt.subplots(figsize=anim_cfg["figsize"], dpi=int(anim_cfg["dpi"]))

    line_n1, = ax.plot([], [], lw=2, label="Pi N=1")
    line_n10, = ax.plot([], [], lw=2, label="Pi N=10")
    line_n100, = ax.plot([], [], lw=2, label="Pi N=100")
    line_bergeron, = ax.plot([], [], lw=2, linestyle="--", label="Bergeron")

    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, va="top")

    ax.set_xlim(x_km[0], x_km[-1])
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Posição ao longo da linha [km]")
    ax.set_ylabel("Tensão [kV]")
    ax.set_title("Perfil de Tensão ao longo da linha: Modelos $\Pi$ vs Bergeron")
    ax.grid(True)
    ax.legend(loc="upper right")

    frames = []

    for frame_idx in tqdm(range(len(frame_times_ms)), desc="Rendering GIF"):
        line_n1.set_data(x_km, pi_profiles_by_n_kv[1][:, frame_idx])
        line_n10.set_data(x_km, pi_profiles_by_n_kv[10][:, frame_idx])
        line_n100.set_data(x_km, pi_profiles_by_n_kv[100][:, frame_idx])
        line_bergeron.set_data(x_km, bergeron_profiles_kv[:, frame_idx])

        time_text.set_text(f"t = {frame_times_ms[frame_idx]:.3f} ms")

        frames.append(figure_to_image(fig).copy())

    plt.close(fig)

    if len(frames) == 0:
        raise RuntimeError("No frames were generated for the GIF.")

    fps = int(anim_cfg["fps"])
    duration_ms = int(round(1000.0 / fps))

    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )

    return [str(gif_path)]


# =========================
# Output directories
# =========================
def ensure_output_dirs(base_dir: Path) -> dict:
    figures_dir = base_dir / "outputs" / "figures"
    animations_dir = base_dir / "outputs" / "animations"
    figures_dir.mkdir(parents=True, exist_ok=True)
    animations_dir.mkdir(parents=True, exist_ok=True)
    return {"figures": figures_dir, "animations": animations_dir}


# =========================
# Main
# =========================
def main() -> None:
    base_dir = Path(__file__).resolve().parent
    dirs = ensure_output_dirs(base_dir)

    n_values = DEFAULT_CONFIG["simulation"]["n_values"]
    required_n = [1, 10, 100]

    if sorted(n_values) != sorted(required_n):
        raise ValueError("Set simulation.n_values exactly to [1, 20, 200].")

    results_by_n = {}

    for n_sections in tqdm(n_values, desc="Pi simulations"):
        results_by_n[n_sections] = simulate_case(n_sections, DEFAULT_CONFIG)

    time_reference = results_by_n[1]["t"]
    frame_stride = max(1, int(DEFAULT_CONFIG["animation"]["frame_stride"]))
    frame_indices = np.arange(0, len(time_reference), frame_stride)

    frame_times = time_reference[frame_indices]
    frame_times_ms = frame_times * 1.0e3

    spatial_points = int(DEFAULT_CONFIG["animation"]["spatial_points"])
    x_query_m = np.linspace(0.0, float(DEFAULT_CONFIG["line"]["length_m"]), spatial_points)
    x_query_km = x_query_m / 1000.0

    pi_profiles_by_n = {}
    for n_value in n_values:
        pi_profiles_by_n[n_value] = reconstruct_pi_profiles_for_frames(
            results_by_n[n_value],
            x_query_m,
            frame_indices,
        )

    bergeron_profiles = build_bergeron_profiles(frame_times, x_query_m, DEFAULT_CONFIG)

    pi_profiles_by_n_kv = {}
    for n_value in n_values:
        pi_profiles_by_n_kv[n_value] = pi_profiles_by_n[n_value] / 1.0e3

    bergeron_profiles_kv = bergeron_profiles / 1.0e3

    gif_path = dirs["animations"] / DEFAULT_CONFIG["animation"]["gif_name"]

    saved_files = save_gif_from_profiles(
        x_query_km,
        frame_times_ms,
        pi_profiles_by_n_kv,
        bergeron_profiles_kv,
        gif_path,
        DEFAULT_CONFIG,
    )

    print("\nGenerated files:")
    for path in sorted((base_dir / "outputs").rglob("*")):
        if path.is_file():
            print(path)

    if len(saved_files) > 0:
        print("\nAnimation files saved:")
        for path in saved_files:
            print(path)


if __name__ == "__main__":
    main()