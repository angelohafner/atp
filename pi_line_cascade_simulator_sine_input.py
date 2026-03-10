import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image


# =========================
# User-editable parameters
# =========================
DEFAULT_CONFIG = {
    "line": {
        "length_m": 30_000.0,
        "r_per_m_ohm": 8.0e-4,
        "l_per_m_h": 1.0e-6,
        "c_per_m_f": 10.0e-12,
    },
    "source": {
        "kind": "sin",  # Options: "step", "dc", "ramp", "sin"
        "amplitude_v": 10_000.0,
        "frequency_hz": 50.0e3,
        "t_start_s": 0.0,
        "rise_time_s": 8.0e-6,
        "dc_offset_v": 0.0,
        "resistance_ohm": 316.23,
    },
    "load": {
        "kind": "resistive",  # Options: "open", "resistive"
        "resistance_ohm_values": [10.0 * 316.23, 0.1 * 316.23, 1.0 * 316.23],
    },
    "simulation": {
        "time_end_s": 0.5e-3,
        "dt_s": 1.0e-7,
    },
    "animation": {
        "fps": 10,
        "frame_stride": 10,
        "spatial_points": 100,
        "figsize": (12, 6),
        "dpi": 50,
        "y_min_kv": -10.0,
        "y_max_kv": 10.0,
        "max_bounces": 40,
        "gif_name": "comparison_bergeron_loads.gif",
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
# Bergeron helpers
# =========================
def reflection_coefficient(z_term: float, z0: float) -> float:
    if np.isinf(z_term):
        return 1.0

    denom = z_term + z0
    if abs(denom) < 1.0e-12:
        return -1.0

    return (z_term - z0) / denom


def build_bergeron_profiles(
    frame_times: np.ndarray,
    x_positions_m: np.ndarray,
    config: dict,
    load_resistance_ohm: float,
) -> np.ndarray:
    line_cfg = config["line"]
    source_cfg = config["source"]
    anim_cfg = config["animation"]

    length_m = float(line_cfg["length_m"])
    l_per_m = float(line_cfg["l_per_m_h"])
    c_per_m = float(line_cfg["c_per_m_f"])

    z0 = math.sqrt(l_per_m / c_per_m)
    velocity = 1.0 / math.sqrt(l_per_m * c_per_m)

    z_source = max(float(source_cfg.get("resistance_ohm", 0.0)), 1.0e-12)
    z_load = max(float(load_resistance_ohm), 1.0e-12)

    gamma_s = reflection_coefficient(z_source, z0)
    gamma_l = reflection_coefficient(z_load, z0)

    launch_coeff = z0 / (z0 + z_source)
    max_bounces = int(anim_cfg["max_bounces"])

    t_grid = frame_times[np.newaxis, :]
    x_grid = x_positions_m[:, np.newaxis]

    v_total = np.zeros((len(x_positions_m), len(frame_times)), dtype=float)

    for m_idx in tqdm(range(max_bounces), desc=f"Bergeron RL={z_load:.2f} ohm"):
        amp_forward = launch_coeff * (gamma_s * gamma_l) ** m_idx
        time_forward = t_grid - (2.0 * m_idx * length_m + x_grid) / velocity
        v_total = v_total + amp_forward * source_voltage_array(time_forward, source_cfg)

        amp_backward = launch_coeff * gamma_l * (gamma_s * gamma_l) ** m_idx
        time_backward = t_grid - ((2.0 * m_idx + 2.0) * length_m - x_grid) / velocity
        v_total = v_total + amp_backward * source_voltage_array(time_backward, source_cfg)

    return v_total


# =========================
# GIF export
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
    bergeron_profiles_by_label_kv: dict,
    gif_path: Path,
    config: dict,
) -> list:
    anim_cfg = config["animation"]

    fig, ax = plt.subplots(figsize=anim_cfg["figsize"], dpi=int(anim_cfg["dpi"]))

    lines = {}
    for label in bergeron_profiles_by_label_kv:
        line_obj, = ax.plot([], [], lw=4, label=label, alpha=0.5)
        lines[label] = line_obj

    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, va="top")

    ax.set_xlim(x_km[0], x_km[-1])
    ax.set_ylim(float(anim_cfg["y_min_kv"]), float(anim_cfg["y_max_kv"]))
    ax.set_xlabel("Posicao ao longo da linha [km]")
    ax.set_ylabel("Tensao [kV]")
    ax.set_title("Perfil de Tensao ao longo da linha: Bergeron")
    ax.grid(True)
    #ax.legend(loc="upper right")

    frames = []

    for frame_idx in tqdm(range(len(frame_times_ms)), desc="Rendering GIF"):
        for label, line_obj in lines.items():
            line_obj.set_data(x_km, bergeron_profiles_by_label_kv[label][:, frame_idx])

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

    return {
        "figures": figures_dir,
        "animations": animations_dir,
    }


# =========================
# Main
# =========================
def main() -> None:
    base_dir = Path(__file__).resolve().parent
    dirs = ensure_output_dirs(base_dir)

    sim_cfg = DEFAULT_CONFIG["simulation"]
    anim_cfg = DEFAULT_CONFIG["animation"]
    load_cfg = DEFAULT_CONFIG["load"]

    dt = float(sim_cfg["dt_s"])
    time_end = float(sim_cfg["time_end_s"])

    n_steps = int(np.floor(time_end / dt))
    time_reference = np.arange(n_steps + 1, dtype=float) * dt

    if time_reference[-1] < time_end:
        time_reference = np.append(time_reference, time_end)
    else:
        time_reference[-1] = time_end

    frame_stride = max(1, int(anim_cfg["frame_stride"]))
    frame_indices = np.arange(0, len(time_reference), frame_stride)

    frame_times = time_reference[frame_indices]
    frame_times_ms = frame_times * 1.0e3

    spatial_points = int(anim_cfg["spatial_points"])
    x_query_m = np.linspace(0.0, float(DEFAULT_CONFIG["line"]["length_m"]), spatial_points)
    x_query_km = x_query_m / 1000.0

    z0 = math.sqrt(
        float(DEFAULT_CONFIG["line"]["l_per_m_h"]) /
        float(DEFAULT_CONFIG["line"]["c_per_m_f"])
    )

    bergeron_profiles_by_label = {}

    for load_resistance in load_cfg["resistance_ohm_values"]:
        label = f"Bergeron RL={load_resistance / z0:.1f}*Z0"
        bergeron_profiles_by_label[label] = build_bergeron_profiles(
            frame_times,
            x_query_m,
            DEFAULT_CONFIG,
            load_resistance,
        )

    bergeron_profiles_by_label_kv = {}
    for label in bergeron_profiles_by_label:
        bergeron_profiles_by_label_kv[label] = bergeron_profiles_by_label[label] / 1.0e3

    gif_path = dirs["animations"] / anim_cfg["gif_name"]

    saved_files = save_gif_from_profiles(
        x_query_km,
        frame_times_ms,
        bergeron_profiles_by_label_kv,
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