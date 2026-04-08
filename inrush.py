import numpy as np
from scipy.integrate import solve_ivp
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from PIL import Image, ImageDraw


# ============================================================
# Simulation parameters
# ============================================================

FREQUENCY_HZ = 60.0
OMEGA_RAD_PER_S = 2.0 * np.pi * FREQUENCY_HZ
CYCLE_PERIOD_S = 1.0 / FREQUENCY_HZ

V_RMS = 200.0
V_PEAK = np.sqrt(2.0) * V_RMS

L_REF = 0.5

# For a transformer-like inrush, winding resistance should be
# much smaller than the reactance omega*L_ref so the offset decays slowly.
R_SERIES_FACTOR = 0.005
R_SERIES = R_SERIES_FACTOR * OMEGA_RAD_PER_S * L_REF

SIMULATION_TIME_S = 0.5
SWITCH_CLOSING_TIME_S = 0.020
ENERGIZATION_ANGLE_DEG = 0.0
LAMBDA_0 = 0.0

RTOL = 1.0e-8
ATOL = 1.0e-10
POINTS_PER_CYCLE = 128

ANIMATION_DURATION_S = 4.0
GIF_FPS = 25
OUTPUT_DIR = Path(__file__).resolve().parent / "animations"
ANIMATION_FILE = OUTPUT_DIR / "source_voltage_current_animation.gif"
GIF_FRAME_WIDTH = 1400
GIF_FRAME_HEIGHT = 800

# Nonlinear magnetic model parameters
# i(lambda) = a1*lambda + a3*lambda^3 + a5*lambda^5
A1 = 1.0 / L_REF
A3 = 2.8
A5 = 6.0


# ============================================================
# Magnetic model functions
# ============================================================

def magnetizing_current_from_lambda(lambda_wb):
    """
    Return magnetizing current from flux linkage using
    a smooth odd polynomial.
    """
    current_a = A1 * lambda_wb + A3 * (lambda_wb ** 3) + A5 * (lambda_wb ** 5)
    return current_a


def magnetic_model_description():
    text = (
        "Magnetic model: i(lambda) = a1*lambda + a3*lambda^3 + a5*lambda^5\n"
        f"a1 = {A1:.6f} A/Wb-turn\n"
        f"a3 = {A3:.6f} A/(Wb-turn)^3\n"
        f"a5 = {A5:.6f} A/(Wb-turn)^5\n"
        "Low-flux incremental inductance is approximately L_ref.\n"
        "At higher flux linkage, current grows much faster, representing core saturation."
    )
    return text


# ============================================================
# Source and switching functions
# ============================================================

def angle_deg_to_rad(angle_deg):
    angle_rad = angle_deg * np.pi / 180.0
    return angle_rad


def source_voltage_after_closing(time_s):
    """
    Source voltage waveform after the switch closes.
    """
    phi_rad = angle_deg_to_rad(ENERGIZATION_ANGLE_DEG)
    voltage_v = V_PEAK * np.sin(OMEGA_RAD_PER_S * (time_s - SWITCH_CLOSING_TIME_S) + phi_rad)
    return voltage_v


def applied_voltage(time_s):
    """
    Applied voltage to the branch.
    Before closing: zero.
    After closing: sinusoidal source.
    """
    if np.isscalar(time_s):
        if time_s < SWITCH_CLOSING_TIME_S:
            return 0.0
        return source_voltage_after_closing(time_s)

    voltage_v = np.zeros_like(time_s)
    mask = time_s >= SWITCH_CLOSING_TIME_S
    voltage_v[mask] = source_voltage_after_closing(time_s[mask])
    return voltage_v


# ============================================================
# Differential equation
# ============================================================

def lambda_ode(time_s, state):
    """
    State variable:
        state[0] = flux linkage lambda

    Equation:
        v_source(t) = R_series*i_m(lambda) + d(lambda)/dt
        d(lambda)/dt = v_source(t) - R_series*i_m(lambda)
    """
    lambda_wb = state[0]
    i_m_a = magnetizing_current_from_lambda(lambda_wb)
    if time_s < SWITCH_CLOSING_TIME_S:
        d_lambda_dt = 0.0
    else:
        d_lambda_dt = applied_voltage(time_s) - R_SERIES * i_m_a
    return np.array([d_lambda_dt], dtype=float)


# ============================================================
# Simulation execution
# ============================================================

def build_time_vector():
    cycles = SIMULATION_TIME_S * FREQUENCY_HZ
    total_points = int(np.ceil(cycles * POINTS_PER_CYCLE)) + 1
    time_vector_s = np.linspace(0.0, SIMULATION_TIME_S, total_points)
    return time_vector_s


def run_simulation():
    time_eval_s = build_time_vector()
    initial_state = np.array([LAMBDA_0], dtype=float)

    solution = solve_ivp(
        fun=lambda_ode,
        t_span=(0.0, SIMULATION_TIME_S),
        y0=initial_state,
        t_eval=time_eval_s,
        method="RK45",
        rtol=RTOL,
        atol=ATOL,
    )

    if not solution.success:
        raise RuntimeError(f"Simulation failed: {solution.message}")

    time_s = solution.t
    lambda_wb = solution.y[0]

    source_voltage_v = applied_voltage(time_s)
    i_m_a = magnetizing_current_from_lambda(lambda_wb)
    source_current_a = np.zeros_like(time_s)
    mask_closed = time_s >= SWITCH_CLOSING_TIME_S
    source_current_a[mask_closed] = i_m_a[mask_closed]
    v_r_series_v = R_SERIES * i_m_a
    v_inductor_v = source_voltage_v - v_r_series_v

    results = {
        "time_s": time_s,
        "lambda_wb": lambda_wb,
        "source_voltage_v": source_voltage_v,
        "source_current_a": source_current_a,
        "v_r_series_v": v_r_series_v,
        "v_inductor_v": v_inductor_v,
        "i_m_a": i_m_a,
    }
    return results


# ============================================================
# Post-processing
# ============================================================

def find_peak(signal, time_s):
    abs_signal = np.abs(signal)
    idx = int(np.argmax(abs_signal))
    peak_data = {
        "index": idx,
        "time_s": time_s[idx],
        "value": signal[idx],
        "abs_value": abs_signal[idx],
    }
    return peak_data


def compute_cycle_mean(time_s, signal):
    cycle_start_s = np.arange(0.0, SIMULATION_TIME_S, CYCLE_PERIOD_S)
    cycle_center_s = []
    cycle_mean = []

    for start_s in cycle_start_s:
        end_s = min(start_s + CYCLE_PERIOD_S, SIMULATION_TIME_S)
        if end_s <= start_s:
            continue

        if np.isclose(end_s, SIMULATION_TIME_S):
            mask = (time_s >= start_s) & (time_s <= end_s)
        else:
            mask = (time_s >= start_s) & (time_s < end_s)

        if not np.any(mask):
            continue

        cycle_center_s.append(0.5 * (start_s + end_s))
        cycle_mean.append(np.mean(signal[mask]))

    return np.array(cycle_center_s), np.array(cycle_mean)


def print_case_report(results):
    time_s = results["time_s"]
    i_m_a = results["i_m_a"]
    lambda_wb = results["lambda_wb"]

    peak_im = find_peak(i_m_a, time_s)
    cycle_mean_time_s, cycle_mean_lambda_wb = compute_cycle_mean(time_s, lambda_wb)

    print("\n============================================================")
    print(f"R_series = {R_SERIES:.6f} ohm")
    print(f"Peak |i_m|     = {peak_im['abs_value']:.6f} A at t = {peak_im['time_s']:.6f} s")
    if cycle_mean_lambda_wb.size > 0:
        print(f"Cycle-mean lambda at t = {cycle_mean_time_s[0]:.6f} s: {cycle_mean_lambda_wb[0]:.6f} Wb-turn")
        print(f"Cycle-mean lambda at t = {cycle_mean_time_s[-1]:.6f} s: {cycle_mean_lambda_wb[-1]:.6f} Wb-turn")


def print_global_info():
    print("=== Nonlinear Inductor Inrush Simulation with Series Resistance ===")
    print(f"Frequency: {FREQUENCY_HZ:.3f} Hz")
    print(f"V_rms: {V_RMS:.3f} V")
    print(f"V_peak: {V_PEAK:.3f} V")
    print(f"L_ref: {L_REF:.6f} H")
    print(f"R_series: {R_SERIES:.6f} ohm")
    print(f"R_series / (omega*L_ref): {R_SERIES_FACTOR:.6f}")
    print(f"Switch closing time: {SWITCH_CLOSING_TIME_S:.6f} s")
    print(f"Energization angle: {ENERGIZATION_ANGLE_DEG:.3f} deg")
    print(f"Initial flux linkage lambda_0: {LAMBDA_0:.6f} Wb-turn")
    print()
    print(magnetic_model_description())
    print()
    print("Note:")
    print("- Flux linkage lambda(t) now depends on both the source voltage and the voltage drop across R_series.")
    print("- The series resistance damps the transient and makes the cycle-mean value of lambda(t) decay over time.")
    print("- In this model, the source current is the same as the magnetizing current.")


# ============================================================
# Plot helpers
# ============================================================

def create_figure(title, xaxis_title, yaxis_title):
    fig = go.Figure()
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


def add_switch_closing_marker(fig):
    fig.add_vline(
        x=SWITCH_CLOSING_TIME_S,
        line_dash="dash",
        line_color="gray",
        annotation_text="Switch closing",
        annotation_position="top right",
    )


def symmetric_axis_range(signal, padding_factor=1.05):
    max_abs_value = float(np.max(np.abs(signal)))
    if max_abs_value == 0.0:
        max_abs_value = 1.0
    limit = padding_factor * max_abs_value
    return [-limit, limit]


def full_time_axis_range(time_s):
    time_start = float(time_s[0])
    time_end = float(time_s[-1])
    if np.isclose(time_start, time_end):
        time_end = time_start + 1.0
    return [time_start, time_end]


def map_to_pixel(value, axis_min, axis_max, pixel_min, pixel_max):
    if np.isclose(axis_min, axis_max):
        return 0.5 * (pixel_min + pixel_max)
    normalized = (value - axis_min) / (axis_max - axis_min)
    return pixel_min + normalized * (pixel_max - pixel_min)


def build_polyline_points(x_values, y_values, x_range, y_range, plot_bounds):
    left_px, top_px, right_px, bottom_px = plot_bounds
    points = []
    for x_value, y_value in zip(x_values, y_values):
        x_px = map_to_pixel(x_value, x_range[0], x_range[1], left_px, right_px)
        y_px = map_to_pixel(y_value, y_range[0], y_range[1], bottom_px, top_px)
        points.append((x_px, y_px))
    return points


def draw_static_plot_elements(draw, time_s, source_voltage_v, source_current_a, frame_size):
    width_px, height_px = frame_size
    margin_left_px = 120
    margin_right_px = 120
    margin_top_px = 80
    margin_bottom_px = 100
    plot_bounds = (
        margin_left_px,
        margin_top_px,
        width_px - margin_right_px,
        height_px - margin_bottom_px,
    )
    x_range = full_time_axis_range(time_s)
    voltage_range = symmetric_axis_range(source_voltage_v)
    current_range = symmetric_axis_range(source_current_a)

    draw.rectangle([(0, 0), (width_px - 1, height_px - 1)], fill="white")
    draw.text((margin_left_px, 24), "Source Voltage and Source Current", fill="black")

    left_px, top_px, right_px, bottom_px = plot_bounds
    draw.rectangle([(left_px, top_px), (right_px, bottom_px)], outline="black", width=2)

    zero_voltage_y = map_to_pixel(0.0, voltage_range[0], voltage_range[1], bottom_px, top_px)
    draw.line([(left_px, zero_voltage_y), (right_px, zero_voltage_y)], fill=(170, 170, 170), width=1)

    switch_x = map_to_pixel(SWITCH_CLOSING_TIME_S, x_range[0], x_range[1], left_px, right_px)
    draw.line([(switch_x, top_px), (switch_x, bottom_px)], fill=(120, 120, 120), width=2)
    draw.text((switch_x + 8, top_px + 8), "Switch closing", fill=(80, 80, 80))

    tick_count = 6
    for tick_idx in range(tick_count + 1):
        frac = tick_idx / tick_count
        x_value = x_range[0] + frac * (x_range[1] - x_range[0])
        x_px = map_to_pixel(x_value, x_range[0], x_range[1], left_px, right_px)
        draw.line([(x_px, bottom_px), (x_px, bottom_px + 8)], fill="black", width=1)
        draw.text((x_px - 18, bottom_px + 14), f"{x_value:.2f}", fill="black")

        voltage_value = voltage_range[0] + frac * (voltage_range[1] - voltage_range[0])
        voltage_y = map_to_pixel(voltage_value, voltage_range[0], voltage_range[1], bottom_px, top_px)
        draw.line([(left_px - 8, voltage_y), (left_px, voltage_y)], fill="black", width=1)
        draw.text((16, voltage_y - 8), f"{voltage_value:.0f}", fill=(31, 119, 180))

        current_value = current_range[0] + frac * (current_range[1] - current_range[0])
        current_y = map_to_pixel(current_value, current_range[0], current_range[1], bottom_px, top_px)
        draw.line([(right_px, current_y), (right_px + 8, current_y)], fill="black", width=1)
        draw.text((right_px + 14, current_y - 8), f"{current_value:.1f}", fill=(214, 39, 40))

    draw.text((0.5 * (left_px + right_px) - 24, height_px - 40), "Time (s)", fill="black")
    draw.text((18, top_px - 30), "Voltage (V)", fill=(31, 119, 180))
    draw.text((right_px - 16, top_px - 30), "Current (A)", fill=(214, 39, 40))

    legend_y = top_px + 18
    draw.line([(left_px + 24, legend_y), (left_px + 84, legend_y)], fill=(31, 119, 180), width=4)
    draw.text((left_px + 94, legend_y - 10), "Source voltage v(t)", fill=(31, 119, 180))
    draw.line([(left_px + 270, legend_y), (left_px + 330, legend_y)], fill=(214, 39, 40), width=4)
    draw.text((left_px + 340, legend_y - 10), "Source current i(t)", fill=(214, 39, 40))

    return {
        "plot_bounds": plot_bounds,
        "x_range": x_range,
        "voltage_range": voltage_range,
        "current_range": current_range,
    }


def build_gif_frame(time_s, source_voltage_v, source_current_a, point_count):
    frame_size = (GIF_FRAME_WIDTH, GIF_FRAME_HEIGHT)
    image = Image.new("RGB", frame_size, "white")
    draw = ImageDraw.Draw(image)
    layout = draw_static_plot_elements(draw, time_s, source_voltage_v, source_current_a, frame_size)

    voltage_points = build_polyline_points(
        time_s[:point_count],
        source_voltage_v[:point_count],
        layout["x_range"],
        layout["voltage_range"],
        layout["plot_bounds"],
    )
    current_points = build_polyline_points(
        time_s[:point_count],
        source_current_a[:point_count],
        layout["x_range"],
        layout["current_range"],
        layout["plot_bounds"],
    )

    if len(voltage_points) >= 2:
        draw.line(voltage_points, fill=(31, 119, 180), width=4)
    if len(current_points) >= 2:
        draw.line(current_points, fill=(214, 39, 40), width=4)

    return image.convert("P", palette=Image.ADAPTIVE)


def build_source_voltage_current_figure(results):
    time_s = results["time_s"]
    source_voltage_v = results["source_voltage_v"]
    source_current_a = results["source_current_a"]
    initial_point_count = 1

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scattergl(
            x=time_s[:initial_point_count],
            y=source_voltage_v[:initial_point_count],
            mode="lines",
            name="Source voltage v(t)",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scattergl(
            x=time_s[:initial_point_count],
            y=source_current_a[:initial_point_count],
            mode="lines",
            name="Source current i(t)",
        ),
        secondary_y=True,
    )
    fig.update_layout(
        title="Source Voltage and Source Current",
        xaxis_title="Time (s)",
        template="plotly_white",
        hovermode="x unified",
        xaxis={"range": full_time_axis_range(time_s), "autorange": False},
    )
    fig.update_yaxes(
        title_text="Voltage (V)",
        range=symmetric_axis_range(source_voltage_v),
        autorange=False,
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text="Current (A)",
        range=symmetric_axis_range(source_current_a),
        autorange=False,
        secondary_y=True,
    )
    add_switch_closing_marker(fig)
    return fig


# ============================================================
# Plots
# ============================================================

def plot_source_voltage_and_current(results):
    fig = build_source_voltage_current_figure(results)
    time_s = results["time_s"]
    fig.data[0].x = time_s
    fig.data[0].y = results["source_voltage_v"]
    fig.data[1].x = time_s
    fig.data[1].y = results["source_current_a"]
    fig.show()


def make_all_plots(results):
    plot_source_voltage_and_current(results)


def save_animated_source_voltage_current(results):
    time_s = results["time_s"]
    source_voltage_v = results["source_voltage_v"]
    source_current_a = results["source_current_a"]
    OUTPUT_DIR.mkdir(exist_ok=True)
    frame_count = max(2, int(round(ANIMATION_DURATION_S * GIF_FPS)))
    frame_indices = np.unique(np.linspace(1, time_s.size, num=frame_count, dtype=int))
    frame_duration_ms = int(round(1000.0 / GIF_FPS))

    frames = []
    for idx in frame_indices:
        frames.append(build_gif_frame(time_s, source_voltage_v, source_current_a, idx))

    frames[0].save(
        ANIMATION_FILE,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration_ms,
        loop=0,
        disposal=2,
    )
    print(f"Animated figure saved to: {ANIMATION_FILE}")
    return ANIMATION_FILE


# ============================================================
# Main execution
# ============================================================

if __name__ == "__main__":
    print_global_info()

    results_dict = run_simulation()
    print_case_report(results_dict)
    make_all_plots(results_dict)
    save_animated_source_voltage_current(results_dict)
