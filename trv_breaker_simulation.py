"""
Transient recovery voltage (TRV) simulation for a simple source-R-L-breaker-fault circuit.

The breaker starts open, closes at a prescribed instant, remains closed for at
least six 60 Hz cycles, and then opens at the first current zero crossing found
after that minimum closed interval.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class CircuitParameters:
    """Electrical parameters consistent with the circuit in the reference figure."""

    frequency_hz: float = 60.0
    source_rms_v: float = 7.2e3
    resistance_ohm: float = 0.1
    inductance_h: float = 1.5e-3
    bus_capacitance_f: float = 0.1e-6
    source_phase_rad: float = 0.0

    @property
    def omega(self) -> float:
        return 2.0 * np.pi * self.frequency_hz

    @property
    def source_peak_v(self) -> float:
        return np.sqrt(2.0) * self.source_rms_v

    @property
    def cycle_s(self) -> float:
        return 1.0 / self.frequency_hz


@dataclass(frozen=True)
class BreakerTiming:
    """Breaker operation timing."""

    close_time_s: float = 0.05
    closed_cycles_before_open: float = 6.0
    post_open_time_s: float = 0.100


@dataclass(frozen=True)
class SimulationSettings:
    """Numerical settings for integration and plotting."""

    time_step_s: float = 1.0e-6
    rtol: float = 1.0e-8
    atol_current: float = 1.0e-6
    atol_voltage: float = 1.0e-3
    animation_duration_s: float = 3.0
    animation_fps: int = 30
    animation_max_points: int = 4000


def source_voltage(t: float | np.ndarray, circuit: CircuitParameters) -> float | np.ndarray:
    """Return the instantaneous sinusoidal source voltage."""
    return circuit.source_peak_v * np.sin(circuit.omega * t + circuit.source_phase_rad)


def open_breaker_ode(t: float, y: np.ndarray, circuit: CircuitParameters) -> list[float]:
    """R-L-C dynamics when the breaker is open and the bus capacitance is energized."""
    inductor_current_a, capacitor_voltage_v = y
    di_dt = (
        source_voltage(t, circuit)
        - circuit.resistance_ohm * inductor_current_a
        - capacitor_voltage_v
    ) / circuit.inductance_h
    dv_dt = inductor_current_a / circuit.bus_capacitance_f
    return [di_dt, dv_dt]


def closed_breaker_ode(t: float, y: np.ndarray, circuit: CircuitParameters) -> list[float]:
    """R-L fault current dynamics when the breaker is closed."""
    (breaker_current_a,) = y
    di_dt = (
        source_voltage(t, circuit) - circuit.resistance_ohm * breaker_current_a
    ) / circuit.inductance_h
    return [di_dt]


def make_time_grid(t_start: float, t_stop: float, step: float) -> np.ndarray:
    """Create a stable time vector that always includes the segment end point."""
    if t_stop < t_start:
        raise ValueError("t_stop must be greater than or equal to t_start")
    if np.isclose(t_start, t_stop):
        return np.array([t_start])

    samples = int(np.floor((t_stop - t_start) / step)) + 1
    grid = t_start + np.arange(samples) * step
    if grid[-1] < t_stop:
        grid = np.append(grid, t_stop)
    else:
        grid[-1] = t_stop
    return grid


def solve_open_segment(
    t_start: float,
    t_stop: float,
    initial_state: np.ndarray,
    circuit: CircuitParameters,
    settings: SimulationSettings,
) -> solve_ivp:
    """Integrate an open-breaker segment."""
    return solve_ivp(
        fun=lambda t, y: open_breaker_ode(t, y, circuit),
        t_span=(t_start, t_stop),
        y0=initial_state,
        method="RK45",
        dense_output=True,
        max_step=settings.time_step_s,
        rtol=settings.rtol,
        atol=[settings.atol_current, settings.atol_voltage],
    )


def solve_closed_segment(
    t_start: float,
    t_stop: float,
    initial_current_a: float,
    circuit: CircuitParameters,
    settings: SimulationSettings,
    zero_crossing_event=None,
) -> solve_ivp:
    """Integrate a closed-breaker segment, optionally stopping at a current zero."""
    events = None if zero_crossing_event is None else zero_crossing_event
    return solve_ivp(
        fun=lambda t, y: closed_breaker_ode(t, y, circuit),
        t_span=(t_start, t_stop),
        y0=[initial_current_a],
        method="RK45",
        dense_output=True,
        max_step=settings.time_step_s,
        rtol=settings.rtol,
        atol=[settings.atol_current],
        events=events,
    )


def find_breaker_opening(
    t_start_search_s: float,
    initial_current_a: float,
    circuit: CircuitParameters,
    settings: SimulationSettings,
) -> solve_ivp:
    """Find the first current zero crossing after the mandatory closed interval."""

    def current_zero_crossing(_t: float, y: np.ndarray) -> float:
        return y[0]

    current_zero_crossing.terminal = True
    current_zero_crossing.direction = 0.0

    search_stop_s = t_start_search_s + 2.0 * circuit.cycle_s
    solution = solve_closed_segment(
        t_start=t_start_search_s,
        t_stop=search_stop_s,
        initial_current_a=initial_current_a,
        circuit=circuit,
        settings=settings,
        zero_crossing_event=current_zero_crossing,
    )

    if solution.t_events[0].size == 0:
        raise RuntimeError("No current zero crossing was detected during the search window.")

    return solution


def symmetric_axis_limit(signal: np.ndarray, margin: float = 1.05, fallback: float = 1.0) -> float:
    """Return a nonzero symmetric axis limit for a signal."""
    limit = margin * float(np.max(np.abs(signal)))
    if not np.isfinite(limit) or limit <= 0.0:
        return fallback
    return limit


def build_voltage_current_figure(
    time_ms: np.ndarray,
    breaker_voltage_kv: np.ndarray,
    breaker_current_ka: np.ndarray,
    close_time_ms: float,
    open_time_ms: float,
) -> go.Figure:
    """Build one figure with voltage on the primary axis and current on the secondary axis."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=time_ms,
            y=breaker_voltage_kv,
            mode="lines",
            line=dict(width=1.8, color="#1f77b4"),
            name="Tensao no disjuntor [kV]",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=time_ms,
            y=breaker_current_ka,
            mode="lines",
            line=dict(width=5, color="rgba(214, 39, 40, 0.5)"),
            name="Corrente no disjuntor [kA]",
        ),
        secondary_y=True,
    )

    fig.add_vline(
        x=close_time_ms,
        line_width=1.5,
        line_dash="dash",
        line_color="green",
        annotation_text="Fechamento",
    )
    fig.add_vline(
        x=open_time_ms,
        line_width=1.5,
        line_dash="dash",
        line_color="red",
        annotation_text="Abertura",
    )
    fig.update_layout(
        title="Tensao e corrente no disjuntor",
        xaxis_title="Tempo [ms]",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
    )

    voltage_limit_kv = symmetric_axis_limit(breaker_voltage_kv)
    current_limit_ka = symmetric_axis_limit(breaker_current_ka)

    # Symmetric axis ranges force both zero lines to the same vertical position.
    fig.update_yaxes(
        title_text="Tensao [kV]",
        range=[-voltage_limit_kv, voltage_limit_kv],
        zeroline=True,
        zerolinewidth=1.2,
        zerolinecolor="black",
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text="Corrente [kA]",
        range=[-current_limit_ka, current_limit_ka],
        zeroline=True,
        zerolinewidth=1.2,
        zerolinecolor="black",
        secondary_y=True,
    )
    return fig


def downsample_for_animation(
    time_ms: np.ndarray,
    breaker_voltage_kv: np.ndarray,
    breaker_current_ka: np.ndarray,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reduce the number of plotted samples while preserving the full time span."""
    if time_ms.size <= max_points:
        return time_ms, breaker_voltage_kv, breaker_current_ka

    indexes = np.unique(np.linspace(0, time_ms.size - 1, max_points, dtype=int))
    return time_ms[indexes], breaker_voltage_kv[indexes], breaker_current_ka[indexes]


def write_voltage_current_animation(
    time_ms: np.ndarray,
    breaker_voltage_kv: np.ndarray,
    breaker_current_ka: np.ndarray,
    close_time_ms: float,
    open_time_ms: float,
    output_path: Path,
    settings: SimulationSettings,
) -> None:
    """Write a GIF that progressively draws voltage and current curves."""
    time_anim_ms, voltage_anim_kv, current_anim_ka = downsample_for_animation(
        time_ms=time_ms,
        breaker_voltage_kv=breaker_voltage_kv,
        breaker_current_ka=breaker_current_ka,
        max_points=settings.animation_max_points,
    )

    # GIF frame delays are quantized in centiseconds, so compute the frame count
    # from the effective delay to keep the final duration close to the target.
    gif_frame_delay_ms = max(
        10.0,
        round((1000.0 / settings.animation_fps) / 10.0) * 10.0,
    )
    requested_frames = max(
        2,
        int(round(settings.animation_duration_s * 1000.0 / gif_frame_delay_ms)),
    )
    frame_count = min(requested_frames, time_anim_ms.size)
    frame_indexes = np.unique(np.linspace(1, time_anim_ms.size, frame_count, dtype=int))
    actual_fps = max(1, int(round(1000.0 / gif_frame_delay_ms)))

    voltage_limit_kv = symmetric_axis_limit(breaker_voltage_kv)
    current_limit_ka = symmetric_axis_limit(breaker_current_ka)

    fig, voltage_axis = plt.subplots(figsize=(11.0, 6.0), dpi=100)
    current_axis = voltage_axis.twinx()

    voltage_line, = voltage_axis.plot(
        [],
        [],
        color="#1f77b4",
        linewidth=1.8,
        label="Tensao no disjuntor [kV]",
    )
    current_line, = current_axis.plot(
        [],
        [],
        color="#d62728",
        linewidth=2.4,
        alpha=0.75,
        label="Corrente no disjuntor [kA]",
    )

    voltage_axis.set_xlim(float(time_ms[0]), float(time_ms[-1]))
    voltage_axis.set_ylim(-voltage_limit_kv, voltage_limit_kv)
    current_axis.set_ylim(-current_limit_ka, current_limit_ka)
    voltage_axis.set_xlabel("Tempo [ms]")
    voltage_axis.set_ylabel("Tensao [kV]", color="#1f77b4")
    current_axis.set_ylabel("Corrente [kA]", color="#d62728")
    voltage_axis.tick_params(axis="y", labelcolor="#1f77b4")
    current_axis.tick_params(axis="y", labelcolor="#d62728")
    voltage_axis.grid(True, alpha=0.25)

    # Symmetric y-limits keep both zero levels aligned throughout the animation.
    voltage_axis.axhline(0.0, color="black", linewidth=0.8)
    voltage_axis.axvline(close_time_ms, color="green", linestyle="--", linewidth=1.2)
    voltage_axis.axvline(open_time_ms, color="red", linestyle="--", linewidth=1.2)
    voltage_axis.set_title("Tensao e corrente no disjuntor")

    time_label = voltage_axis.text(
        0.02,
        0.94,
        "",
        transform=voltage_axis.transAxes,
        bbox=dict(facecolor="white", edgecolor="0.8", alpha=0.9),
    )

    lines = [voltage_line, current_line]
    labels = [line.get_label() for line in lines]
    voltage_axis.legend(lines, labels, loc="upper right")
    fig.tight_layout()

    def init_animation():
        voltage_line.set_data([], [])
        current_line.set_data([], [])
        time_label.set_text("")
        return voltage_line, current_line, time_label

    def update_animation(frame_end: int):
        voltage_line.set_data(time_anim_ms[:frame_end], voltage_anim_kv[:frame_end])
        current_line.set_data(time_anim_ms[:frame_end], current_anim_ka[:frame_end])
        time_label.set_text(f"t = {time_anim_ms[frame_end - 1]:.2f} ms")
        return voltage_line, current_line, time_label

    animation = FuncAnimation(
        fig,
        update_animation,
        frames=frame_indexes,
        init_func=init_animation,
        interval=1000.0 / actual_fps,
        blit=False,
        repeat=False,
    )
    animation.save(output_path, writer=PillowWriter(fps=actual_fps))
    plt.close(fig)


def simulate() -> dict[str, float]:
    """Run the complete breaker operation and TRV simulation."""
    circuit = CircuitParameters()
    timing = BreakerTiming()
    settings = SimulationSettings()

    minimum_open_time_s = (
        timing.close_time_s + timing.closed_cycles_before_open * circuit.cycle_s
    )

    # Segment 1: breaker open before the commanded close time.
    preclose_solution = solve_open_segment(
        t_start=0.0,
        t_stop=timing.close_time_s,
        initial_state=np.array([0.0, 0.0]),
        circuit=circuit,
        settings=settings,
    )
    preclose_current_a, _preclose_capacitor_voltage_v = preclose_solution.y[:, -1]

    # Ideal breaker closing clamps the breaker-side bus voltage to zero.
    closed_until_minimum_solution = solve_closed_segment(
        t_start=timing.close_time_s,
        t_stop=minimum_open_time_s,
        initial_current_a=preclose_current_a,
        circuit=circuit,
        settings=settings,
    )
    current_at_minimum_a = closed_until_minimum_solution.y[0, -1]

    # Segment 2b: continue closed until the first current zero crossing.
    closed_until_zero_solution = find_breaker_opening(
        t_start_search_s=minimum_open_time_s,
        initial_current_a=current_at_minimum_a,
        circuit=circuit,
        settings=settings,
    )
    open_time_s = float(closed_until_zero_solution.t_events[0][0])
    current_at_opening_a = float(closed_until_zero_solution.y_events[0][0][0])
    current_just_before_opening_a = float(
        closed_until_zero_solution.sol(open_time_s - min(1.0e-9, settings.time_step_s / 10.0))[0]
    )

    # Segment 3: breaker opens; the interrupted terminal voltage is the bus capacitor voltage.
    post_open_stop_s = open_time_s + timing.post_open_time_s
    post_open_solution = solve_open_segment(
        t_start=open_time_s,
        t_stop=post_open_stop_s,
        initial_state=np.array([current_at_opening_a, 0.0]),
        circuit=circuit,
        settings=settings,
    )

    preclose_time_s = make_time_grid(0.0, timing.close_time_s, settings.time_step_s)
    closed_min_time_s = make_time_grid(
        timing.close_time_s, minimum_open_time_s, settings.time_step_s
    )
    closed_zero_time_s = make_time_grid(
        minimum_open_time_s, open_time_s, settings.time_step_s
    )
    post_open_time_s = make_time_grid(open_time_s, post_open_stop_s, settings.time_step_s)

    preclose_state = preclose_solution.sol(preclose_time_s)
    closed_min_current_a = closed_until_minimum_solution.sol(closed_min_time_s)[0]
    closed_zero_current_a = closed_until_zero_solution.sol(closed_zero_time_s)[0]
    post_open_state = post_open_solution.sol(post_open_time_s)

    time_s = np.concatenate(
        [
            preclose_time_s[:-1],
            closed_min_time_s[:-1],
            closed_zero_time_s[:-1],
            post_open_time_s,
        ]
    )
    breaker_current_a = np.concatenate(
        [
            np.zeros_like(preclose_time_s[:-1]),
            closed_min_current_a[:-1],
            closed_zero_current_a[:-1],
            np.zeros_like(post_open_time_s),
        ]
    )
    breaker_voltage_v = np.concatenate(
        [
            preclose_state[1, :-1],
            np.zeros_like(closed_min_time_s[:-1]),
            np.zeros_like(closed_zero_time_s[:-1]),
            post_open_state[1],
        ]
    )

    trv_mask = time_s >= open_time_s
    trv_voltage_v = breaker_voltage_v[trv_mask]
    max_trv_v = float(np.max(np.abs(trv_voltage_v)))

    time_ms = time_s * 1.0e3
    close_time_ms = timing.close_time_s * 1.0e3
    open_time_ms = open_time_s * 1.0e3
    breaker_current_ka = breaker_current_a / 1.0e3
    breaker_voltage_kv = breaker_voltage_v / 1.0e3

    voltage_current_fig = build_voltage_current_figure(
        time_ms=time_ms,
        breaker_voltage_kv=breaker_voltage_kv,
        breaker_current_ka=breaker_current_ka,
        close_time_ms=close_time_ms,
        open_time_ms=open_time_ms,
    )

    output_dir = Path.cwd()
    static_html_path = output_dir / "tensao_corrente.html"
    animation_gif_path = output_dir / "tensao_corrente_animado.gif"

    voltage_current_fig.write_html(static_html_path, include_plotlyjs=True)
    write_voltage_current_animation(
        time_ms=time_ms,
        breaker_voltage_kv=breaker_voltage_kv,
        breaker_current_ka=breaker_current_ka,
        close_time_ms=close_time_ms,
        open_time_ms=open_time_ms,
        output_path=animation_gif_path,
        settings=settings,
    )

    print(f"Instante de fechamento: {timing.close_time_s:.9f} s")
    print(f"Instante exato de abertura: {open_time_s:.9f} s")
    print(
        "Corrente imediatamente antes da abertura: "
        f"{current_just_before_opening_a:.6f} A"
    )
    print(f"TRV maxima absoluta: {max_trv_v / 1.0e3:.6f} kV")
    print("Arquivos gerados:")
    print(f"  - {static_html_path}")
    print(f"  - {animation_gif_path}")

    return {
        "close_time_s": timing.close_time_s,
        "open_time_s": open_time_s,
        "current_just_before_opening_a": current_just_before_opening_a,
        "max_trv_v": max_trv_v,
    }


if __name__ == "__main__":
    simulate()
