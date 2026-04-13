"""
Power Transformer Over-Excitation Animation
============================================
Layout – 2x2 equal-size subplots:
  - Top-left    : sinusoidal green flux waveform (builds over time)
  - Bottom-left : empty (intentionally blank)
  - Top-right   : red S-curve (core saturation characteristic) + animated dot
  - Bottom-right: blue magnetizing current waveform (distorted, peaked)

Axis labels are shown on each active subplot (no tick numbers).
Animation plays at half the original speed (15 fps saved, same frame count).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, PillowWriter

# ============================================================
# Physics: nonlinear core saturation model
# ============================================================
T        = 1.0            # fundamental period [s]
OMEGA    = 2 * np.pi / T
PHI_MAX  = 1.58           # peak flux (saturation knee at 1.0)
PHI_KNEE = 1.0            # knee-point flux


def im_of_phi(phi):
    """Magnetizing current as a strongly nonlinear function of flux."""
    k_lin  = 0.22          # slope in linear (unsaturated) region
    k_sat  = 13.0          # gain in saturated region
    n_sat  = 2.35          # exponent (controls sharpness of peaks)
    excess = np.maximum(np.abs(phi) - PHI_KNEE, 0.0)
    return k_lin * phi + k_sat * excess ** n_sat * np.sign(phi)


# ============================================================
# Pre-compute time-domain waveforms
# ============================================================
N_CYCLES = 4
DURATION = N_CYCLES * T
N_PTS    = 4000

t   = np.linspace(0, DURATION, N_PTS)
phi = PHI_MAX * np.sin(OMEGA * t)
im  = im_of_phi(phi)

# Static core characteristic (full S-curve)
phi_char = np.linspace(-1.95, 1.95, 900)
im_char  = im_of_phi(phi_char)

# ============================================================
# Figure & axes
# ============================================================
BG_COLOR   = '#CACACA'    # light-grey background
GRID_COLOR = '#A8A8A8'    # soft grey grid
REF_COLOR  = '#909090'    # reference / center lines

fig = plt.figure(figsize=(10, 7), facecolor=BG_COLOR)
gs  = gridspec.GridSpec(
    2, 2,
    left=0.10, right=0.97,
    top=0.96,  bottom=0.09,
    wspace=0.38, hspace=0.38,
)

ax_phi = fig.add_subplot(gs[0, 0])   # top-left    : flux waveform
# gs[1, 0] is intentionally left empty (no subplot created)
ax_sat = fig.add_subplot(gs[0, 1])   # top-right   : saturation curve
ax_im  = fig.add_subplot(gs[1, 1])   # bottom-right: magnetizing current

# Uniform visual style: no tick numbers, clean spines, soft grid
LABEL_STYLE = dict(fontsize=7.5, color='#444444', labelpad=4)

for ax in (ax_phi, ax_sat, ax_im):
    ax.set_facecolor(BG_COLOR)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(GRID_COLOR)
        sp.set_linewidth(0.7)
    ax.grid(True, color=GRID_COLOR, linewidth=0.45, alpha=0.7)

# Axis labels (descriptive text only, no tick values)
ax_phi.set_ylabel('Fluxo magnético (Wb)',                    **LABEL_STYLE)
ax_phi.set_xlabel('Tempo [s]',                               **LABEL_STYLE)
ax_sat.set_ylabel('Fluxo magnético (Wb)',                    **LABEL_STYLE)
ax_sat.set_xlabel('Corrente Equivalente de Magnetização (Ae)', **LABEL_STYLE)
ax_im.set_ylabel('Corrente Equivalente de Magnetização (Ae)', **LABEL_STYLE)
ax_im.set_xlabel('Tempo [s]',                                **LABEL_STYLE)

# ---- axis limits ----
PHI_LIM = (-2.05, 2.05)
I_PEAK  = max(abs(im.min()), abs(im.max()))
I_LIM   = (-I_PEAK * 1.18, I_PEAK * 1.18)
I_CHAR_LIM = (im_char.min() * 1.18, im_char.max() * 1.18)

ax_phi.set_xlim(0, DURATION);         ax_phi.set_ylim(PHI_LIM)
ax_sat.set_xlim(I_CHAR_LIM);          ax_sat.set_ylim(PHI_LIM)
ax_im.set_xlim(0, DURATION);          ax_im.set_ylim(I_LIM)

# ---- centre / reference lines ----
for ax in (ax_phi, ax_im):
    ax.axhline(0, color=REF_COLOR, lw=0.9)
ax_sat.axhline(0, color=REF_COLOR, lw=0.9)
ax_sat.axvline(0, color=REF_COLOR, lw=0.9)

# Dashed horizontal lines at saturation knee on flux and S-curve panels
for lv in (PHI_KNEE, -PHI_KNEE):
    ax_phi.axhline(lv, color=REF_COLOR, lw=0.7, ls='--')
    ax_sat.axhline(lv, color=REF_COLOR, lw=0.7, ls='--')

# Dashed horizontal lines on current panel at the current values that
# correspond to the saturation knee point (im_of_phi(±PHI_KNEE))
IM_KNEE = im_of_phi(PHI_KNEE)   # positive knee current
for lv in (IM_KNEE, -IM_KNEE):
    ax_im.axhline(lv, color=REF_COLOR, lw=0.7, ls='--')

# ---- static S-curve (core characteristic) ----
# x = magnetizing current, y = flux  →  matches reference image orientation
ax_sat.plot(im_char, phi_char, color='#CC2200', lw=2.6, zorder=3)

# ============================================================
# Animated artists
# ============================================================
FPS      = 30
N_FRAMES = int(FPS * DURATION)

line_phi, = ax_phi.plot([], [], color='#00CC00', lw=2.3, zorder=3)
line_im,  = ax_im.plot([],  [], color='#1144BB', lw=2.1, zorder=3)

# Dot on saturation curve (orange, prominent)
dot_sat, = ax_sat.plot([], [], 'o',
                        color='#FF4400', ms=10,
                        mec='white', mew=1.3, zorder=5)

# Trailing dots on flux and current plots (show current position)
dot_phi, = ax_phi.plot([], [], 'o',
                        color='#00FF55', ms=7,
                        mec='white', mew=1.0, zorder=5)
dot_im,  = ax_im.plot([],  [], 'o',
                        color='#3366EE', ms=7,
                        mec='white', mew=1.0, zorder=5)


def init():
    for artist in (line_phi, line_im, dot_sat, dot_phi, dot_im):
        artist.set_data([], [])
    return line_phi, line_im, dot_sat, dot_phi, dot_im


def animate(frame):
    # Map frame index → data index
    idx = max(1, int(round(frame / N_FRAMES * N_PTS)))
    idx = min(idx, N_PTS - 1)

    # Build waveforms progressively (left to right in time)
    line_phi.set_data(t[:idx],  phi[:idx])
    line_im.set_data( t[:idx],  im[:idx])

    # Animated dots track the current operating point
    dot_sat.set_data([im[idx - 1]],  [phi[idx - 1]])   # (x=i, y=phi) on S-curve
    dot_phi.set_data([t[idx - 1]],   [phi[idx - 1]])
    dot_im.set_data( [t[idx - 1]],   [im[idx - 1]])

    return line_phi, line_im, dot_sat, dot_phi, dot_im


# ============================================================
# Render and save
# ============================================================
anim = FuncAnimation(
    fig, animate, init_func=init,
    frames=N_FRAMES, interval=1000 // FPS, blit=True,
)

OUTPUT  = 'transformer_overexcitation.gif'
FPS_OUT = FPS // 2          # save at half speed (15 fps) → 2× slower playback
print(f"Rendering {N_FRAMES} frames, saved at {FPS_OUT} fps (half speed) …")
anim.save(OUTPUT, writer=PillowWriter(fps=FPS_OUT), dpi=90)
plt.close(fig)
print(f"Saved: {OUTPUT}")
