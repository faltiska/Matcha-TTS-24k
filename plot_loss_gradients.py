"""
Interactive chart of loss-function gradients.

Shows how the gradient (d_loss / d_error) changes with the absolute error for:
  - MAE       (Mean Absolute Error)
  - MSE       (Mean Squared Error)
  - Huber     (controlled by delta slider)
  - SmoothL1  (controlled by beta slider)
  - log_cosh

Controls:
  - Huber delta  : single Slider
  - SmoothL1 beta: single Slider
  - MAE range    : Slider — set the x-axis upper limit

Run with:
    python plot_loss_gradients.py
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets


# ---------------------------------------------------------------------------
# Visual style
# ---------------------------------------------------------------------------

mpl.rcParams.update({
    "font.size":        16,
    "axes.titlesize":   17,
    "axes.labelsize":   15,
    "legend.fontsize":  14,
    "xtick.labelsize":  14,
    "ytick.labelsize":  14,
})

LINE_WIDTH = 3.0


# ---------------------------------------------------------------------------
# Gradient formulas
# Each function returns the gradient at every point in `abs_error`.
# ---------------------------------------------------------------------------

def gradient_mae(abs_error):
    """Gradient of MAE = sign(error).  Magnitude is always 1.0."""
    return np.ones_like(abs_error)


def gradient_mse(abs_error):
    """Gradient of MSE = 2 * error.  Grows linearly without bound."""
    return 2.0 * abs_error


def gradient_huber(abs_error, delta):
    """
    Huber gradient:
      |e| <= delta  ->  e          (quadratic region, grows linearly)
      |e|  > delta  ->  delta      (linear region, flat at delta)

    The gradient is continuous but has a kink at |e| = delta.
    """
    return np.where(abs_error <= delta, abs_error, delta)


def gradient_smooth_l1(abs_error, beta):
    """
    SmoothL1 (PyTorch convention) gradient:
      |e| < beta  ->  e / beta     (quadratic region, grows up to 1.0)
      |e| >= beta ->  1.0          (linear region, flat at 1.0)

    The gradient is always capped at 1.0, regardless of beta.
    A smaller beta makes it reach 1.0 faster (steeper quadratic region).
    """
    return np.where(abs_error < beta, abs_error / beta, 1.0)


def gradient_log_cosh(abs_error):
    """
    log-cosh gradient = tanh(e).
    Approaches 1.0 asymptotically; never exceeds 1.0.
    Smooth everywhere with no kinks.
    """
    return np.tanh(abs_error)


# ---------------------------------------------------------------------------
# Build the figure  (twice the default 10×6 → 20×12)
# ---------------------------------------------------------------------------

INITIAL_DELTA    = 0.9
INITIAL_BETA     = 1.3
INITIAL_X_MAX    = 2.5
RESOLUTION       = 1000   # number of x points

fig, ax = plt.subplots(figsize=(20, 12))
# Bottom margin reserves space for three slider rows
plt.subplots_adjust(left=0.08, right=0.97, top=0.95, bottom=0.22)

abs_error = np.linspace(0.0, INITIAL_X_MAX, RESOLUTION)

line_mae,     = ax.plot(abs_error, gradient_mae(abs_error),                        label="MAE",      linewidth=LINE_WIDTH)
line_mse,     = ax.plot(abs_error, gradient_mse(abs_error),                        label="MSE",      linewidth=LINE_WIDTH)
line_huber,   = ax.plot(abs_error, gradient_huber(abs_error,     INITIAL_DELTA),   label="Huber",    linewidth=LINE_WIDTH)
line_sl1,     = ax.plot(abs_error, gradient_smooth_l1(abs_error, INITIAL_BETA),    label="SmoothL1", linewidth=LINE_WIDTH)
line_logcosh, = ax.plot(abs_error, gradient_log_cosh(abs_error),                   label="log_cosh", linewidth=LINE_WIDTH)

ax.set_xlabel("Absolute error  |e|")
ax.set_ylabel("Gradient  d_loss / d_error")
ax.set_title("Loss-function gradients")
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# Sliders
# ---------------------------------------------------------------------------

SLIDER_COLOR = "lightgoldenrodyellow"

# Two single Sliders for scalar params
ax_delta = plt.axes([0.12, 0.13, 0.76, 0.02], facecolor=SLIDER_COLOR)
ax_beta  = plt.axes([0.12, 0.07, 0.76, 0.02], facecolor=SLIDER_COLOR)

slider_delta = widgets.Slider(ax_delta, "Huber delta",   0.05, 5.0, valinit=INITIAL_DELTA, valstep=0.05)
slider_beta  = widgets.Slider(ax_beta,  "SmoothL1 beta", 0.05, 5.0, valinit=INITIAL_BETA,  valstep=0.05)

# Single Slider for the x-axis upper limit
ax_range = plt.axes([0.12, 0.01, 0.76, 0.02], facecolor=SLIDER_COLOR)
slider_range = widgets.Slider(
    ax_range, "MAE range",
    valmin=0.1, valmax=10.0,
    valinit=INITIAL_X_MAX,
    valstep=0.1,
)


def redraw(_event):
    delta  = slider_delta.val
    beta   = slider_beta.val
    x_max  = slider_range.val

    abs_error = np.linspace(0.0, x_max, RESOLUTION)

    line_mae.set_xdata(abs_error)
    line_mae.set_ydata(gradient_mae(abs_error))

    line_mse.set_xdata(abs_error)
    line_mse.set_ydata(gradient_mse(abs_error))

    line_huber.set_xdata(abs_error)
    line_huber.set_ydata(gradient_huber(abs_error, delta))

    line_sl1.set_xdata(abs_error)
    line_sl1.set_ydata(gradient_smooth_l1(abs_error, beta))

    line_logcosh.set_xdata(abs_error)
    line_logcosh.set_ydata(gradient_log_cosh(abs_error))

    ax.set_xlim(0.0, x_max)
    ax.relim()
    ax.autoscale_view(scalex=False)   # y-axis auto-scales; x is fixed to slider
    fig.canvas.draw_idle()


slider_delta.on_changed(redraw)
slider_beta.on_changed(redraw)
slider_range.on_changed(redraw)

plt.show()
