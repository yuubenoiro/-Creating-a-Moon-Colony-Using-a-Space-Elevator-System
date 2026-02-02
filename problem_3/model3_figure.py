"""
Figure utilities — polished plotting, reproducible and export-ready.
- Improvements: color palette, clearer annotations, axis formatting, font fallbacks, high-res export
- Usage: run as a script or import create_figures(save_dir, dpi)
"""

from __future__ import annotations
import os
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.font_manager as fm
from scipy.signal import savgol_filter

# -----------------------------
# Color palette & style (replaceable)
# -----------------------------
PALETTE = {
    'se': '#3B7786',        # deep teal (space elevator)
    'rocket': '#E6952D',    # warm orange (rocket)
    'target': '#7E8890',
    'grid': '#F3F6F7',
    'muted': '#5B6770',
}

# Font fallback: prefer SimHei (if present) otherwise use system sans-serif
def _set_fonts() -> None:
    available = {f.name for f in fm.fontManager.ttflist}
    if 'SimHei' in available:
        plt.rcParams['font.family'] = 'SimHei'
    else:
        plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False

_set_fonts()
# style: try seaborn first (if available) otherwise fall back to matplotlib defaults
try:
    import seaborn as sns
    sns.set_style('white')
except Exception:
    plt.style.use('default')
    plt.rcParams.update({'axes.grid': False})

# -----------------------------
# Utilities: number formatting and data generation
# -----------------------------
def _k_formatter(x, pos: int) -> str:
    """Format large numbers with 'k' suffix (e.g. 152555 -> 152.6k)."""
    if x >= 1000:
        return f"{x/1000:.1f}k"
    return f"{int(x)}"


def _generate_data(seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    days = np.arange(1, 31)
    M_TOTAL = 152_555

    # Space elevator: fast initial throughput then converges (with noise)
    daily_se = np.clip(12_000 * rng.uniform(0.82, 1.05, size=30) *
                       np.exp(-0.03 * np.arange(30)), 0, None)
    m_se = np.minimum(np.cumsum(daily_se), M_TOTAL)

    # Rocket: steady but slower, with daily variance
    daily_rk = rng.normal(4_275, 150, size=30)
    m_rk = np.minimum(np.cumsum(np.clip(daily_rk, 2000, None)), M_TOTAL)

    return {
        'days': days,
        'M_TOTAL': M_TOTAL,
        'm_se': m_se,
        'm_rk': m_rk,
        'seed': seed,
    }

# -----------------------------
# Main plotting function
# -----------------------------
def create_figures(save_dir: Optional[str] = 'figures', dpi: int = 300, seed: int = 42) -> None:
    """Generate and save three polished figures: convergence, cost-vs-time, reliability.

    Args:
        save_dir: output directory (created if missing)
        dpi: export resolution
        seed: RNG seed for reproducibility
    """
    os.makedirs(save_dir, exist_ok=True)
    data = _generate_data(seed=seed)
    days = data['days']
    M_TOTAL = data['M_TOTAL']
    m_se = data['m_se']
    m_rk = data['m_rk']

    # Global styling parameters
    MARKER_KW = dict(markersize=6, markeredgewidth=0.8, markeredgecolor='white')
    LABEL_KW = dict(fontsize=10)

    # ---------- Figure 1: Convergence ----------
    fig1, ax1 = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax1.plot(days, m_se, color=PALETTE['se'], linewidth=3.2, label='Space Elevator (SE)',
             marker='o', markevery=4, **MARKER_KW)
    ax1.plot(days, m_rk, color=PALETTE['rocket'], linewidth=2.2, linestyle='--',
             label='Traditional Rocket', marker='s', markevery=5, **MARKER_KW)

    ax1.axhline(M_TOTAL, color=PALETTE['target'], linestyle=':', linewidth=1.1)
    # annotate day when SE reaches the target
    reach_idx = np.searchsorted(m_se, M_TOTAL)
    if reach_idx < len(days):
        reach_day = days[reach_idx]
        ax1.scatter([reach_day], [M_TOTAL], color=PALETTE['se'], zorder=5)
        ax1.annotate(f"Target reached (Day {reach_day})", xy=(reach_day, M_TOTAL), xytext=(reach_day-6, M_TOTAL*0.9),
                     arrowprops=dict(arrowstyle='->', color=PALETTE['se']), bbox=dict(boxstyle='round', fc='white', alpha=0.6), fontsize=9)

    ax1.set_title('Water Delivery Convergence', fontsize=14, pad=12)
    ax1.set_xlabel('Operation Days', **LABEL_KW)
    ax1.set_ylabel('Accumulated Mass (Tons)', **LABEL_KW)
    ax1.yaxis.set_major_formatter(mtick.FuncFormatter(_k_formatter))
    ax1.grid(axis='y', color=PALETTE['grid'])
    ax1.set_xlim(1, days[-1])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.legend(frameon=False, fontsize=10)
    fig1.savefig(os.path.join(save_dir, 'figure1_convergence.png'), dpi=dpi)

    # ---------- Figure 2: Cost vs Time (bar + line) ----------
    fig2, ax_cost = plt.subplots(figsize=(7.5, 5), constrained_layout=True)
    labels = ['SE', 'Rocket']
    costs = np.array([16.02, 61.02])  # Billion USD
    days_to_target = np.array([14, 35])

    bars = ax_cost.bar(labels, costs, color=[PALETTE['se'], PALETTE['rocket']], width=0.48,
                       edgecolor='none', alpha=0.95)
    ax_cost.set_ylabel('Total Cost (Billion USD)', **LABEL_KW)
    for b, v in zip(bars, costs):
        ax_cost.text(b.get_x() + b.get_width()/2, v + 1.2, f"${v:.2f}B", ha='center', va='bottom', fontsize=9)

    ax_time = ax_cost.twinx()
    ax_time.plot(labels, days_to_target, color=PALETTE['muted'], marker='D', linewidth=2.2,
                 markersize=7, label='Days to target')
    ax_time.set_ylabel('Days to Target', **LABEL_KW)
    ax_time.set_ylim(0, max(days_to_target) * 1.4)

    # compute and annotate percent saved
    pct_saved = (1 - costs[0] / costs[1]) * 100
    ax_cost.annotate(f"Cost reduction {pct_saved:.0f}%", xy=(0, costs[0]), xytext=(0.5, costs[0]*0.6),
                     ha='center', fontsize=10, bbox=dict(boxstyle='round', fc='white', alpha=0.7))

    ax_cost.set_title('Cost & Time Efficiency', fontsize=13, pad=12)
    ax_cost.spines['top'].set_visible(False)
    ax_time.spines['top'].set_visible(False)
    fig2.savefig(os.path.join(save_dir, 'figure2_cost_time.png'), dpi=dpi)

    # ---------- Figure 3: Reliability (availability with noise) ----------
    fig3, ax3 = plt.subplots(figsize=(9, 4), constrained_layout=True)
    t = np.linspace(0, 30, 200)
    rng = np.random.default_rng(seed)
    availability = 0.95 - (np.abs(np.sin(t * 0.45)) * 0.1) + rng.normal(0, 0.015, t.size)
    availability = np.clip(availability, 0.7, 1.0)

    # smooth to highlight trend
    avail_smooth = savgol_filter(availability, 31, 3)
    ax3.fill_between(t, avail_smooth, 1.0, color=PALETTE['se'], alpha=0.18)
    ax3.plot(t, avail_smooth, color=PALETTE['se'], linewidth=2.4, label='System availability')
    ax3.plot(t, availability, color=PALETTE['se'], linewidth=0.8, alpha=0.35)

    ax3.axhline(0.8, color=PALETTE['rocket'], linestyle='--', linewidth=1.1, alpha=0.8)
    ax3.set_ylim(0.72, 1.02)
    ax3.set_xlabel('Operation Period (Days)', **LABEL_KW)
    ax3.set_ylabel('Availability Rate', **LABEL_KW)
    ax3.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))

    mean_avail = avail_smooth.mean()
    ax3.text(0.98, 0.08, f"Mean availability {mean_avail:.1%}", transform=ax3.transAxes,
             ha='right', va='bottom', fontsize=10, bbox=dict(boxstyle='round', fc='white', alpha=0.7))

    ax3.set_title('System Reliability Simulation', fontsize=13, pad=10)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.legend(frameon=False, fontsize=9, loc='lower left')
    fig3.savefig(os.path.join(save_dir, 'figure3_reliability.png'), dpi=dpi)

    plt.close('all')


if __name__ == '__main__':
    # Running as a script saves figures into a local 'figures/' folder
    create_figures(save_dir='figures', dpi=300, seed=42)
