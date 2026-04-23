"""
pareto_energy_latency.py
Generates an energy (J/inst) vs instruction latency (s/inst) Pareto scatter
using REAL McPAT power numbers — matches professor's Slide 3 format exactly.

Real McPAT numbers (45nm HP, 3 GHz gem5 X86O3CPU):
  Config           IPC(ROI)  TotalPower(W)
  Baseline         0.932     9.970
  Graph Reorder    0.986     9.691
  Scalar Quant     1.026     9.884
  Reorder+Quant    1.039     9.936
  L2=2MB (HW)      1.036    11.335
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

CLOCK_HZ = 3.0e9  # 3 GHz

configs = [
    # (label,        ipc_roi,  total_power_W,  color,        marker, zorder)
    ("Baseline\n(256kB L2)",  0.932,  9.970,  "#888888",    "o",    2),
    ("Graph\nReorder",        0.986,  9.691,  "#2196F3",    "s",    3),
    ("Scalar\nQuant (int8)",  1.026,  9.884,  "#FF9800",    "^",    3),
    ("Reorder +\nQuant ★",   1.039,  9.936,  "#4CAF50",    "D",    5),  # Pareto optimal
    ("L2=2MB\n(HW only)",     1.036, 11.335,  "#F44336",    "X",    3),
]

# Derived metrics
latency_ns = []   # nanoseconds per instruction  (1 / (IPC × freq)) × 1e9
energy_pj  = []   # picojoules per instruction    (Power × CPI / freq) × 1e12

for label, ipc, power, color, marker, zo in configs:
    cpi = 1.0 / ipc
    lat = cpi / CLOCK_HZ * 1e9          # ns/inst
    eng = power * cpi / CLOCK_HZ * 1e12  # pJ/inst
    latency_ns.append(lat)
    energy_pj.append(eng)

# ── Find Pareto-optimal frontier (min latency AND min energy simultaneously)
# A point is Pareto-optimal if no other point strictly dominates it on both axes.
n = len(configs)
is_pareto = [True] * n
for i in range(n):
    for j in range(n):
        if i == j:
            continue
        if latency_ns[j] <= latency_ns[i] and energy_pj[j] <= energy_pj[i] and \
           (latency_ns[j] < latency_ns[i] or energy_pj[j] < energy_pj[i]):
            is_pareto[i] = False
            break

# Sort Pareto points by latency for frontier line
pareto_pts = sorted(
    [(latency_ns[i], energy_pj[i]) for i in range(n) if is_pareto[i]],
    key=lambda p: p[0]
)

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5.5))
fig.patch.set_facecolor("white")
ax.set_facecolor("#F9F9F9")

# Draw Pareto frontier step-line
if len(pareto_pts) >= 2:
    px = [p[0] for p in pareto_pts]
    py = [p[1] for p in pareto_pts]
    ax.step(px, py, where="post", color="#4CAF50", linewidth=1.8,
            linestyle="--", alpha=0.7, label="Pareto frontier")

# Scatter points
for i, (label, ipc, power, color, marker, zo) in enumerate(configs):
    ax.scatter(latency_ns[i], energy_pj[i],
               s=220, color=color, marker=marker, zorder=zo,
               edgecolors="black", linewidths=0.8)
    # Label offset to avoid overlap
    offsets = {
        0: (5,  6),   # Baseline
        1: (5,  6),   # Reorder
        2: (5, -14),  # Quant
        3: (5,  6),   # Reorder+Quant
        4: (5,  6),   # L2=2MB
    }
    dx, dy = offsets.get(i, (5, 5))
    ax.annotate(
        label,
        xy=(latency_ns[i], energy_pj[i]),
        xytext=(dx, dy), textcoords="offset points",
        fontsize=8.5, ha="left",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7)
    )

# ── Annotation: "Pareto-optimal" callout for Reorder+Quant
best_idx = 3  # Reorder+Quant
ax.annotate(
    "Pareto-optimal:\nlowest latency\n& lowest energy",
    xy=(latency_ns[best_idx], energy_pj[best_idx]),
    xytext=(-70, -40), textcoords="offset points",
    fontsize=8, color="#2E7D32", fontweight="bold",
    arrowprops=dict(arrowstyle="->", color="#2E7D32", lw=1.5),
    bbox=dict(boxstyle="round", fc="#E8F5E9", ec="#4CAF50", alpha=0.9)
)

# ── Axes labels / title
ax.set_xlabel("Instruction Latency  (ns / inst = CPI / freq)", fontsize=11)
ax.set_ylabel("Energy per Instruction  (pJ / inst = Power × CPI / freq)", fontsize=11)
ax.set_title(
    "Performance–Energy Pareto  (McPAT 45nm HP, gem5 SE mode)\n"
    "Lower-left = better.  ★ marks sole Pareto-optimal point.",
    fontsize=11, fontweight="bold"
)

# ── "Better" direction text label (no arrow — axis limits not set yet here)
ax.text(0.98, 0.97, "← lower latency is better\n↓ lower energy is better",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=8, color="gray", style="italic")

# Legend handles
legend_elems = [
    plt.scatter([], [], s=100, color=c[3], marker=c[4],
                edgecolors="black", linewidths=0.8, label=c[0].replace("\n", " "))
    for c in configs
]
ax.legend(handles=legend_elems, fontsize=8, loc="upper right",
          framealpha=0.9, title="Configuration")

ax.grid(True, linestyle="--", alpha=0.4, color="gray")

# Force axis limits to show all points with breathing room
x_pad = 1.5
y_pad = 30
ax.set_xlim(min(latency_ns) - x_pad, max(latency_ns) + x_pad * 4)
ax.set_ylim(min(energy_pj) - y_pad, max(energy_pj) + y_pad * 3)

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), "..", "results", "pareto_energy_latency.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"[saved] {out}")

# ── Print table
print("\n{'Config':<22} {'Latency(ns/inst)':>18} {'Energy(pJ/inst)':>16} {'Pareto?':>8}")
print("-" * 70)
for i, (label, ipc, power, color, marker, zo) in enumerate(configs):
    name = label.replace("\n", " ")
    print(f"  {name:<22} {latency_ns[i]:>16.3f}   {energy_pj[i]:>14.2f}   {'★ YES' if is_pareto[i] else 'no':>8}")
