"""
gen_arch_diagram.py  --  EEL6764 HNSW Architecture Diagram  v4
Root-cause fix: all long text uses ha='left' anchored to its zone boundary.
Center-aligned text is only used for short strings inside boxes.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

OUT = r"d:\GEM5_Class_Project\report\figures\fig0_arch_diagram.png"

# ── figure canvas ─────────────────────────────────────────────────────────────
FW, FH = 24, 11
fig, ax = plt.subplots(figsize=(FW, FH))
ax.set_xlim(0, FW); ax.set_ylim(0, FH); ax.axis("off")
fig.patch.set_facecolor("#F4F6F9")

# Zone boundaries (x)
Z_PIPE_L,  Z_PIPE_R  = 0.20,  2.90   # pipeline
Z_CACHE_L, Z_CACHE_R = 3.10,  5.55   # cache hierarchy
Z_HNSW_L,  Z_HNSW_R  = 5.75, 14.00   # HNSW + bottom panels
Z_TABLE_L, Z_TABLE_R = 14.30, 23.80  # results table
TITLE_H = 0.72                         # height of title bar at top

# Usable y range (below title bar)
SUB_H = 0.55                           # sub-header strip height
Y_TOP = FH - TITLE_H - SUB_H - 0.10  # content starts below sub-header
Y_BOT = 0.35

# ── helpers ───────────────────────────────────────────────────────────────────
def fbox(x, y, w, h, fc, ec, lw=1.5, r=0.15, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle=f"round,pad=0.04,rounding_size={r}",
        fc=fc, ec=ec, lw=lw, zorder=z))

def txt(x, y, s, c="#111", fs=8.5, ha="center", va="center",
        bold=False, italic=False, z=6):
    ax.text(x, y, s, ha=ha, va=va, fontsize=fs, color=c, zorder=z,
            fontweight="bold" if bold else "normal",
            fontstyle="italic" if italic else "normal")

def arr(x1, y1, x2, y2, c="#555", lw=1.6, head=11):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=c, lw=lw,
                        mutation_scale=head), zorder=5)

def hline(y, x0=None, x1=None, c="#CCC", lw=0.7):
    ax.plot([x0 or 0, x1 or FW], [y, y], color=c, lw=lw, zorder=3)

# ════════════════════════════════════════════════════════════════════════
# Title banner
# ════════════════════════════════════════════════════════════════════════
ax.add_patch(plt.Rectangle((0, FH - TITLE_H), FW, TITLE_H,
                            fc="#1565C0", ec="none", zorder=6))
txt(FW/2, FH - TITLE_H/2,
    "EEL6764  ·  HNSW on gem5 X86O3CPU  —  Microarchitectural Analysis Overview",
    "white", fs=13.5, bold=True, z=7)

# ════════════════════════════════════════════════════════════════════════
# A. X86O3CPU Pipeline  (zone: Z_PIPE_L – Z_PIPE_R)
# ════════════════════════════════════════════════════════════════════════
PX  = Z_PIPE_L
PW  = Z_PIPE_R - Z_PIPE_L    # 2.70
PH  = 0.68
PG  = 0.18
SY  = FH - TITLE_H - SUB_H   # bottom edge of sub-header strip
TOP = SY - PH - 0.15          # FETCH box top sits 0.15 below strip bottom

stages = [
    ("FETCH",     "4-wide · 3 GHz",       "#C8E6C9", "#2E7D32"),
    ("DECODE",    "x86 → µ-ops",          "#BBDEFB", "#1565C0"),
    ("RENAME",    "Tomasulo reg. map",    "#BBDEFB", "#1565C0"),
    ("DISPATCH",  "Reservation stations", "#BBDEFB", "#1565C0"),
    ("EXECUTE",   "IntALU / FPU / LSU",   "#BBDEFB", "#1565C0"),
    ("WRITEBACK", "CDB broadcast",        "#BBDEFB", "#1565C0"),
    ("ROB",       "128 entries",          "#FFE0B2", "#E65100"),
    ("COMMIT",    "In-order retire",      "#C8E6C9", "#2E7D32"),
]

stage_cy = []
for i, (name, sub, fc, ec) in enumerate(stages):
    y = TOP - i * (PH + PG)
    fbox(PX, y, PW, PH, fc, ec, lw=1.6)
    mid = y + PH/2
    stage_cy.append(mid)
    txt(PX + PW/2, mid + 0.13, name, fs=9, bold=True)
    txt(PX + PW/2, mid - 0.15, sub,  fs=7.5, c="#444")
    if i < len(stages) - 1:
        arr(PX + PW/2, y, PX + PW/2, y - PG, c="#1565C0", lw=1.4)


# Back-pressure arrow
rob_top = TOP - 6*(PH+PG) + PH
arr_x = PX - 0.10
ax.annotate("", xy=(arr_x, TOP + 0.05),
            xytext=(arr_x, rob_top - 0.05),
            arrowprops=dict(arrowstyle="<|-", color="#C62828",
                            lw=2.2, mutation_scale=14), zorder=5)
txt(arr_x - 0.42, (TOP + rob_top)/2,
    "Back-\npressure\n(SQ full\n2.97 M×)", "#C62828", fs=7, bold=True)

# ════════════════════════════════════════════════════════════════════════
# B. Cache Hierarchy  (zone: Z_CACHE_L – Z_CACHE_R)
# ════════════════════════════════════════════════════════════════════════
CX = Z_CACHE_L
CW = Z_CACHE_R - Z_CACHE_L   # 2.45

caches = [
    ("L1-I Cache",  "32 kB · 0.15 MPKI",         "#E8F5E9","#388E3C",  8.60, 0.65),
    ("L1-D Cache",  "32 kB · 7.60 MPKI",          "#FFECB3","#F57F17",  7.10, 0.65),
    ("L2 Cache",    "256 kB · 91.7% miss rate",    "#FFCDD2","#C62828",  5.60, 0.65),
    ("DDR4-2400",   "19.2 GB/s peak · 1.61 GB/s used\n(8.4% util — latency-bound)",
                                                    "#EDE7F6","#4527A0",  3.90, 0.95),
]

cmid = {}
for name, sub, fc, ec, cy, h in caches:
    fbox(CX, cy - h/2, CW, h, fc, ec, lw=1.6)
    cmid[name] = cy
    line2 = "\n" in sub
    txt(CX + CW/2, cy + (0.15 if line2 else 0.10), name, fs=9, bold=True)
    if line2:
        parts = sub.split("\n")
        txt(CX + CW/2, cy - 0.13, parts[0], fs=7.5, c="#444")
        txt(CX + CW/2, cy - 0.32, parts[1], fs=7.0, c="#666")
    else:
        txt(CX + CW/2, cy - 0.15, sub, fs=7.5, c="#444")


# Inter-cache arrows
def _h(name):
    return next(c[5] for c in caches if c[0]==name)
pairs = [("L1-I Cache","L1-D Cache"),("L1-D Cache","L2 Cache"),("L2 Cache","DDR4-2400")]
for a, b in pairs:
    arr(CX + CW/2, cmid[a] - _h(a)/2,
        CX + CW/2, cmid[b] + _h(b)/2, c="#5D4037", lw=1.5)

# Callout: left-anchored just past cache zone, above the DIVIDER line
txt(Z_CACHE_R + 0.08, cmid["L2 Cache"] + 0.30,
    "91.7% miss", "#C62828", fs=7.5, bold=True, ha="left")
txt(Z_CACHE_R + 0.08, cmid["L2 Cache"] + 0.05,
    "(pass-thru)", "#C62828", fs=7.0, ha="left")

# Pipeline → cache connections
arr(PX + PW, stage_cy[0], CX, cmid["L1-I Cache"], c="#388E3C", lw=1.2)
txt((PX+PW + CX)/2, cmid["L1-I Cache"] + 0.30, "I-fetch",   "#388E3C", fs=7.5, italic=True)
arr(PX + PW, stage_cy[4], CX, cmid["L1-D Cache"], c="#F57F17", lw=1.2)
txt((PX+PW + CX)/2, cmid["L1-D Cache"] + 0.30, "Load/Store","#F57F17", fs=7.5, italic=True)

# ════════════════════════════════════════════════════════════════════════
# Divider line: top half (HNSW) vs bottom half (CPI + Roofline)
# ════════════════════════════════════════════════════════════════════════
DIVIDER_Y = 5.10
hline(DIVIDER_Y, Z_HNSW_L, Z_TABLE_R, c="#BDBDBD", lw=1.2)

# ════════════════════════════════════════════════════════════════════════
# C. HNSW Pointer-Chasing  (zone: Z_HNSW_L – Z_HNSW_R, y > DIVIDER_Y)
# ════════════════════════════════════════════════════════════════════════
NR    = 0.42
NODE_CY = 8.10
NODE_X  = [6.50, 8.30, 10.10, 11.90]
LABELS  = ["Node A", "Node B", "Node C", "Node D"]
NCOLORS = ["#90CAF9","#90CAF9","#90CAF9","#B2EBF2"]

for i, (nx, lbl, fc) in enumerate(zip(NODE_X, LABELS, NCOLORS)):
    ax.add_patch(plt.Circle((nx, NODE_CY), NR, fc=fc, ec="#1565C0", lw=1.8, zorder=3))
    txt(nx, NODE_CY, lbl, "#0D47A1", fs=8.5, bold=True)
    if i < 3:
        # arrow
        arr(nx + NR, NODE_CY, NODE_X[i+1] - NR, NODE_CY, c="#1565C0", lw=2.0)
        mid_x = (nx + NODE_X[i+1]) / 2
        # labels ABOVE arrow — short strings, centered between two nodes (safe)
        txt(mid_x, NODE_CY + 0.72, "ptr-chase",          "#C62828", fs=8.0, bold=True)
        txt(mid_x, NODE_CY + 0.44, "DRAM miss (~100 cyc)","#C62828", fs=7.5)

ax.text(NODE_X[-1] + NR + 0.18, NODE_CY, "→  …", fontsize=13,
        color="#1565C0", va="center", zorder=4)


# RAW annotations — LEFT-aligned, anchored at Z_HNSW_L (no leftward bleed)
ANN_X = Z_HNSW_L + 0.10
txt(ANN_X, NODE_CY - 0.80,
    "RAW hazard: addr(Node B) is inside the result of load(Node A) — loads cannot be overlapped",
    "#C62828", fs=8, ha="left", italic=True)
txt(ANN_X, NODE_CY - 1.18,
    "Forwarding removes 1–2 cycle ALU stalls; DRAM stalls = 100+ cycles — forwarding is irrelevant",
    "#BF360C", fs=7.5, ha="left", italic=True)

# ════════════════════════════════════════════════════════════════════════
# D. CPI Stack Bar  (zone: Z_HNSW_L – 8.1, y < DIVIDER_Y)
# ════════════════════════════════════════════════════════════════════════
BX  = Z_HNSW_L + 0.10
BW  = 1.70
BY0 = Y_BOT + 0.30
BSC = (DIVIDER_Y - BY0 - 0.60) / 0.996   # scale so bar fills zone height

cpi_items = [
    (0.250, "Ideal 4-wide",  "0.250  (25%)", "#66BB6A"),
    (0.322, "Mem stall",     "0.322  (32%)", "#EF5350"),
    (0.032, "Branch",        "0.032   (3%)", "#FFA726"),
    (0.392, "Other ILP<4",   "0.392  (39%)", "#AB47BC"),
]
cy0 = BY0
for val, lbl1, lbl2, fc in cpi_items:
    h = val * BSC
    ax.add_patch(plt.Rectangle((BX, cy0), BW, h, fc=fc, ec="white", lw=1.2, zorder=3))
    my = cy0 + h/2
    if h > 0.55:
        txt(BX + BW/2, my + 0.10, lbl1, "white", fs=8.5, bold=True)
        txt(BX + BW/2, my - 0.15, lbl2, "white", fs=7.5)
    else:
        # Branch slice is thin — put label to the right (inside HNSW zone, not cache zone)
        txt(BX + BW + 0.12, my, f"{lbl1}  {lbl2}", "#444", fs=7.5, ha="left")
    cy0 += h

txt(BX + BW/2, BY0 - 0.22, "CPI = 0.996", "#111", fs=9, bold=True)
txt(BX + BW/2, DIVIDER_Y + 0.22, "CPI Stack", "#333", fs=8.5, bold=True)

# ════════════════════════════════════════════════════════════════════════
# E. Roofline Key Finding  (x: 8.4 – 14.0, y < DIVIDER_Y)
# ════════════════════════════════════════════════════════════════════════
RX = BX + BW + 0.55
RY = Y_BOT
RW = Z_TABLE_L - RX - 0.25
RH = DIVIDER_Y - RY - 0.10

fbox(RX, RY, RW, RH, "#FFF8E1", "#F9A825", lw=2, r=0.20)
txt(RX + RW/2, RY + RH - 0.32, "Roofline Model — Key Finding", "#E65100", fs=10, bold=True)

rows = [
    ("Arithmetic Intensity (AI)",             "2.09 FLOP/byte",                             "#37474F"),
    ("Standard ridge pt. (peak BW 19.2 GB/s)","0.63 FLOP/byte  →  appears COMPUTE-BOUND",   "#C62828"),
    ("Corrected ridge (eff. BW 1.61 GB/s)",   "7.45 FLOP/byte  →  truly LATENCY-BOUND",     "#C62828"),
    ("FP unit busy counters",                 "≈ 0%  (FPU idle despite AI > standard ridge)","#C62828"),
    ("0-issue stall cycles",                  "32.7% of all cycles  (pure DRAM wait)",       "#E65100"),
    ("Amdahl ceiling (eliminate all stalls)", "+47.8% IPC max  →  ceiling = 1.484 IPC",     "#37474F"),
    ("Best HW result  (L2 = 2 MB)",           "+10.6% IPC    2.40× die area",                "#1565C0"),
    ("Best SW result  (Reorder + Quant)",     "+10.2% IPC    1.00× die area   ← Pareto",    "#2E7D32"),
]
ROW_H_R = (RH - 0.65) / len(rows)
for j, (k, v, vc) in enumerate(rows):
    ry = RY + RH - 0.70 - j * ROW_H_R
    # key: left-aligned at left edge of Roofline box
    txt(RX + 0.16, ry,        k + ":", "#37474F", fs=8.0, ha="left", bold=True)
    txt(RX + 0.16, ry - 0.22, v,       vc,        fs=8.0, ha="left")
    ax.plot([RX+0.1, RX+RW-0.1], [ry-ROW_H_R*0.48, ry-ROW_H_R*0.48],
            color="#E8E8E8", lw=0.5, zorder=3)

# ════════════════════════════════════════════════════════════════════════
# F. All-22 Configs Table  (zone: Z_TABLE_L – Z_TABLE_R, full height)
# ════════════════════════════════════════════════════════════════════════
TX = Z_TABLE_L
TW = Z_TABLE_R - TX    # 9.50
TY = Y_BOT
TH = Y_TOP - TY

fbox(TX, TY, TW, TH, "#E8F5E9", "#388E3C", lw=2, r=0.20)

C0 = TX + 0.20
C1 = TX + 5.30
C2 = TX + 6.90
C3 = TX + 8.35
HY = TY + TH - 0.68

for cx, hdr in [(C0,"Configuration"),(C1,"IPC gain"),(C2,"Area"),(C3,"Verdict")]:
    txt(cx, HY, hdr, "#37474F", fs=9, ha="left", bold=True)
ax.plot([TX+0.1, TX+TW-0.1], [HY-0.20, HY-0.20], color="#BDBDBD", lw=1.0, zorder=4)

table = [
    ("── Hardware ──", "",       "",       ""),
    ("L2 = 512 kB",    "+0.3%",  "1.20×",  "NO"),
    ("L2 = 1 MB",      "+0.5%",  "1.60×",  "NO"),
    ("L2 = 2 MB",      "+10.6%", "2.40×",  "MARGINAL"),
    ("ROB = 32",       "−13.4%", "0.96×",  "NO"),
    ("ROB = 64",       "+0.5%",  "0.97×",  "NO"),
    ("ROB = 256",      "+0.7%",  "1.05×",  "NO"),
    ("HW Stride Pf.",  "0.0%",   "1.02×",  "NO"),
    ("DDR5-6400",      "−10.1%", "1.00×",  "NO"),
    ("HBM 1.0",        "0.0%",   "1.00×",  "NO"),
    ("MSHRs = 128",    "0.0%",   "1.00×",  "NO"),
    ("── Software ──", "",       "",       ""),
    ("SW Prefetch",    "−2.7%",  "1.00×",  "NO"),
    ("Graph Reorder",  "+1.8%",  "1.00×",  "NO"),
    ("Scalar Quant",   "+3.7%",  "1.00×",  "MARGINAL"),
    ("Reorder+Quant",  "+10.2%", "1.00×",  "✓ YES"),
    ("Reorder+L2=2MB", "+11.7%", "2.40×",  "MARGINAL"),
]
VC = {"NO":"#C62828","MARGINAL":"#E65100","✓ YES":"#2E7D32","":"#888"}
RH_T = (TH - 1.05) / len(table)

for j, (cfg, gain, area, verd) in enumerate(table):
    ry  = HY - 0.38 - j * RH_T
    sep = cfg.startswith("──")
    best= verd == "✓ YES"
    tc  = VC.get(verd, "#37474F")

    if best:
        fbox(TX+0.10, ry - RH_T*0.46, TW-0.20, RH_T*0.90,
             "#C8E6C9", "#388E3C", lw=1.2, r=0.06, z=2)

    txt(C0, ry, cfg,  "#888" if sep else "#37474F",
        fs=9 if not sep else 8, ha="left",
        bold=(best or sep), italic=sep)
    if not sep:
        gc = "#C62828" if gain.startswith("−") else "#2E7D32" if best else "#1565C0"
        txt(C1, ry, gain,  gc,  fs=9, ha="left", bold=best)
        txt(C2, ry, area,  "#37474F", fs=9, ha="left")
        txt(C3, ry, verd,  tc,  fs=9, ha="left", bold=best)

    ax.plot([TX+0.1, TX+TW-0.1], [ry - RH_T*0.50, ry - RH_T*0.50],
            color="#E0E0E0", lw=0.5, zorder=3)

# ════════════════════════════════════════════════════════════════════════
# Sub-header strip (sits between title banner and content)
# ════════════════════════════════════════════════════════════════════════
ax.add_patch(plt.Rectangle((0, SY), FW, SUB_H, fc="#E3EAF4", ec="none", zorder=6))
ax.plot([0, FW], [SY, SY], color="#1565C0", lw=1.0, zorder=6)

sub_labels = [
    ((Z_PIPE_L + Z_PIPE_R)/2,  "X86O3CPU Pipeline  (gem5 v24.1 SE)",       "#0D47A1"),
    ((Z_CACHE_L + Z_CACHE_R)/2, "Cache Hierarchy",                          "#1B5E20"),
    ((Z_HNSW_L + Z_TABLE_L)/2,  "HNSW Graph Traversal — Serial Pointer-Chasing  (MLP ≈ 1–2)", "#0D47A1"),
    ((Z_TABLE_L + Z_TABLE_R)/2, "All 22 Configurations",                    "#1B5E20"),
]
for sx, slbl, sc in sub_labels:
    txt(sx, SY + SUB_H/2, slbl, sc, fs=9, bold=True, z=7)

# ── save ──────────────────────────────────────────────────────────────────────
plt.tight_layout(pad=0.1)
plt.savefig(OUT, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved: {OUT}")
