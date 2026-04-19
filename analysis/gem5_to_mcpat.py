"""
gem5_to_mcpat.py  --  Convert gem5 stats.txt to McPAT XML input
                      then run McPAT and report Power + Area

Usage:
    python3 gem5_to_mcpat.py --stats m5out/stats.txt --l2-size 256
    python3 gem5_to_mcpat.py --stats m5out/stats.txt --l2-size 512
    python3 gem5_to_mcpat.py --sweep-dir /workspace/results

McPAT binary expected at: /opt/gem5/build/mcpat/mcpat
"""

import argparse
import os
import re
import subprocess
import sys

MCPAT_BIN = "/opt/gem5/build/mcpat/mcpat"

# ---------------------------------------------------------------------------
# Parse key values from gem5 stats.txt
# ---------------------------------------------------------------------------

def parse_gem5_stats(stats_path):
    with open(stats_path) as f:
        text = f.read()

    def get(pattern, default=0):
        m = re.search(pattern, text, re.MULTILINE)
        return float(m.group(1)) if m else default

    return {
        "sim_seconds":   get(r"simSeconds\s+([\d.e+\-]+)"),
        "sim_insts":     get(r"simInsts\s+([\d.e+]+)"),
        "sim_ticks":     get(r"simTicks\s+([\d.e+]+)"),
        "l1d_hits":      get(r"board\.cache_hierarchy\.l1d-cache-0\.overallHits::total\s+([\d.e+]+)"),
        "l1d_misses":    get(r"board\.cache_hierarchy\.l1d-cache-0\.overallMisses::total\s+([\d.e+]+)"),
        "l1i_hits":      get(r"board\.cache_hierarchy\.l1i-cache-0\.overallHits::total\s+([\d.e+]+)"),
        "l1i_misses":    get(r"board\.cache_hierarchy\.l1i-cache-0\.overallMisses::total\s+([\d.e+]+)"),
        "l2_hits":       get(r"board\.cache_hierarchy\.l2-cache-0\.overallHits::total\s+([\d.e+]+)"),
        "l2_misses":     get(r"board\.cache_hierarchy\.l2-cache-0\.overallMisses::total\s+([\d.e+]+)"),
    }

# ---------------------------------------------------------------------------
# Generate McPAT XML for a simple in-order x86 core + cache hierarchy
# Based on the McPAT Xeon template, simplified for TimingSimpleCPU
# ---------------------------------------------------------------------------

def gen_mcpat_xml(stats, l2_size_kb=256):
    sim_sec  = max(stats["sim_seconds"], 1e-9)
    sim_inst = max(stats["sim_insts"], 1)

    # Derived rates (per second)
    l1d_acc  = (stats["l1d_hits"] + stats["l1d_misses"]) / sim_sec
    l1i_acc  = (stats["l1i_hits"] + stats["l1i_misses"]) / sim_sec
    l2_acc   = (stats["l2_hits"]  + stats["l2_misses"])  / sim_sec
    l2_miss  = stats["l2_misses"] / sim_sec

    return f"""<?xml version="1.0" ?>
<component id="root" name="root">
<component id="system" name="system">
  <param name="number_of_processors" value="1"/>
  <param name="number_of_L1Directories" value="0"/>
  <param name="number_of_L2Directories" value="0"/>
  <param name="number_of_L2s" value="1"/>
  <param name="Private_L2" value="1"/>
  <param name="number_of_L3s" value="0"/>
  <param name="number_of_NoCs" value="0"/>
  <param name="homogeneous_L2s" value="1"/>
  <param name="homogeneous_L3s" value="1"/>
  <param name="core_tech_node" value="22"/>
  <param name="target_core_clockrate" value="3000"/>
  <param name="temperature" value="340"/>
  <param name="number_cache_levels" value="2"/>
  <param name="interconnect_projection_type" value="0"/>
  <param name="device_type" value="0"/>
  <param name="longer_channel_device" value="1"/>
  <param name="Embedded" value="0"/>
  <param name="machine_bits" value="64"/>
  <param name="virtual_address_width" value="48"/>
  <param name="physical_address_width" value="48"/>
  <param name="virtual_memory_page_size" value="4096"/>
  <stat name="total_cycles"          value="{int(stats['sim_ticks'] / 333)}"/>
  <stat name="idle_cycles"           value="0"/>
  <stat name="busy_cycles"           value="{int(stats['sim_ticks'] / 333)}"/>

  <component id="system.core0" name="core0">
    <param name="clock_rate" value="3000"/>
    <param name="vdd" value="0"/>
    <param name="power_gating_vcc" value="-1"/>
    <param name="opt_local" value="0"/>
    <param name="instruction_length" value="32"/>
    <param name="opcode_width" value="7"/>
    <param name="x86" value="1"/>
    <param name="micro_opcode_width" value="8"/>
    <param name="machine_type" value="1"/>
    <param name="number_hardware_threads" value="1"/>
    <param name="fetch_width" value="1"/>
    <param name="number_instruction_fetch_ports" value="1"/>
    <param name="decode_width" value="1"/>
    <param name="execute_width" value="1"/>
    <param name="peak_issue_width" value="1"/>
    <param name="commit_width" value="1"/>
    <param name="fp_issue_width" value="1"/>
    <param name="prediction_width" value="0"/>
    <param name="pipelines_per_core" value="1,1"/>
    <param name="pipeline_depth" value="5,5"/>
    <param name="ALU_per_core" value="1"/>
    <param name="MUL_per_core" value="1"/>
    <param name="FPU_per_core" value="1"/>
    <param name="instruction_buffer_size" value="16"/>
    <param name="decoded_stream_buffer_size" value="16"/>
    <param name="instruction_window_scheme" value="0"/>
    <param name="instruction_window_size" value="16"/>
    <param name="fp_instruction_window_size" value="16"/>
    <param name="ROB_size" value="16"/>
    <param name="archi_Regs_IRF_size" value="16"/>
    <param name="archi_Regs_FRF_size" value="32"/>
    <param name="phy_Regs_IRF_size" value="16"/>
    <param name="phy_Regs_FRF_size" value="32"/>
    <param name="rename_scheme" value="0"/>
    <param name="register_windows_size" value="0"/>
    <param name="LSU_order" value="inorder"/>
    <param name="store_buffer_size" value="4"/>
    <param name="load_buffer_size" value="4"/>
    <param name="memory_ports" value="1"/>
    <param name="RAS_size" value="8"/>
    <stat name="total_instructions"        value="{int(sim_inst)}"/>
    <stat name="int_instructions"          value="{int(sim_inst * 0.6)}"/>
    <stat name="fp_instructions"           value="{int(sim_inst * 0.1)}"/>
    <stat name="branch_instructions"       value="{int(sim_inst * 0.1)}"/>
    <stat name="branch_mispredictions"     value="{int(sim_inst * 0.01)}"/>
    <stat name="load_instructions"         value="{int(sim_inst * 0.25)}"/>
    <stat name="store_instructions"        value="{int(sim_inst * 0.1)}"/>
    <stat name="committed_instructions"    value="{int(sim_inst)}"/>
    <stat name="committed_int_instructions" value="{int(sim_inst * 0.6)}"/>
    <stat name="committed_fp_instructions"  value="{int(sim_inst * 0.1)}"/>
    <stat name="pipeline_duty_cycle"       value="1"/>
    <stat name="total_cycles"              value="{int(stats['sim_ticks'] / 333)}"/>
    <stat name="busy_cycles"               value="{int(stats['sim_ticks'] / 333)}"/>
    <stat name="idle_cycles"               value="0"/>
    <stat name="instruction_buffer_reads"  value="{int(sim_inst)}"/>
    <stat name="instruction_buffer_write"  value="{int(sim_inst)}"/>
    <stat name="ROB_reads"                 value="0"/>
    <stat name="ROB_writes"                value="0"/>
    <stat name="rename_reads"              value="0"/>
    <stat name="rename_writes"             value="0"/>
    <stat name="fp_rename_reads"           value="0"/>
    <stat name="fp_rename_writes"          value="0"/>
    <stat name="inst_window_reads"         value="0"/>
    <stat name="inst_window_writes"        value="0"/>
    <stat name="inst_window_wakeup_accesses" value="0"/>
    <stat name="fp_inst_window_reads"      value="0"/>
    <stat name="fp_inst_window_writes"     value="0"/>
    <stat name="fp_inst_window_wakeup_accesses" value="0"/>
    <stat name="int_regfile_reads"         value="{int(sim_inst * 0.5)}"/>
    <stat name="float_regfile_reads"       value="{int(sim_inst * 0.1)}"/>
    <stat name="int_regfile_writes"        value="{int(sim_inst * 0.4)}"/>
    <stat name="float_regfile_writes"      value="{int(sim_inst * 0.05)}"/>
    <stat name="function_calls"            value="{int(sim_inst * 0.01)}"/>
    <stat name="context_switches"          value="0"/>
    <stat name="ialu_accesses"             value="{int(sim_inst * 0.6)}"/>
    <stat name="fpu_accesses"              value="{int(sim_inst * 0.1)}"/>
    <stat name="mul_accesses"              value="{int(sim_inst * 0.02)}"/>
    <stat name="cdb_alu_accesses"          value="0"/>
    <stat name="cdb_mul_accesses"          value="0"/>
    <stat name="cdb_fpu_accesses"          value="0"/>
    <stat name="load_buffer_reads"         value="{int(sim_inst * 0.25)}"/>
    <stat name="load_buffer_writes"        value="{int(sim_inst * 0.25)}"/>
    <stat name="load_buffer_cams"          value="0"/>
    <stat name="store_buffer_reads"        value="{int(sim_inst * 0.1)}"/>
    <stat name="store_buffer_writes"       value="{int(sim_inst * 0.1)}"/>
    <stat name="store_buffer_cams"         value="0"/>
    <stat name="store_buffer_forwards"     value="0"/>
    <stat name="main_memory_access"        value="{int(l2_miss)}"/>
    <stat name="main_memory_read"          value="{int(l2_miss * 0.7)}"/>
    <stat name="main_memory_write"         value="{int(l2_miss * 0.3)}"/>

    <component id="system.core0.itlb" name="itlb">
      <param name="number_entries" value="64"/>
      <stat name="total_accesses" value="{int(l1i_acc)}"/>
      <stat name="total_misses"   value="10"/>
      <stat name="conflicts"      value="0"/>
    </component>
    <component id="system.core0.icache" name="icache">
      <param name="icache_config" value="32768,32,8,1,1,3,1,0"/>
      <param name="buffer_sizes"  value="4,4,4,4"/>
      <stat name="read_accesses"  value="{int(l1i_acc)}"/>
      <stat name="read_misses"    value="{int(stats['l1i_misses'])}"/>
      <stat name="conflicts"      value="0"/>
    </component>
    <component id="system.core0.dtlb" name="dtlb">
      <param name="number_entries" value="64"/>
      <stat name="total_accesses" value="{int(l1d_acc)}"/>
      <stat name="total_misses"   value="10"/>
      <stat name="conflicts"      value="0"/>
    </component>
    <component id="system.core0.dcache" name="dcache">
      <param name="dcache_config" value="32768,32,8,1,1,3,1,0"/>
      <param name="buffer_sizes"  value="4,4,4,4"/>
      <stat name="read_accesses"  value="{int(l1d_acc * 0.7)}"/>
      <stat name="write_accesses" value="{int(l1d_acc * 0.3)}"/>
      <stat name="read_misses"    value="{int(stats['l1d_misses'] * 0.7)}"/>
      <stat name="write_misses"   value="{int(stats['l1d_misses'] * 0.3)}"/>
      <stat name="conflicts"      value="0"/>
    </component>
    <component id="system.core0.BTB" name="BTB">
      <param name="BTB_config" value="4096,4,2,1,1,3"/>
      <stat name="read_accesses"  value="{int(sim_inst * 0.1)}"/>
      <stat name="write_accesses" value="{int(sim_inst * 0.01)}"/>
    </component>
  </component>

  <component id="system.L20" name="L20">
    <param name="L2_config" value="{l2_size_kb * 1024},32,8,1,1,6,1,0"/>
    <param name="buffer_sizes" value="8,8,8,8"/>
    <param name="clockrate"    value="3000"/>
    <param name="vdd"          value="0"/>
    <param name="power_gating_vcc" value="-1"/>
    <param name="ports"        value="1,1,1"/>
    <param name="device_type"  value="0"/>
    <stat name="read_accesses"  value="{int(l2_acc * 0.7)}"/>
    <stat name="write_accesses" value="{int(l2_acc * 0.3)}"/>
    <stat name="read_misses"    value="{int(stats['l2_misses'] * 0.7)}"/>
    <stat name="write_misses"   value="{int(stats['l2_misses'] * 0.3)}"/>
    <stat name="conflicts"      value="0"/>
    <stat name="duty_cycle"     value="1"/>
  </component>

  <component id="system.mc" name="mc">
    <param name="mc_clock"              value="200"/>
    <param name="vdd"                   value="0"/>
    <param name="power_gating_vcc"      value="-1"/>
    <param name="peak_transfer_rate"    value="3200"/>
    <param name="block_size"            value="64"/>
    <param name="number_mcs"            value="1"/>
    <param name="memory_channels_per_mc" value="1"/>
    <param name="number_ranks"          value="1"/>
    <param name="withPHY"               value="0"/>
    <param name="req_window_size_per_channel" value="16"/>
    <param name="IO_buffer_size_per_channel"  value="16"/>
    <param name="databus_width"         value="128"/>
    <param name="addressbus_width"      value="48"/>
    <stat name="memory_accesses"        value="{int(l2_miss)}"/>
    <stat name="memory_reads"           value="{int(l2_miss * 0.7)}"/>
    <stat name="memory_writes"          value="{int(l2_miss * 0.3)}"/>
  </component>

  <component id="system.niu" name="niu">
    <param name="type" value="0"/>
    <param name="clockrate" value="3000"/>
    <param name="vdd" value="0"/>
    <param name="power_gating_vcc" value="-1"/>
    <stat name="duty_cycle" value="0"/>
    <stat name="total_load_perc" value="0"/>
  </component>
  <component id="system.pcie" name="pcie">
    <param name="type" value="0"/>
    <param name="withPHY" value="0"/>
    <param name="clockrate" value="3000"/>
    <param name="vdd" value="0"/>
    <param name="power_gating_vcc" value="-1"/>
    <param name="num_channels" value="1"/>
    <stat name="duty_cycle" value="0"/>
    <stat name="total_load_perc" value="0"/>
  </component>
  <component id="system.flashc" name="flashc">
    <param name="number_flashcs" value="0"/>
    <param name="type" value="1"/>
    <param name="withPHY" value="0"/>
    <param name="peak_transfer_rate" value="200"/>
    <param name="vdd" value="0"/>
    <param name="power_gating_vcc" value="-1"/>
    <stat name="duty_cycle" value="0"/>
    <stat name="total_load_perc" value="0"/>
  </component>

</component>
</component>
"""

# ---------------------------------------------------------------------------
# Run McPAT and parse output
# ---------------------------------------------------------------------------

def run_mcpat(xml_path):
    result = subprocess.run(
        [MCPAT_BIN, "-infile", xml_path, "-print_level", "5"],
        capture_output=True, text=True
    )
    out = result.stdout

    def get(pattern):
        m = re.search(pattern, out)
        return float(m.group(1)) if m else None

    return {
        "total_power_w":    get(r"Total Leakage = ([\d.]+)") ,
        "dynamic_power_w":  get(r"Runtime Dynamic = ([\d.]+)"),
        "subthreshold_w":   get(r"Subthreshold Leakage = ([\d.]+)"),
        "area_mm2":         get(r"Area = ([\d.]+)"),
        "raw":              out,
    }

def print_ppa(label, ppa):
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    if ppa["dynamic_power_w"] is not None:
        print(f"  Dynamic Power     {ppa['dynamic_power_w']:>10.4f} W")
    if ppa["subthreshold_w"] is not None:
        print(f"  Leakage Power     {ppa['subthreshold_w']:>10.4f} W")
    if ppa["area_mm2"] is not None:
        print(f"  Area              {ppa['area_mm2']:>10.4f} mm²")
    print(f"{'='*50}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_one(stats_path, l2_kb, label=None):
    stats   = parse_gem5_stats(stats_path)
    xml     = gen_mcpat_xml(stats, l2_size_kb=l2_kb)
    xml_path = stats_path.replace("stats.txt", "mcpat_input.xml")
    with open(xml_path, "w") as f:
        f.write(xml)
    ppa = run_mcpat(xml_path)
    print_ppa(label or f"L2={l2_kb}kB", ppa)
    return ppa

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats",     help="Path to single stats.txt")
    parser.add_argument("--l2-size",   type=int, default=256, help="L2 size in kB")
    parser.add_argument("--sweep-dir", help="Directory with hnsw_l2_* subdirs")
    args = parser.parse_args()

    if args.stats:
        run_one(args.stats, args.l2_size)

    elif args.sweep_dir:
        size_map = {"256kB": 256, "512kB": 512, "1MB": 1024, "2MB": 2048}
        print(f"\n{'Config':<20} {'Dyn Power (W)':<18} {'Leakage (W)':<15} {'Area (mm²)'}")
        print("-" * 65)
        for entry in sorted(os.listdir(args.sweep_dir)):
            stats_path = os.path.join(args.sweep_dir, entry, "stats.txt")
            if not os.path.isfile(stats_path):
                continue
            l2_kb = next((v for k, v in size_map.items() if k in entry), 256)
            ppa = run_one(stats_path, l2_kb, label=entry)
            dyn = ppa["dynamic_power_w"] or 0
            leak = ppa["subthreshold_w"] or 0
            area = ppa["area_mm2"] or 0
            print(f"  {entry:<18} {dyn:<18.4f} {leak:<15.4f} {area:.4f}")
    else:
        print("Usage: python3 gem5_to_mcpat.py --sweep-dir /workspace/results")
