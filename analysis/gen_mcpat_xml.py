#!/usr/bin/env python3
"""
gen_mcpat_xml.py  --  gem5 stats.txt -> McPAT XML
Uses Penryn.xml as structural template (known-good McPAT input).
Replaces stats and key params with gem5 measured values.

Usage:
    python3 analysis/gen_mcpat_xml.py \
        --stats results/hnsw_l2_256kB_roi/stats.txt \
        --out   mcpat_baseline.xml --l2-kb 256
    ./mcpat/mcpat -infile mcpat_baseline.xml -print_level 5
"""
import re, argparse

def get(txt, pat, default=0.0):
    m = re.search(pat, txt, re.MULTILINE)
    return float(m.group(1)) if m else default

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats",  required=True)
    ap.add_argument("--out",    required=True)
    ap.add_argument("--l2-kb",  type=int, default=256)
    ap.add_argument("--label",  default="HNSW gem5")
    args = ap.parse_args()

    txt = open(args.stats).read()

    # ── parse gem5 stats ──────────────────────────────────────────────────────
    sim_insts   = int(get(txt, r"simInsts\s+([\d.e+\-]+)"))
    num_cycles  = int(get(txt, r"board\.processor\.cores\.core\.numCycles\s+([\d.e+\-]+)"))
    idle_cycles = int(get(txt, r"board\.processor\.cores\.core\.idleCycles\s+([\d.e+\-]+)"))
    ipc         =     get(txt, r"board\.processor\.cores\.core\.ipc\s+([\d.e+\-]+)")
    int_alu     = int(get(txt, r"statIssuedInstType_0::IntAlu\s+([\d.e+\-]+)"))
    int_mult    = int(get(txt, r"statIssuedInstType_0::IntMult\s+([\d.e+\-]+)"))
    float_add   = int(get(txt, r"statIssuedInstType_0::FloatAdd\s+([\d.e+\-]+)"))
    float_mult  = int(get(txt, r"statIssuedInstType_0::FloatMult\s+([\d.e+\-]+)"))
    mem_read    = int(get(txt, r"statIssuedInstType_0::MemRead\s+([\d.e+\-]+)"))
    mem_write   = int(get(txt, r"statIssuedInstType_0::MemWrite\s+([\d.e+\-]+)"))
    fmem_read   = int(get(txt, r"statIssuedInstType_0::FloatMemRead\s+([\d.e+\-]+)"))
    fmem_write  = int(get(txt, r"statIssuedInstType_0::FloatMemWrite\s+([\d.e+\-]+)"))
    issued_tot  = int(get(txt, r"statIssuedInstType_0::total\s+([\d.e+\-]+)"))
    bp_lookups  = int(get(txt, r"branchPred\.lookups_0::total\s+([\d.e+\-]+)"))
    bp_miss     = int(get(txt, r"branchPred\.condIncorrect\s+([\d.e+\-]+)"))
    l1d_acc     = int(get(txt, r"l1d-cache-0\.overallAccesses::processor\.cores\.core\.data\s+([\d.e+\-]+)"))
    l1d_miss    = int(get(txt, r"l1d-cache-0\.overallMisses::processor\.cores\.core\.data\s+([\d.e+\-]+)"))
    l1i_acc     = int(get(txt, r"l1i-cache-0\.overallAccesses::processor\.cores\.core\.inst\s+([\d.e+\-]+)"))
    l1i_miss    = int(get(txt, r"l1i-cache-0\.overallMisses::processor\.cores\.core\.inst\s+([\d.e+\-]+)"))
    l2_acc      = int(get(txt, r"l2-cache-0\.overallAccesses::total\s+([\d.e+\-]+)"))
    l2_miss     = int(get(txt, r"l2-cache-0\.overallMisses::total\s+([\d.e+\-]+)"))
    dram_r      = int(get(txt, r"mem_ctrl\.readBursts\s+([\d.e+\-]+)"))
    dram_w      = int(get(txt, r"mem_ctrl\.writeBursts\s+([\d.e+\-]+)"))

    # ── derived ───────────────────────────────────────────────────────────────
    busy        = num_cycles - idle_cycles
    int_i       = int_alu + int_mult
    fp_i        = float_add + float_mult
    ld_i        = mem_read + fmem_read
    st_i        = mem_write + fmem_write
    duty        = round(ipc / 4.0, 4)
    rd_frac     = ld_i / max(ld_i + st_i, 1)
    l1d_racc    = int(l1d_acc  * rd_frac);  l1d_wacc  = l1d_acc  - l1d_racc
    l1d_rmiss   = int(l1d_miss * rd_frac);  l1d_wmiss = l1d_miss - l1d_rmiss
    l2_racc     = int(l2_acc  * 0.75);      l2_wacc   = l2_acc  - l2_racc
    l2_rmiss    = int(l2_miss * 0.75);      l2_wmiss  = l2_miss - l2_rmiss
    l2_bytes    = args.l2_kb * 1024

    print(f"Stats OK — cycles={num_cycles:,}  insts={sim_insts:,}  IPC={ipc:.3f}")

    # ── XML (Penryn structure, our stats) ─────────────────────────────────────
    xml = f"""<?xml version="1.0" ?>
<!-- McPAT input for: {args.label} | gem5 SE, X86O3CPU 4-wide OOO 3GHz, 45nm -->
<component id="root" name="root">
 <component id="system" name="system">
  <param name="number_of_cores"           value="1"/>
  <param name="number_of_L1Directories"   value="0"/>
  <param name="number_of_L2Directories"   value="0"/>
  <param name="number_of_L2s"             value="1"/>
  <param name="Private_L2"                value="0"/>
  <param name="number_of_L3s"             value="0"/>
  <param name="number_of_NoCs"            value="1"/>
  <param name="homogeneous_cores"         value="1"/>
  <param name="homogeneous_L2s"           value="1"/>
  <param name="homogeneous_L1Directories" value="1"/>
  <param name="homogeneous_L2Directories" value="1"/>
  <param name="homogeneous_L3s"           value="1"/>
  <param name="homogeneous_ccs"           value="1"/>
  <param name="homogeneous_NoCs"          value="1"/>
  <param name="core_tech_node"            value="45"/>
  <param name="target_core_clockrate"     value="3000"/>
  <param name="temperature"               value="360"/>
  <param name="number_cache_levels"       value="2"/>
  <param name="interconnect_projection_type" value="0"/>
  <param name="device_type"               value="0"/>
  <param name="longer_channel_device"     value="1"/>
  <param name="power_gating"              value="0"/>
  <param name="machine_bits"              value="64"/>
  <param name="virtual_address_width"     value="48"/>
  <param name="physical_address_width"    value="48"/>
  <param name="virtual_memory_page_size"  value="4096"/>
  <stat name="total_cycles" value="{num_cycles}"/>
  <stat name="idle_cycles"  value="{idle_cycles}"/>
  <stat name="busy_cycles"  value="{busy}"/>

  <component id="system.core0" name="core0">
   <param name="clock_rate"                    value="3000"/>
   <param name="vdd"                           value="0"/>
   <param name="opt_local"                     value="0"/>
   <param name="instruction_length"            value="32"/>
   <param name="opcode_width"                  value="16"/>
   <param name="x86"                           value="1"/>
   <param name="micro_opcode_width"            value="8"/>
   <param name="machine_type"                  value="0"/>
   <param name="number_hardware_threads"       value="1"/>
   <param name="fetch_width"                   value="4"/>
   <param name="number_instruction_fetch_ports" value="1"/>
   <param name="decode_width"                  value="4"/>
   <param name="issue_width"                   value="4"/>
   <param name="peak_issue_width"              value="4"/>
   <param name="commit_width"                  value="4"/>
   <param name="fp_issue_width"                value="2"/>
   <param name="prediction_width"              value="1"/>
   <param name="pipelines_per_core"            value="1,1"/>
   <param name="pipeline_depth"                value="14,14"/>
   <param name="ALU_per_core"                  value="4"/>
   <param name="MUL_per_core"                  value="1"/>
   <param name="FPU_per_core"                  value="2"/>
   <param name="instruction_buffer_size"       value="32"/>
   <param name="decoded_stream_buffer_size"    value="16"/>
   <param name="instruction_window_scheme"     value="0"/>
   <param name="instruction_window_size"       value="64"/>
   <param name="fp_instruction_window_size"    value="32"/>
   <param name="ROB_size"                      value="128"/>
   <param name="archi_Regs_IRF_size"           value="16"/>
   <param name="archi_Regs_FRF_size"           value="32"/>
   <param name="phy_Regs_IRF_size"             value="256"/>
   <param name="phy_Regs_FRF_size"             value="256"/>
   <param name="rename_scheme"                 value="0"/>
   <param name="checkpoint_depth"              value="1"/>
   <param name="register_windows_size"         value="0"/>
   <param name="LSU_order"                     value="inorder"/>
   <param name="store_buffer_size"             value="32"/>
   <param name="load_buffer_size"              value="32"/>
   <param name="memory_ports"                  value="2"/>
   <param name="RAS_size"                      value="64"/>
   <stat name="total_instructions"       value="{sim_insts}"/>
   <stat name="int_instructions"         value="{int_i}"/>
   <stat name="fp_instructions"          value="{fp_i}"/>
   <stat name="branch_instructions"      value="{bp_lookups}"/>
   <stat name="branch_mispredictions"    value="{bp_miss}"/>
   <stat name="load_instructions"        value="{ld_i}"/>
   <stat name="store_instructions"       value="{st_i}"/>
   <stat name="committed_instructions"   value="{sim_insts}"/>
   <stat name="committed_int_instructions" value="{int_i}"/>
   <stat name="committed_fp_instructions"  value="{fp_i}"/>
   <stat name="pipeline_duty_cycle"      value="{duty}"/>
   <stat name="total_cycles"             value="{num_cycles}"/>
   <stat name="idle_cycles"              value="{idle_cycles}"/>
   <stat name="busy_cycles"              value="{busy}"/>
   <stat name="ROB_reads"               value="{sim_insts*2}"/>
   <stat name="ROB_writes"              value="{sim_insts}"/>
   <stat name="rename_reads"            value="{int_i*3}"/>
   <stat name="rename_writes"           value="{int_i}"/>
   <stat name="fp_rename_reads"         value="{fp_i*2}"/>
   <stat name="fp_rename_writes"        value="{fp_i}"/>
   <stat name="inst_window_reads"       value="{issued_tot}"/>
   <stat name="inst_window_writes"      value="{issued_tot}"/>
   <stat name="inst_window_wakeup_accesses" value="{issued_tot*2}"/>
   <stat name="fp_inst_window_reads"    value="{fp_i}"/>
   <stat name="fp_inst_window_writes"   value="{fp_i}"/>
   <stat name="fp_inst_window_wakeup_accesses" value="{fp_i*2}"/>
   <stat name="int_regfile_reads"       value="{int_i*2}"/>
   <stat name="float_regfile_reads"     value="{fp_i*2}"/>
   <stat name="int_regfile_writes"      value="{int_i}"/>
   <stat name="float_regfile_writes"    value="{fp_i}"/>
   <stat name="function_calls"          value="0"/>
   <stat name="context_switches"        value="0"/>
   <stat name="ialu_accesses"           value="{int_alu}"/>
   <stat name="fpu_accesses"            value="{fp_i}"/>
   <stat name="mul_accesses"            value="{int_mult}"/>
   <stat name="cdb_alu_accesses"        value="{int_alu}"/>
   <stat name="cdb_mul_accesses"        value="{int_mult}"/>
   <stat name="cdb_fpu_accesses"        value="{fp_i}"/>
   <stat name="IFU_duty_cycle"          value="0.25"/>
   <stat name="LSU_duty_cycle"          value="0.25"/>
   <stat name="MemManU_I_duty_cycle"    value="0.25"/>
   <stat name="MemManU_D_duty_cycle"    value="0.25"/>
   <stat name="ALU_duty_cycle"          value="1"/>
   <stat name="MUL_duty_cycle"          value="0.3"/>
   <stat name="FPU_duty_cycle"          value="0.3"/>
   <stat name="ALU_cdb_duty_cycle"      value="1"/>
   <stat name="MUL_cdb_duty_cycle"      value="0.3"/>
   <stat name="FPU_cdb_duty_cycle"      value="0.3"/>
   <param name="number_of_BPT" value="2"/>
   <component id="system.core0.predictor" name="PBT">
    <param name="local_predictor_size"      value="10,3"/>
    <param name="local_predictor_entries"   value="1024"/>
    <param name="global_predictor_entries"  value="4096"/>
    <param name="global_predictor_bits"     value="2"/>
    <param name="chooser_predictor_entries" value="4096"/>
    <param name="chooser_predictor_bits"    value="2"/>
   </component>
   <component id="system.core0.itlb" name="itlb">
    <param name="number_entries" value="64"/>
    <stat name="total_accesses"  value="{l1i_acc}"/>
    <stat name="total_misses"    value="{l1i_miss}"/>
    <stat name="conflicts"       value="0"/>
   </component>
   <component id="system.core0.icache" name="icache">
    <param name="icache_config" value="32768,64,8,1,1,4,64,0"/>
    <param name="buffer_sizes"  value="16,16,16,0"/>
    <stat name="read_accesses"  value="{l1i_acc}"/>
    <stat name="read_misses"    value="{l1i_miss}"/>
    <stat name="conflicts"      value="0"/>
   </component>
   <component id="system.core0.dtlb" name="dtlb">
    <param name="number_entries" value="64"/>
    <stat name="total_accesses"  value="{l1d_acc}"/>
    <stat name="total_misses"    value="{l1d_miss}"/>
    <stat name="conflicts"       value="0"/>
   </component>
   <component id="system.core0.dcache" name="dcache">
    <param name="dcache_config" value="32768,64,8,1,1,4,64,1"/>
    <param name="buffer_sizes"  value="16,16,16,16"/>
    <stat name="read_accesses"  value="{l1d_racc}"/>
    <stat name="write_accesses" value="{l1d_wacc}"/>
    <stat name="read_misses"    value="{l1d_rmiss}"/>
    <stat name="write_misses"   value="{l1d_wmiss}"/>
    <stat name="conflicts"      value="0"/>
   </component>
   <param name="number_of_BTB" value="2"/>
   <component id="system.core0.BTB" name="BTB">
    <param name="BTB_config"    value="4096,4,2,1,1,3"/>
    <stat name="read_accesses"  value="{bp_lookups}"/>
    <stat name="write_accesses" value="{bp_miss}"/>
   </component>
  </component>

  <component id="system.L1Directory0" name="L1Directory0">
   <param name="Directory_type" value="0"/>
   <param name="Dir_config"     value="4096,2,0,1,100,100,8"/>
   <param name="buffer_sizes"   value="8,8,8,8"/>
   <param name="clockrate"      value="3000"/>
   <param name="vdd"            value="0"/>
   <param name="ports"          value="1,1,1"/>
   <param name="device_type"    value="0"/>
   <stat name="read_accesses"   value="0"/>
   <stat name="write_accesses"  value="0"/>
   <stat name="read_misses"     value="0"/>
   <stat name="write_misses"    value="0"/>
   <stat name="conflicts"       value="0"/>
  </component>

  <component id="system.L2Directory0" name="L2Directory0">
   <param name="Directory_type" value="1"/>
   <param name="Dir_config"     value="1048576,16,16,1,2,100"/>
   <param name="buffer_sizes"   value="8,8,8,8"/>
   <param name="clockrate"      value="3000"/>
   <param name="vdd"            value="0"/>
   <param name="ports"          value="1,1,1"/>
   <param name="device_type"    value="0"/>
   <stat name="read_accesses"   value="0"/>
   <stat name="write_accesses"  value="0"/>
   <stat name="read_misses"     value="0"/>
   <stat name="write_misses"    value="0"/>
   <stat name="conflicts"       value="0"/>
  </component>

  <component id="system.L20" name="L20">
   <param name="L2_config"    value="{l2_bytes},64,16,1,8,20,64,1"/>
   <param name="buffer_sizes" value="16,16,16,16"/>
   <param name="clockrate"    value="3000"/>
   <param name="vdd"          value="0"/>
   <param name="ports"        value="1,1,1"/>
   <param name="device_type"  value="0"/>
   <stat name="read_accesses"  value="{l2_racc}"/>
   <stat name="write_accesses" value="{l2_wacc}"/>
   <stat name="read_misses"    value="{l2_rmiss}"/>
   <stat name="write_misses"   value="{l2_wmiss}"/>
   <stat name="conflicts"      value="0"/>
   <stat name="duty_cycle"     value="0.5"/>
  </component>

  <component id="system.L30" name="L30">
   <param name="L3_config"    value="16777216,64,16,16,16,100,1"/>
   <param name="clockrate"    value="3000"/>
   <param name="ports"        value="1,1,1"/>
   <param name="device_type"  value="0"/>
   <param name="vdd"          value="0"/>
   <param name="buffer_sizes" value="16,16,16,16"/>
   <stat name="read_accesses"  value="0"/>
   <stat name="write_accesses" value="0"/>
   <stat name="read_misses"    value="0"/>
   <stat name="write_misses"   value="0"/>
   <stat name="conflicts"      value="0"/>
   <stat name="duty_cycle"     value="1.0"/>
  </component>

  <component id="system.NoC0" name="noc0">
   <param name="clockrate"        value="3000"/>
   <param name="vdd"              value="0"/>
   <param name="type"             value="0"/>
   <param name="horizontal_nodes" value="1"/>
   <param name="vertical_nodes"   value="1"/>
   <param name="has_global_link"  value="0"/>
   <param name="link_throughput"  value="1"/>
   <param name="link_latency"     value="1"/>
   <param name="input_ports"      value="1"/>
   <param name="output_ports"     value="1"/>
   <param name="flit_bits"        value="256"/>
   <param name="chip_coverage"    value="1"/>
   <param name="link_routing_over_percentage" value="0.5"/>
   <stat name="total_accesses"    value="{l2_miss}"/>
   <stat name="duty_cycle"        value="1"/>
  </component>

  <component id="system.mc" name="mc">
   <param name="type"                        value="0"/>
   <param name="mc_clock"                    value="300"/>
   <param name="vdd"                         value="0"/>
   <param name="peak_transfer_rate"          value="19200"/>
   <param name="block_size"                  value="64"/>
   <param name="number_mcs"                  value="1"/>
   <param name="memory_channels_per_mc"      value="1"/>
   <param name="number_ranks"                value="1"/>
   <param name="withPHY"                     value="0"/>
   <param name="req_window_size_per_channel" value="32"/>
   <param name="IO_buffer_size_per_channel"  value="32"/>
   <param name="databus_width"               value="64"/>
   <param name="addressbus_width"            value="48"/>
   <stat name="memory_accesses" value="{dram_r+dram_w}"/>
   <stat name="memory_reads"    value="{dram_r}"/>
   <stat name="memory_writes"   value="{dram_w}"/>
  </component>

  <component id="system.niu" name="niu">
   <param name="type" value="0"/><param name="clockrate" value="350"/>
   <param name="vdd" value="0"/><param name="number_units" value="0"/>
   <stat name="duty_cycle" value="1.0"/><stat name="total_load_perc" value="0.7"/>
  </component>
  <component id="system.pcie" name="pcie">
   <param name="type" value="0"/><param name="withPHY" value="1"/>
   <param name="clockrate" value="350"/><param name="vdd" value="0"/>
   <param name="number_units" value="0"/><param name="num_channels" value="8"/>
   <stat name="duty_cycle" value="1.0"/><stat name="total_load_perc" value="0.7"/>
  </component>
  <component id="system.flashc" name="flashc">
   <param name="number_flashcs" value="0"/><param name="type" value="1"/>
   <param name="withPHY" value="1"/><param name="peak_transfer_rate" value="200"/>
   <param name="vdd" value="0"/>
   <stat name="duty_cycle" value="1.0"/><stat name="total_load_perc" value="0.7"/>
  </component>

 </component>
</component>
"""
    with open(args.out, "w") as f:
        f.write(xml)
    print(f"Written: {args.out}")
    print(f"Run: /workspace/mcpat/mcpat -infile {args.out} -print_level 5")

if __name__ == "__main__":
    main()
