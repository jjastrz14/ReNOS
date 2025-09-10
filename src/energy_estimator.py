#!/usr/bin/env python3
"""
Energy estimator that adds SRAM reads/writes and reply flits.

- For each WRITE:
    * keeps the original WRITE accounting (network/memory)
    * adds a SRAM write of the same size
    * adds a REPLY packet sized 1 flit (flit size read from arch; field in bits -> converted to bytes)
- For each COMP_OP:
    * keeps the original COMP_OP accounting (flops)
    * adds a SRAM read of the COMP_OP 'size'

Usage:
  python energy_estimator.py path/to/workload.json
  cat workload.json | python energy_estimator.py -
"""

import json
import sys
import argparse
import pandas as pd
from collections import defaultdict
import os

# --- Default energy parameters (Joules) - change to your calibrated values ---
energy_params = {
    "COMP_OP":   {"energy_per_flop": 1e-12},   # 1 pJ per flop (example)
    "WRITE":     {"energy_per_byte": 5e-12},   # 5 pJ per byte (example)
    "WRITE_REQ": {"energy_per_byte": 5e-12},   # same as WRITE by default
    # added types for SRAM and reply
    "SRAM_READ":  {"energy_per_byte": 2e-12},  # example: 2 pJ/byte
    "SRAM_WRITE": {"energy_per_byte": 3e-12},  # example: 3 pJ/byte
    "REPLY":      {"energy_per_byte": 5e-12},  # treat reply as network byte cost by default
    # fallback DEFAULT if needed
    "DEFAULT":    {"energy_per_byte": 5e-12, "energy_per_flop": 1e-12}
}

def load_json_from_path(path):
    """Load JSON from file path or '-' for stdin."""
    try:
        if path == '-' or path is None:
            return json.load(sys.stdin)
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading JSON from '{path}': {e}", file=sys.stderr)
        sys.exit(2)

def aggregate_and_estimate(data, energy_params):
    agg = defaultdict(lambda: {"count":0, "total_bytes":0, "total_flops":0})

    workload = data.get("workload", [])
    if not isinstance(workload, list):
        print("Input JSON does not contain a 'workload' array.", file=sys.stderr)
        sys.exit(3)

    # determine flit size (bits) -> convert to bytes (ceiling)
    arch = data.get("arch", {}) or {}
    flit_bits = None
    # try common keys
    for key in ("flit_size", "width_phit"):
        if key in arch:
            try:
                flit_bits = int(arch[key])
                break
            except Exception:
                pass
    if flit_bits is None:
        print("Warning: flit size not found in arch (keys 'flit_size' or 'width_phit'). REPLY bytes set to 0.", file=sys.stderr)
        reply_bytes = 0
    else:
        reply_bytes = (flit_bits + 7) // 8  # ceil bits->bytes

    for entry in workload:
        typ = entry.get("type", "UNKNOWN")
        agg[typ]["count"] += 1
        # size field assumed bytes
        try:
            size_val = int(entry.get("size", 0) or 0)
        except Exception:
            size_val = 0
        agg[typ]["total_bytes"] += size_val

        # sum ct_required and pt_required if present (flops)
        flops = 0
        for key in ("ct_required", "pt_required"):
            if key in entry and entry[key] is not None:
                try:
                    flops += int(entry[key])
                except Exception:
                    pass
        agg[typ]["total_flops"] += flops

        # --- additional synthetic events per your request ---
        if typ == "WRITE":
            # add a SRAM write of the same size
            agg["SRAM_WRITE"]["count"] += 1
            agg["SRAM_WRITE"]["total_bytes"] += size_val
            # add a REPLY packet of size 1 flit (converted to bytes)
            if reply_bytes > 0:
                agg["REPLY"]["count"] += 1
                agg["REPLY"]["total_bytes"] += reply_bytes
        elif typ == "COMP_OP":
            # add a SRAM read of the COMP_OP 'size'
            agg["SRAM_READ"]["count"] += 1
            agg["SRAM_READ"]["total_bytes"] += size_val

    # compute energy
    rows = []
    total_energy = 0.0
    for typ, vals in agg.items():
        bytes_ = vals["total_bytes"]
        flops_ = vals["total_flops"]
        energy_from_bytes = 0.0
        energy_from_flops = 0.0

        params = energy_params.get(typ, None)
        if params is None:
            params = energy_params.get("DEFAULT", {})

        if "energy_per_byte" in params and bytes_:
            energy_from_bytes = bytes_ * params["energy_per_byte"]
        if "energy_per_flop" in params and flops_:
            energy_from_flops = flops_ * params["energy_per_flop"]

        # extra fallback (if DEFAULT present)
        if energy_from_bytes == 0.0 and "energy_per_byte" in energy_params.get("DEFAULT", {}):
            energy_from_bytes = bytes_ * energy_params["DEFAULT"]["energy_per_byte"]
        if energy_from_flops == 0.0 and "energy_per_flop" in energy_params.get("DEFAULT", {}):
            energy_from_flops = flops_ * energy_params["DEFAULT"]["energy_per_flop"]

        energy = energy_from_bytes + energy_from_flops
        total_energy += energy
        rows.append({
            "type": typ,
            "count": vals["count"],
            "total_bytes": bytes_,
            "total_flops": flops_,
            "energy_bytes_J": energy_from_bytes,
            "energy_flops_J": energy_from_flops,
            "energy_total_J": energy
        })

    df = pd.DataFrame(rows).sort_values("type").reset_index(drop=True)
    return df, total_energy

def main():
    parser = argparse.ArgumentParser(description="Estimate energy from a workload JSON.")
    parser.add_argument("input", nargs="?", default="-",
                        help="Path to workload JSON file, or '-' to read from stdin (default '-')")
    parser.add_argument("--out", default="./energy_estimate_breakdown.csv",
                        help="CSV output path (default ./energy_estimate_breakdown.csv)")
    args = parser.parse_args()

    data = load_json_from_path(args.input)
    df, total_energy = aggregate_and_estimate(data, energy_params)

    # Save CSV and print summary
    #out_path = args.out
    #try:
    #    df.to_csv(out_path, index=False)
    #except Exception as e:
    #    print(f"Warning: failed to write CSV to {out_path}: {e}", file=sys.stderr)

    pd.set_option("display.float_format", "{:.6e}".format)
    print("\nSUMMARY:")
    if df.empty:
        print("No workload entries found.")
    else:
        print(df.to_string(index=False))
    print(f"\nTOTAL estimated energy = {total_energy:.6e} J  ({total_energy*1e6:.3f} microjoules)")
    #print(f"\nCSV breakdown saved to: {os.path.abspath(out_path)}")

if __name__ == "__main__":
    main()
