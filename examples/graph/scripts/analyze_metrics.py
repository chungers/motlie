#!/usr/bin/env python3
"""
Analyze performance metrics from graph algorithm demos.
Reads CSV data and generates summary tables and findings.
"""

import pandas as pd
import sys
from pathlib import Path

def load_data(csv_file):
    """Load performance metrics from CSV file."""
    try:
        df = pd.read_csv(csv_file)
        return df
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        sys.exit(1)

def check_correctness(df):
    """Check if result hashes match between implementations."""
    print("=" * 80)
    print("CORRECTNESS VERIFICATION")
    print("=" * 80)
    print()

    issues = []
    for algo in df['algorithm'].unique():
        for scale in sorted(df['scale'].unique()):
            subset = df[(df['algorithm'] == algo) & (df['scale'] == scale)]

            if len(subset) < 2:
                continue

            ref_hash = subset[subset['implementation'] == 'reference']['result_hash'].values
            motlie_hash = subset[subset['implementation'] == 'motlie_db']['result_hash'].values

            if len(ref_hash) > 0 and len(motlie_hash) > 0:
                ref = ref_hash[0]
                motlie = motlie_hash[0]

                if ref != motlie and ref != 'ERROR' and motlie != 'ERROR':
                    issues.append(f"  ✗ {algo:15s} scale={scale:5d}: Hashes differ! ref={ref} motlie={motlie}")
                elif ref == 'ERROR' or motlie == 'ERROR':
                    issues.append(f"  ⚠ {algo:15s} scale={scale:5d}: Error occurred")
                else:
                    print(f"  ✓ {algo:15s} scale={scale:5d}: Results match")

    if issues:
        print()
        print("ISSUES FOUND:")
        for issue in issues:
            print(issue)
    else:
        print()
        print("✓ All implementations produce matching results!")

    print()

def generate_summary_table(df):
    """Generate summary table comparing implementations."""
    print("=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)
    print()

    for algo in sorted(df['algorithm'].unique()):
        print(f"\n{algo.upper()}:")
        print("-" * 80)

        algo_df = df[df['algorithm'] == algo].copy()

        # Create pivot table
        for scale in sorted(algo_df['scale'].unique()):
            scale_df = algo_df[algo_df['scale'] == scale]

            ref_row = scale_df[scale_df['implementation'] == 'reference']
            motlie_row = scale_df[scale_df['implementation'] == 'motlie_db']

            if len(ref_row) == 0 or len(motlie_row) == 0:
                continue

            nodes = ref_row['nodes'].values[0] if len(ref_row) > 0 else motlie_row['nodes'].values[0]
            edges = ref_row['edges'].values[0] if len(ref_row) > 0 else motlie_row['edges'].values[0]

            ref_time = ref_row['time_ms'].values[0] if len(ref_row) > 0 else 'N/A'
            motlie_time = motlie_row['time_ms'].values[0] if len(motlie_row) > 0 else 'N/A'

            ref_mem = ref_row['memory_kb'].values[0] if len(ref_row) > 0 else 'N/A'
            motlie_mem = motlie_row['memory_kb'].values[0] if len(motlie_row) > 0 else 'N/A'

            # Calculate ratios
            time_ratio = "N/A"
            mem_ratio = "N/A"

            try:
                if ref_time != 'N/A' and motlie_time != 'N/A' and ref_time != 'ERROR' and motlie_time != 'ERROR':
                    time_ratio = f"{float(motlie_time) / float(ref_time):.2f}x"
            except:
                pass

            try:
                if ref_mem != 'N/A' and motlie_mem != 'N/A' and ref_mem != 'ERROR' and motlie_mem != 'ERROR':
                    mem_ratio = f"{float(motlie_mem) / float(ref_mem):.2f}x"
            except:
                pass

            print(f"Scale {scale:5d} | Nodes: {nodes:8s} | Edges: {edges:8s}")
            print(f"  Time:   Reference: {str(ref_time):12s} ms | motlie_db: {str(motlie_time):12s} ms | Ratio: {time_ratio}")
            print(f"  Memory: Reference: {str(ref_mem):12s} KB | motlie_db: {str(motlie_mem):12s} KB | Ratio: {mem_ratio}")
            print()

def find_crossover_points(df):
    """Find memory crossover points where motlie_db becomes more efficient."""
    print("=" * 80)
    print("MEMORY CROSSOVER ANALYSIS")
    print("=" * 80)
    print()

    for algo in sorted(df['algorithm'].unique()):
        algo_df = df[df['algorithm'] == algo].copy()

        crossover_found = False
        for scale in sorted(algo_df['scale'].unique()):
            scale_df = algo_df[algo_df['scale'] == scale]

            ref_row = scale_df[scale_df['implementation'] == 'reference']
            motlie_row = scale_df[scale_df['implementation'] == 'motlie_db']

            if len(ref_row) == 0 or len(motlie_row) == 0:
                continue

            ref_mem = ref_row['memory_kb'].values[0]
            motlie_mem = motlie_row['memory_kb'].values[0]

            try:
                if ref_mem != 'N/A' and motlie_mem != 'N/A' and ref_mem != 'ERROR' and motlie_mem != 'ERROR':
                    ref_mem_val = float(ref_mem)
                    motlie_mem_val = float(motlie_mem)

                    if motlie_mem_val <= ref_mem_val and not crossover_found:
                        print(f"  ✓ {algo:15s}: Crossover at scale {scale:5d} (motlie_db: {motlie_mem_val:.2f} KB <= reference: {ref_mem_val:.2f} KB)")
                        crossover_found = True
                        break
            except:
                pass

        if not crossover_found:
            print(f"  ✗ {algo:15s}: No crossover found in tested scales")

    print()

def main():
    """Main analysis function."""
    # Determine CSV file path
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        csv_file = Path(__file__).parent.parent / "data" / "performance_metrics.csv"

    if not Path(csv_file).exists():
        print(f"Error: CSV file not found: {csv_file}")
        print("Please run the data collection script first: ./scripts/collect_all_metrics.sh")
        sys.exit(1)

    print(f"Loading data from: {csv_file}")
    print()

    df = load_data(csv_file)

    # Remove header rows that might have been duplicated
    df = df[df['algorithm'] != 'algorithm']

    # Convert numeric columns
    for col in ['scale', 'nodes', 'edges', 'time_ms', 'memory_kb']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Run analyses
    check_correctness(df)
    generate_summary_table(df)
    find_crossover_points(df)

    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
