#!/usr/bin/env python3
"""
Compare OpenFold3-MLX vs Google AF3 Server Results

This script helps analyze and compare results between:
1. Local OpenFold3-MLX benchmarks (from benchmark_runner.py)
2. Google AlphaFold3 server results (downloaded after submission)

Usage:
    python compare_af3_results.py --mlx results/benchmark_results.json --af3 af3_server_results/

Expected AF3 directory structure:
    af3_server_results/
    ├── T1201/
    │   ├── fold_T1201_model_0.cif
    │   ├── confidence_T1201_model_0.json
    │   └── ... (other files)
    └── T1206/
        └── ...
"""

import argparse
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import glob
import sys


def load_mlx_results(results_file: Path) -> pd.DataFrame:
    """Load OpenFold3-MLX benchmark results"""
    with open(results_file, 'r') as f:
        mlx_results = json.load(f)

    df = pd.DataFrame(mlx_results)
    print(f"📊 Loaded {len(df)} MLX results")
    return df


def parse_af3_results(af3_dir: Path) -> pd.DataFrame:
    """Parse Google AF3 server results from download directory"""
    results = []

    for seq_dir in af3_dir.glob("T*"):
        if not seq_dir.is_dir():
            continue

        seq_id = seq_dir.name

        # Find confidence files
        confidence_files = list(seq_dir.glob("*confidence*.json"))
        structure_files = list(seq_dir.glob("*model*.cif"))

        if not confidence_files:
            print(f"⚠️  No confidence file found for {seq_id}")
            results.append({
                'sequence_id': seq_id,
                'af3_plddt': None,
                'af3_structure_files': len(structure_files),
                'af3_available': False
            })
            continue

        # Parse confidence scores (assuming similar format to OpenFold3)
        try:
            with open(confidence_files[0], 'r') as f:
                conf_data = json.load(f)

            if 'plddt' in conf_data:
                plddt_scores = conf_data['plddt']
                mean_plddt = sum(plddt_scores) / len(plddt_scores)
            elif 'confidenceScore' in conf_data:  # Alternative AF3 format
                mean_plddt = conf_data['confidenceScore']
            else:
                print(f"⚠️  Unknown confidence format for {seq_id}")
                mean_plddt = None

            results.append({
                'sequence_id': seq_id,
                'af3_plddt': mean_plddt,
                'af3_structure_files': len(structure_files),
                'af3_available': True
            })

            print(f"✅ {seq_id}: pLDDT {mean_plddt:.1f}" if mean_plddt else f"❌ {seq_id}: No pLDDT")

        except Exception as e:
            print(f"❌ Error parsing {seq_id}: {e}")
            results.append({
                'sequence_id': seq_id,
                'af3_plddt': None,
                'af3_structure_files': len(structure_files),
                'af3_available': False
            })

    df = pd.DataFrame(results)
    print(f"📊 Parsed {len(df)} AF3 results")
    return df


def compare_results(mlx_df: pd.DataFrame, af3_df: pd.DataFrame) -> Dict:
    """Compare MLX vs AF3 results and generate statistics"""

    # Merge datasets
    comparison_df = mlx_df.merge(af3_df, on='sequence_id', how='outer')

    # Filter to sequences with both results
    both_available = comparison_df[
        (comparison_df['mean_plddt'].notna()) &
        (comparison_df['af3_plddt'].notna())
    ].copy()

    if len(both_available) == 0:
        return {
            'error': 'No sequences with both MLX and AF3 results found',
            'comparison_df': comparison_df
        }

    # Calculate differences
    both_available['plddt_diff'] = both_available['mean_plddt'] - both_available['af3_plddt']
    both_available['plddt_ratio'] = both_available['mean_plddt'] / both_available['af3_plddt']

    stats = {
        'total_sequences': len(comparison_df),
        'mlx_available': len(comparison_df[comparison_df['mean_plddt'].notna()]),
        'af3_available': len(comparison_df[comparison_df['af3_plddt'].notna()]),
        'both_available': len(both_available),

        'mlx_stats': {
            'mean_plddt': both_available['mean_plddt'].mean(),
            'median_plddt': both_available['mean_plddt'].median(),
            'std_plddt': both_available['mean_plddt'].std(),
        },

        'af3_stats': {
            'mean_plddt': both_available['af3_plddt'].mean(),
            'median_plddt': both_available['af3_plddt'].median(),
            'std_plddt': both_available['af3_plddt'].std(),
        },

        'comparison_stats': {
            'mean_diff': both_available['plddt_diff'].mean(),
            'median_diff': both_available['plddt_diff'].median(),
            'std_diff': both_available['plddt_diff'].std(),
            'mean_ratio': both_available['plddt_ratio'].mean(),
            'mlx_wins': len(both_available[both_available['plddt_diff'] > 0]),
            'af3_wins': len(both_available[both_available['plddt_diff'] < 0]),
            'ties': len(both_available[abs(both_available['plddt_diff']) < 0.1]),
        },

        'comparison_df': comparison_df,
        'both_available_df': both_available
    }

    return stats


def print_comparison_report(stats: Dict):
    """Print a formatted comparison report"""

    if 'error' in stats:
        print(f"❌ {stats['error']}")
        return

    print("\n" + "=" * 80)
    print("OPENFOLD3-MLX vs GOOGLE AF3 SERVER COMPARISON")
    print("=" * 80)

    print(f"📊 Dataset Overview:")
    print(f"  Total CASP16 sequences: {stats['total_sequences']}")
    print(f"  MLX results available: {stats['mlx_available']}")
    print(f"  AF3 results available: {stats['af3_available']}")
    print(f"  Both available (compared): {stats['both_available']}")
    print(f"  Coverage: {stats['both_available']/stats['total_sequences']:.1%}")

    print(f"\n🏆 pLDDT Accuracy Results:")
    print(f"  OpenFold3-MLX:")
    print(f"    Mean pLDDT: {stats['mlx_stats']['mean_plddt']:.1f}")
    print(f"    Median pLDDT: {stats['mlx_stats']['median_plddt']:.1f}")
    print(f"    Std Dev: {stats['mlx_stats']['std_plddt']:.1f}")

    print(f"  Google AF3 Server:")
    print(f"    Mean pLDDT: {stats['af3_stats']['mean_plddt']:.1f}")
    print(f"    Median pLDDT: {stats['af3_stats']['median_plddt']:.1f}")
    print(f"    Std Dev: {stats['af3_stats']['std_plddt']:.1f}")

    print(f"\n📈 Head-to-Head Comparison:")
    diff = stats['comparison_stats']['mean_diff']
    print(f"  Mean pLDDT difference (MLX - AF3): {diff:+.1f}")
    print(f"  Median pLDDT difference: {stats['comparison_stats']['median_diff']:+.1f}")
    print(f"  Performance ratio (MLX/AF3): {stats['comparison_stats']['mean_ratio']:.3f}")

    print(f"\n🥊 Win/Loss Record:")
    print(f"  MLX wins (higher pLDDT): {stats['comparison_stats']['mlx_wins']}")
    print(f"  AF3 wins (higher pLDDT): {stats['comparison_stats']['af3_wins']}")
    print(f"  Ties (±0.1 pLDDT): {stats['comparison_stats']['ties']}")

    if diff > 0:
        print(f"\n🎉 OpenFold3-MLX shows {diff:.1f} point average advantage!")
    elif diff < -0.1:
        print(f"\n📊 Google AF3 shows {abs(diff):.1f} point average advantage")
    else:
        print(f"\n⚖️  Results are essentially equivalent (difference < 0.1)")


def save_detailed_comparison(stats: Dict, output_file: Path):
    """Save detailed comparison to CSV"""
    df = stats['both_available_df']

    # Reorder columns for clarity
    output_cols = [
        'sequence_id', 'mean_plddt', 'af3_plddt', 'plddt_diff', 'plddt_ratio',
        'wall_time_seconds', 'peak_memory_gb', 'af3_structure_files'
    ]

    available_cols = [col for col in output_cols if col in df.columns]
    df_output = df[available_cols]

    df_output.to_csv(output_file, index=False)
    print(f"\n💾 Detailed comparison saved to: {output_file}")


def main():
    """Main comparison function"""
    parser = argparse.ArgumentParser(description="Compare OpenFold3-MLX vs Google AF3 results")
    parser.add_argument("--mlx", type=Path, required=True,
                       help="Path to MLX benchmark_results.json")
    parser.add_argument("--af3", type=Path, required=True,
                       help="Path to AF3 results directory")
    parser.add_argument("--output", type=Path, default="comparison_results.csv",
                       help="Output file for detailed comparison")

    args = parser.parse_args()

    if not args.mlx.exists():
        print(f"❌ MLX results file not found: {args.mlx}")
        sys.exit(1)

    if not args.af3.exists():
        print(f"❌ AF3 results directory not found: {args.af3}")
        sys.exit(1)

    print("Loading and comparing OpenFold3-MLX vs Google AF3 results...")
    print("=" * 60)

    # Load results
    mlx_df = load_mlx_results(args.mlx)
    af3_df = parse_af3_results(args.af3)

    # Compare results
    print(f"\n🔍 Performing comparison analysis...")
    stats = compare_results(mlx_df, af3_df)

    # Generate report
    print_comparison_report(stats)

    # Save detailed results
    if 'both_available_df' in stats:
        save_detailed_comparison(stats, args.output)

    print(f"\n✅ Comparison complete!")


if __name__ == "__main__":
    main()