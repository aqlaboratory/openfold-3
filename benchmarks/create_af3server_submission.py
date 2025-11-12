#!/usr/bin/env python3
"""
Create Google AlphaFold3 Server Submission JSON from CASP16 sequences

This script converts the CASP16 monomers FASTA file into the proper JSON format
for batch submission to Google's AlphaFold3 server for direct comparison.

Usage:
    python create_af3server_submission.py
"""

import json
import sys
import re
from pathlib import Path
from datetime import datetime


def sanitize_job_name(name):
    """Sanitize job name by converting non-standard characters to dashes"""
    # Replace spaces, colons, pipes, and other special chars with dashes
    sanitized = re.sub(r'[^a-zA-Z0-9._-]', '-', name)
    # Remove multiple consecutive dashes
    sanitized = re.sub(r'-+', '-', sanitized)
    # Remove leading/trailing dashes
    sanitized = sanitized.strip('-')
    return sanitized


def parse_fasta(fasta_file):
    """Parse FASTA file and return list of (id, description, sequence) tuples"""
    sequences = []
    current_id = None
    current_desc = None
    current_seq = []

    with open(fasta_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith('>'):
                # Save previous sequence
                if current_id is not None:
                    sequences.append((current_id, current_desc, ''.join(current_seq)))

                # Parse new header
                header_parts = line[1:].split(' ', 1)
                current_id = header_parts[0]
                current_desc = header_parts[1] if len(header_parts) > 1 else ""
                current_seq = []
            else:
                current_seq.append(line)

        # Save last sequence
        if current_id is not None:
            sequences.append((current_id, current_desc, ''.join(current_seq)))

    return sequences


def create_af3_server_job(seq_id, description, sequence):
    """Create a single AF3 server job object for a protein sequence"""

    # Clean up description for job name
    clean_desc = description.replace('|', ' - ').strip()
    if len(clean_desc) > 100:
        clean_desc = clean_desc[:97] + "..."

    # Create raw job name
    raw_name = f"{seq_id}: {clean_desc}" if clean_desc else seq_id

    # Sanitize the job name to remove incompatible characters
    job_name = sanitize_job_name(raw_name)

    return {
        "name": job_name,
        "modelSeeds": [],
        "sequences": [
            {
                "proteinChain": {
                    "sequence": sequence,
                    "count": 1,
                    # Use a recent date for template cutoff (optional)
                    "maxTemplateDate": "2024-05-08"
                }
            }
        ],
        "dialect": "alphafoldserver",
        "version": 1
    }


def main():
    """Main function"""
    print("Creating Google AlphaFold3 Server submission JSON from CASP16 sequences")
    print("=" * 80)

    # Input and output paths
    fasta_file = Path("benchmarks/casp16_monomers.fasta")
    output_file = Path("benchmarks/af3server_submission_casp16.json")

    if not fasta_file.exists():
        print(f"❌ Error: {fasta_file} not found")
        print("Please run this script from the benchmarks directory")
        sys.exit(1)

    # Parse FASTA sequences
    print(f"📄 Parsing sequences from {fasta_file}...")
    sequences = parse_fasta(fasta_file)
    print(f"✅ Found {len(sequences)} sequences")

    # Create AF3 server jobs
    print(f"🔨 Converting to AF3 server format...")
    jobs = []

    for seq_id, description, sequence in sequences:
        job = create_af3_server_job(seq_id, description, sequence)
        jobs.append(job)
        print(f"  - {seq_id}: {len(sequence)} residues")

    # Write output JSON
    print(f"💾 Writing submission JSON to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(jobs, f, indent=2, ensure_ascii=False)

    # Summary
    total_residues = sum(len(seq[2]) for seq in sequences)
    print(f"\n📊 Summary:")
    print(f"  Total jobs: {len(jobs)}")
    print(f"  Total residues: {total_residues:,}")
    print(f"  Average length: {total_residues / len(sequences):.1f} residues")
    print(f"  Output file: {output_file}")
    print(f"  File size: {output_file.stat().st_size / 1024:.1f} KB")

    print(f"\n🚀 Ready for submission to Google AlphaFold3 server!")
    print(f"   Upload {output_file} to: https://alphafoldserver.com/")

    # Show first few job names as examples
    print(f"\n📋 Example job names:")
    for i, job in enumerate(jobs[:5]):
        print(f"  {i+1}. {job['name']}")
    if len(jobs) > 5:
        print(f"  ... and {len(jobs) - 5} more")

    print(f"\n✅ Submission file created successfully!")


if __name__ == "__main__":
    main()