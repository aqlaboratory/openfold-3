#!/bin/bash

# Minimal offline prediction script
# Usage: ./predict.sh "SEQUENCE"

if [ $# -eq 0 ]; then
    echo "Usage: $0 <protein_sequence>"
    echo "Example: $0 'MKLHYVAVLTLAILMFLTWLPASLSCNKAL'"
    exit 1
fi

SEQUENCE="$1"
OUTPUT_DIR="debug_$(date +%s)"
QUERY_NAME="debug_query"

# Create temporary query JSON
cat > temp_query.json << EOF
{
    "seeds": [42],
    "queries": {
        "${QUERY_NAME}": {
            "chains": [
                {
                    "molecule_type": "protein",
                    "chain_ids": ["A"],
                    "sequence": "${SEQUENCE}"
                }
            ]
        }
    }
}
EOF

echo "🧬 Running offline prediction for sequence: ${SEQUENCE}"
echo "📁 Output directory: ${OUTPUT_DIR}"
echo "🗂️ Query file: temp_query.json"

# Run prediction with verbose output
python openfold3/run_openfold.py predict \
    --query_json temp_query.json \
    --offline_mode true \
    --foldseek_database_dir /Users/gtaghon/foldseek_databases \
    --runner_yaml offline_runner_config.yaml \
    --output_dir "${OUTPUT_DIR}" \

echo ""
echo "📊 Results summary:"
if [ -d "${OUTPUT_DIR}/${QUERY_NAME}/seed_42" ]; then
    for conf_file in "${OUTPUT_DIR}/${QUERY_NAME}/seed_42"/*_confidences_aggregated.json; do
        if [ -f "$conf_file" ]; then
            sample=$(basename "$conf_file" | sed 's/.*_sample_\([0-9]*\)_.*/\1/')
            plddt=$(grep -o '"avg_plddt": [0-9.]*' "$conf_file" | cut -d' ' -f2)
            echo "  Sample ${sample}: plDDT = ${plddt}"
        fi
    done
else
    echo "  ❌ No results found"
fi

echo ""
echo "🔍 Debug info:"
echo "  - Query JSON: temp_query.json"
echo "  - Output directory: ${OUTPUT_DIR}"
echo "  - MSA temp directories in /var/folders/*/T/ (may be cleaned up)"

# Keep the query file for debugging
echo "  - Query file preserved for debugging"