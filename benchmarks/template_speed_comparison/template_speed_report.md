# FOLDSEEK vs ColabFold Template Speed Benchmark

Generated on: 2025-11-12 14:42:59

## Executive Summary

### Speed Performance
- **FOLDSEEK**: 22.9s average template retrieval
- **ColabFold**: 148.9s average template retrieval
- **Speedup**: 6.5x faster with FOLDSEEK

### Template Coverage
- **FOLDSEEK**: 6.3 average templates
- **ColabFold**: 3.0 average templates

## Detailed Results

| Sequence ID | Length | FOLDSEEK Time (s) | FOLDSEEK Templates | ColabFold Time (s) | ColabFold Templates | Speedup |
|-------------|---------|-------------------|--------------------|--------------------|---------------------|----------|
| T1201 | 210 | 17.7 | 10 | 123.0 | 5 | 6.9x |
| T1206 | 237 | 16.4 | 0 | 131.1 | 0 | 8.0x |
| T1207 | 144 | 17.3 | 4 | 103.2 | 2 | 6.0x |
| T1208s1 | 328 | 33.1 | 12 | 158.4 | 6 | 4.8x |
| T1208s2 | 318 | 21.8 | 3 | 155.4 | 1 | 7.1x |
| T1212 | 466 | 18.5 | 5 | 199.8 | 2 | 10.8x |
| T1214 | 677 | 36.9 | 20 | 263.1 | 10 | 7.1x |
| T1219 | 32 | 20.4 | 2 | 69.6 | 1 | 3.4x |
| T1226 | 123 | 25.8 | 0 | 96.9 | 0 | 3.8x |
| T1227s1 | 427 | 20.7 | 7 | 188.1 | 3 | 9.1x |


