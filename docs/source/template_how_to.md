# A How-To Guide for Running OF3 Inference with Templates

This document contains instructions on how to use template information for OF3 predictions. Here, we assume that you already generated all of your template alignments or intend to fetch them from Colabfold on-the-fly. If you do not have any precomputed template alignments and do not want to use Colabfold, refer to our [MSA Generation Guide](precomputed_msa_generation_how_to.md) before consulting this document. If you need further clarifications on how some of the template components of our inference pipeline work, refer to [this explanatory document](template_explanation.md).

The template pipeline currently supports monomeric templates and has been tested for protein chains only.

The main steps detailed in this guide are:
1. Providing files for template featurization
2. Adding template information to the inference query json
3. Modifying the template pipeline settings and high-throughput workflow support

## 1. Template Files

Template featurization requires query-to-template **alignments** and template **structures**.

### 1.1. Template Aligment File Format

Template alignments can be provided in either `sto`, `a3m` or `m8` format.

#### 1.1.1. STO

Files in `sto` format expect the fields provided by default by hmmer alignment tools (hmmsearch, hmmalign). These are:
1. metadata headers: `#=GS <entry id>_<chain id>/<start>-<end> mol:<molecule type>`
    - `#=GS`: indicates header info
    - `<entry id>_<chain id>`: entry identifier indicating which structure file to parse (usually PDB entry ID) and chain identifier indicating which chain in this complex is to be used as the template chain
    - `<start>-<end>`: start and end residue indices (1-indexed) indicating which position of the aligned template sequence with respect to the full template sequence
    - `mol:<molecule type>`: type of the template molecule, currently only support *protein*
2. alignment rows: `<entry id>_<chain id>    ALIGNED-SEQUENCE`
    - `<entry id>_<chain id>`: to match the alignment to the header, may contain /start-end positions but these are not used
    - `ALIGNED-SEQUENCE`: the actual sequence alignment, may be split across multiple rows

<details>
<summary>Example `sto` template alignment format ...</summary>
<pre><code>
# STOCKHOLM 1.0

#=GS query_A/1-100 DE [subseq from] mol:protein length:100
#=GS template_B/50-150 DE [subseq from] mol:protein length:200

query_A     MKLLVVDDA--GQKFT
template_B  MK--VVDDARGQGKFT
//
</code></pre>
</details>

<br>

Note that the `sto` parser attempts to derive the query-to-template residue correspondences from the existing alignment. If this is not possbile, we realign the template sequences to the provided query sequence using Kalign. More on this in the [template processing explanatory document](template_explanation.md).

#### 1.1.2. A3M

Files in the `a3m` format require the standard fasta format with optional start/end positions:
1. headers: `><entry ID>_<chain ID>/<start>-<end>`
    - `<entry id>_<chain id>`: entry identifier indicating which structure file to parse (usually PDB entry ID) and chain identifier indicating which chain in this complex is to be used as the template chain
    - `<start>-<end>`: *optional*, start and end residue indices (1-indexed) indicating which position of the aligned template sequence with respect to the full template sequence
2. alignment rows: `ALIGNED-SEQUENCE`
    - `ALIGNED-SEQUENCE`: the actual sequence, needs to be aligned if the header contains start-end positions, otherwise the unaligned sequence

<details>
<summary>Example `a3m` template alignment format ...</summary>
<pre><code>
>query_A/1-100
MKLLVVDDA--GQGKFT
>template_B/50-150
MK--VVDDAaRGQGKFT
</code></pre>
</details>

<br>

Note that the `a3m` parser attempts to derive the query-to-template residue correspondences from the existing alignment. If this is not possbile, we realign the template sequences to the provided query sequence using Kalign. More on this in the [template processing explanatory document](template_explanation.md).

#### 1.1.3. M8

Files in `m8` format expect the standard BLAST tabular output format with 12 tab-separated columns. We only use columns 1. (`<entry ID>_<chain ID>`), 3. (sequence identity of the template to the query) and 11. (e value). For all columns, see https://linsalrob.github.io/ComputationalGenomicsManual/SequenceFileFormats/.

<details>
<summary>Example `m8` template alignment format ...</summary>
<pre><code>
query_A	template_B	85.7	14	2	0	1	14	50	63	1e-05	28.1
query_A	template_C	71.4	14	4	0	5	18	75	88	2e-03	22.3
</code></pre>
</details>

<br>

Note that since `m8` files do not provide actual alignments, we only use them to identify which structure files to get templates from and always realign the associated set of sequences, derived from the structure files, to the query sequence using Kalign. More on this in the [template processing explanatory document](template_explanation.md).

### 1.2. Template Structure File Format

Template structures currently can only be provided in `cif` format. An upcoming release will add support for parsing templates from `pdb` files.

## 2. Specifying Template Information in the Inference Query File

### 2.1. Specifying Alignments

### 2.2. Using Specific Templates

## 3. Modifying Template Settings and Optimizations for High-Throughput Workflows

### 3.1. 