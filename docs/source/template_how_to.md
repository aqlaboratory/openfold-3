# A How-To Guide for Running OF3 Inference with Templates

This document contains instructions on how to use template information for OF3 predictions. Here, we assume that you already generated all of your template alignments or intend to fetch them from Colabfold on-the-fly. If you do not have any precomputed template alignments and do not want to use Colabfold, refer to our {doc}`MSA Generation Guide <precomputed_msa_generation_how_to>` before consulting this document. If you need further clarifications on how some of the template components of our inference pipeline work, refer to {doc}`this explanatory document <template_explanation>`.

The template pipeline currently supports monomeric templates and has been tested for protein chains only.

The main steps detailed in this guide are:
1. {ref}`Providing files for template featurization <1-template-files>`
2. {ref}`Adding template information to the inference query json <2-specifying-template-information-in-the-inference-query-file>`
3. {ref}`High-throughput workflow support <3-optimizations-for-high-throughput-workflows>`

(1-template-files)=
## 1. Template Files

Template featurization requires query-to-template **alignments** and template **structures**.

(11-template-aligment-file-format)=
### 1.1. Template Aligment File Format

Template alignments can be provided in either `sto`, `a3m` or `m8` format. Template alignments from the Colabfold server are in `m8` format.

#### 1.1.1. STO

Files in `sto` format need to contain the fields provided by default by hmmer alignment tools (hmmsearch, hmmalign). These are:
1. metadata headers: `#=GS <entry id>_<chain id>/<start>-<end> mol:<molecule type>`
    - `#=GS`: indicates header info
    - `<entry id>_<chain id>`: entry identifier indicating which structure file to parse (usually PDB entry ID) and chain identifier indicating which chain in this complex is to be used as the template chain
    - `<start>-<end>`: start and end residue indices (1-indexed) indicating which position of the aligned template sequence with respect to the full template sequence
    - `mol:<molecule type>`: type of the template molecule, currently only support *protein*
2. alignment rows: `<entry id>_<chain id>    ALIGNED-SEQUENCE`
    - `<entry id>_<chain id>`: to match the alignment to the header, may contain /start-end positions but these are not used
    - `ALIGNED-SEQUENCE`: the actual sequence alignment, may be split across multiple rows

```
# STOCKHOLM 1.0

#=GS entry1_A/1-100 mol:protein
#=GS entry2_B/50-150 mol:protein

entry1_A     MKLLVVDDA--GQKFT
entry2_B     MK--VVDDARGQGKFT
//
```

Note that the `sto` parser attempts to derive the query-to-template residue correspondences from the existing alignment. If this is not possbile, we realign the template sequences to the provided query sequence using Kalign. More on this in the [template processing explanatory document](template_explanation.md).

#### 1.1.2. A3M

Files in the `a3m` format require the standard fasta format with optional start/end positions:
1. headers: `><entry ID>_<chain ID>/<start>-<end>`
    - `<entry id>_<chain id>`: entry identifier indicating which structure file to parse (usually PDB entry ID) and chain identifier indicating which chain in this complex is to be used as the template chain
    - `<start>-<end>`: *optional*, start and end residue indices (1-indexed) indicating which position of the aligned template sequence with respect to the full template sequence
2. alignment rows: `ALIGNED-SEQUENCE`
    - `ALIGNED-SEQUENCE`: the actual sequence, needs to be aligned if the header contains start-end positions, otherwise the unaligned sequence

```
>entry1_A/1-100
MKLLVVDDA--GQGKFT
>entry2_B/50-150
MK--VVDDAaRGQGKFT
```

Note that the `a3m` parser attempts to derive the query-to-template residue correspondences from the existing alignment. If this is not possbile, we realign the template sequences to the provided query sequence using Kalign. More on this in the [template processing explanatory document](template_explanation.md).

#### 1.1.3. M8

Files in `m8` format expect the standard BLAST tabular output format with 12 tab-separated columns. We only use columns 1. (`<entry ID>_<chain ID>`), 3. (sequence identity of the template to the query) and 11. (e value). For all columns, see https://linsalrob.github.io/ComputationalGenomicsManual/SequenceFileFormats/.

```
query_A	template_B	85.7	14	2	0	1	14	50	63	1e-05	28.1
query_A	template_C	71.4	14	4	0	5	18	75	88	2e-03	22.3
```

Note that since `m8` files do not provide actual alignments, we only use them to identify which structure files to get templates from, retrieve sequences from these structure files and always realign them to the query sequence using Kalign. More on this in the [template processing explanatory document](template_explanation.md).

### 1.2. Template Structure File Format

Template structures currently can only be provided in `cif` format. An upcoming release will add support for parsing templates from `pdb` files.

(2-specifying-template-information-in-the-inference-query-file)=
## 2. Specifying Template Information in the Inference Query File

### 2.1. Specifying Alignments

The data pipeline needs to know which template alignment to use for which chain. This information is provided by specifying the {ref}`paths to the alignments <31-protein-chains>` for each chain's `template_alignment_file_path` field in the inference query json file.

Note that when fetching alignments from the Colabfold server, `template_alignment_file_path` fields are automatically populated.

<details>
<summary>Template alignment file path example ...</summary>
<pre><code>
{
    "queries": {
        "example_query": {
            "chains": [
                {
                    "molecule_type": "protein",
                    "chain_ids": "A",
                    "sequence": "GCTLSAEDKAAVERSKMIDRNLREDGEKAAREVKLLLLGAGESGKSTIVKQMKIIHEAGYSEEECKQYKAVVYSNTIQSIIAIIRAMGRLKIDFGDAARADDARQLFVLAGAAEEGFMTAELAGVIKRLWKDSGVQACFNRSREYQLNDSAAYYLNDLDRIAQPNYIPTQQDVLRTRVKTTGIVETHFTFKDLHFKMFDVGAQRSERKKWIHCFEGVTAIIFCVALSDYDLVLAEDEEMNRMHESMKLFDSICNNKWFTDTSIILFLNKKDLFEEKIKKSPLTICYPEYAGSNTYEEAAAYIQCQFEDLNKRKDTKEIYTHFTCATDTKNVQFVFDAVTDVIIKNNLKDCGLF",
                    "template_alignment_file_path": "example_chain_A.sto"
                },
                {
                    "molecule_type": "protein",
                    "chain_ids": "B",
                    "sequence": "MSELDQLRQEAEQLKNQIRDARKACADATLSQITNNIDPVGRIQMRTRRTLRGHLAKIYAMHWGTDSRLLVSASQDGKLIIWDSYTTNKVHAIPLRSSWVMTCAYAPSGNYVACGGLDNICSIYNLKTREGNVRVSRELAGHTGYLSCCRFLDDNQIVTSSGDTTCALWDIETGQQTTTFTGHTGDVMSLSLAPDTRLFVSGACDASAKLWDVREGMCRQTFTGHESDINAICFFPNGNAFATGSDDATCRLFDLRADQELMTYSHDNIICGITSVSFSKSGRLLLAGYDDFNCNVWDALKADRAGVLAGHDNRVSCLGVTDDGMAVATGSWDSFLKIWN",
                    "template_alignment_file_path": "example_chain_B.sto"
                },
                {
                    "molecule_type": "protein",
                    "chain_ids": "C",
                    "sequence": "MASNNTASIAQARKLVEQLKMEANIDRIKVSKAAADLMAYCEAHAKEDPLLTPVPASENPFREKKFFSAIL",
                    "template_alignment_file_path": "example_chain_C.sto"
                },
            ],
        }
    }
}
</code></pre>
</details>

### 2.2. Using Specific Templates

By default, the template pipeline automatically populates the `template_entry_chain_ids` field with [n templates](../../openfold3/core/data/pipelines/preprocessing/template.py#L1519) from the alignment, which is then further subset to the [top k templates](../../openfold3/projects/of3_all_atom/config/dataset_config_components.py#L112) during featurization for inference.

In an **upcoming release**, we will add support for specifying *specific templates* for the data pipeline to use for featurization. This will be possible through the `template_entry_chain_ids` field:

```
{
    "queries": {
        "example_query": {
            "chains": [
                {
                    "molecule_type": "protein",
                    "chain_ids": "A",
                    "sequence": "EXAMPLEPROTEINSEQUENCE",
                    "template_alignment_file_path": "example_chain_A.sto",
                    "template_entry_chain_ids": ["entry1_A", "entry2_B", "entry3_A"]
                },
            ],
        }
    }
}
```

Note that the corresponding template IDs need to be present in the provided raw alignment file, so here, IDs `"entry1_A"`, `"entry2_B"`, `"entry3_A"` and corresponding alignments need be present in `example_chain_A.sto` like so:

```
# STOCKHOLM 1.0

#=GS entry1_A/1-100 mol:protein
#=GS entry2_B/50-150 mol:protein

entry1_A     MKLLVVDDA--GQKFT
entry2_B     MK--VVDDARGQGKFT
entry3_A     MK----DDARGQGKFT
//
```

(3-optimizations-for-high-throughput-workflows)=
## 3. Optimizations for High-Throughput Workflows

For high-throughput use cases, where a large number of structures are to be predicted, template processing can take a significant amount of time even with the built-in {doc}`deduplication utility <template_explanation>` we have for template alignment and structure processing. To avoid having to spend GPU compute on data transformations, we provide separate template preprocessing scripts to generate the necessary inputs from which template featurization can run efficiently in a subsequent job without being a bottleneck to the model forward pass.

### 3.1. Template Alignment Preprocessing

A recommended workflow for providing template data for very large datasets is the following:
1. Compute {ref}`template alignments <311-precomputed-template-alignments>`.
2. [Download the PDB](../../scripts/snakemake_msa/download_of3_databases.py) or other template structure dataset locally.
3. Precompute the {ref}`template precache <312-template-precache>` from template structures to speed up template cache precomputation.
4. Precompute the {ref}`template cache <313-template-cache>` from template alignments and the template precache.
5. Preparse the {ref}`template structures <32-template-structure-preprocessing>` into template structure arrays.

This workflow produces a set of *template cache entries* and *preparsed template structures* for on-the-fly data processing that happens concurrently with the model forward pass. Each of these steps are detailed below.

(311-precomputed-template-alignments)=
#### 3.1.1. Precomputed Template Alignments

Our template processing pipeline accepts MSAs generated from our {doc}`OF3-style MSA pipeline <precomputed_msa_generation_how_to>` or from other workflows as long as they are in one of the {ref}`expected formats <11-template-aligment-file-format>`.

(312-template-precache)=
#### 3.1.2. Template Precache

We found that preprocessing template alignments for large datasets can take a long time, partly due to the requirement to parse template structures so we can correspond them to the template alignment sequences. We provide a [preprocessing script](../../scripts/data_preprocessing/preprocess_template_alignments_precache_of3.py) that compresses template structure files into metadata files which we call *template precache entries*, containing the release date and a mapping from chain `asym_id` identifiers to their canonical sequences denoted in the structure file:

```python
{
    'release_date': <datetime.datetime>,
    'chain_id_seq_map': 
    {
        '<chain ID>': '<canonical sequence>',
        '<chain ID>': '<canonical sequence>',
        <...>
    },
}
```

You can run this script using:
```
python preprocess_template_alignments_precache_of3.py \
    --runner_yaml <path/to/runner.yml>
```

with runner.yml like this:

```
template_preprocessor_settings:
  n_processes: 4
  chunksize: 1
  structure_directory: <path/to/template/structures>
  structure_file_format: "cif"
  precache_directory: <path/to/output/precache>
```

Using these files instead of the raw structure files during template cache creation drastically speeds up processing. For the full PDB we observed a reduction of template processing runtimes from *120 hours without a template precache* to *3 hours with a template precache* (including precache computation time) when running template cache creation on 250 parallel processes.

(313-template-cache)=
#### 3.1.2. Template Cache

Under the hood, the OF3 inference pipeline uses a preprocessed version of the template alignments during online data processing, which we call the *template cache*. In short, each unique sequence in the inference query set gets its own template cache entry and each of these cache entries contain processed and validated template alignment data:

```python
{
    '<template entry ID>_<template chain ID>': {
        'index': <int>,
        'release_date': <datetime.datetime>,
        'idx_map': <np.array>
    },
    '<template entry ID>_<template chain ID>': {
        'index': <int>,
        'release_date': <datetime.datetime>,
        'idx_map': <np.array>
    },
}
```

You can read more about what template cache entry files contain, how they are generated and why we do this preprocesing in the {doc}`template explanatory document <template_explanation>`. 

By default, the inference pipeline automatically generates the template cache entries. However, for larger datasets, we provide a [template alignment preprocesing script](../../scripts/data_preprocessing/preprocess_template_alignments_new_of3.py), which preprocesses the template alignments (and optionally the template structures). Below is an example run script:

```
python preprocess_template_alignments_new_of3.py \
    --input_set_path <path/to/input/query.json> \
    --input_set_type "predict" \
    --runner_yaml <path/to/runner.yml> \
    --output_set_path <path/to/updated/output/query.json> \
```

where `input_set_path` is the inference query.json, `output_set_path` is the output json with the updated template information following preprocessing and `runner_yaml` contains the preprocessing configuration, for example:

```
template_preprocessor_settings:
  n_processes: 4  
  chunksize: 1
  precache_directory: <path/to/precache>
  cache_directory: <path/to/output/template/cache>
```

This script runs 4 parallel processes to preprocesse the template alignments specified under the `template_alignment_file_path` field of each chain in the inference query json, using the template structures precached at the path given by `precache_directory` and outputs the template cache to `cache_directory`. If precaching was not done, you can run processing from the raw structures by specifying them under the `structure_directory` field and dropping `precache_directory`.

(32-template-structure-preprocessing)=
### 3.2. Template Structure Preprocessing

One of the main bottlenecks we found in template featurization is the parsing of the template cif files. More on this in the {doc}`template explanatory document <template_explanation>`. You can preprocess template structures into biotite [AtomArrays](https://www.biotite-python.org/latest/apidoc/biotite.structure.AtomArray.html) using our [template structure preprocessing script](../../scripts/data_preprocessing/preprocess_template_structures_of3.py):

```
python preprocess_template_structures_of3.py \
    --runner_yaml <path/to/runner.yml>
```

and runner.yml

```
template_preprocessor_settings:
  moltypes: "protein"
  n_processes: 4
  chunksize: 1
  structure_directory: <path/to/template/structures>
  structure_file_format: "cif"
  structure_array_directory: <path/to/output/structure/arrays>
  ccd_file_path: <optional/path/to/ccd/file>
```

where a CCD file can be optionally provided if the template structures contain custom ligands or other chemical components.