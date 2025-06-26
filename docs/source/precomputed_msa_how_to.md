# A How-To Guide for Precomputed MSAs in the OF3 Inference Pipeline

In this document, we intend to provide a guide on how to format and organize precomputed multiple sequence alignments (MSAs) and how to provide settings for the Openfold3 inference pipeline to use these MSAs correctly for creating MSA features for Openfold3. Use this guide if you already generated MSAs using your own or our internal OF3-style pipeline. If you have yet to generate MSAs and would like to use our workflow, refer to our [MSA Generation Guide](precomputed_msa_generation_how_to.md). If you need further clarifications on how some of the MSA components of our inference pipeline work, refer to [this explanatory document](precomputed_msa_explanation.md).

The main steps detailed in this guide are:
1. [Providing the MSAs in the expected format](precomputed_msa_how_to.md#1-precomputed-msa-files)
2. [Organizing the MSAs in the expected directory structure](precomputed_msa_how_to.md#2-precomputed-msa-directory-structure-and-file-name-conventions)
3. [Preparsing MSAs into NPZ format](precomputed_msa_how_to.md#3-preparsing-raw-msas-into-npz-format)
4. [Adding the MSA file/directory paths to the inference query json](precomputed_msa_how_to.md#4-specifying-paths-in-the-inference-query-file)
5. [Updating the MSA pipeline settings](precomputed_msa_how_to.md#5-modifying-msa-settings-for-custom-precomputed-msas)

IMPORTANT: 

*Using your own MSAs generated with a custom workflow*: 
- Steps 1, 2, 4, 5 are required, step 3 is optional, but recommended if your dataset is large or redundant in the number of unique sequences.
- Make sure to **consult steps 1 and 2 beforehand**, especially if your use case requires the computation of a large number of alignments. This is to ensure that your MSAs are generated in the expected format, with the expected filenames and are organized in the expected directory structure.

*Using our OF3-style MSA generation pipeline*
- Only step 4 is required, step 3 is optional, but recommended if your dataset is large or redundant in the number of unique sequences.

## 1. Precomputed MSA Files

In the OF3 inference pipeline, we differentiate between two types of MSAs:
1. **main MSAs**
    - regular alignments directly returned by popular alignment tools 
    - used to provide features for every protein and RNA chain
2. **paired MSAs** 
    - alignments whose rows are arranged based on species information relative to other chains' alignments in a specific bioassembly
    - by default, computed from designated main MSAs on-the-fly and only for complexes with at least two unique protein chains
    - if precomputed, can be provided explicitly for protein and RNA chains

Refer to the [MSA Input Components](precomputed_msa_explanation.md#1-msa-input-feature-components) section for further details.

### 1.1. General MSA File Format

MSAs generated with a custom workflow should follow the same format as that of the files output by our [snakemake MSA generation pipeline](precomputed_msa_generation_how_to.md). Both main and precomputed paired MSAs
- can be in either `a3m` or `sto` format
- need to have the query sequence (the protein or RNA sequence for which the structure is to be predicted) as the first sequence in the MSA

<details>
<summary>Example `a3m` for PDB entry 5k36 protein chain B ...</summary>
<pre><code>
>5k36_B
GPDHMSRLEIYSPEGLRLDGRRWNELRRFESSINTHPHAADGSSYMEQGNNKIITLVKGPKEPRLKSQMDTSKALLNVSVNITKFSKFERSKSSHKNERRVLEIQTSLVRMFEKNVMLNIYPRTVIDIEIHVLEQDGGIMGSLINGITLALIDAGISMFDYISGISVGLYDTTPLLDTNSLEENAMSTVTLGVVGKSEKLSLLLVEDKIPLDRLENVLAIGIAGAHRVRDLMDEELRKHAQKRVSNASAR
>tr|D5G6D5|D5G6D5_TUBMM
---NRPSSHLHKPSSLPSTSHSFlKKLENVPLRNPLTRRPPhRRPSYVEHGNTKVICSVNGPIEPRAASARNSERATVTVDVCFAAFSGTDRKKRG-KSDKRVLEMQSALSRTFATTLLTTLHPRSEVHISLHILSQDGSILATCVNAATLALVDAGVPMSDYVTACTVASYTNpdesgEPLLDMSSAEEMDLPGITLATVGRSDKISLLQLETKVRLERLEGMLAVGIDGCGKIRQLLDGVIREHGNKMARMGAL-
>MGYP001248485810
---TMSRFDFYNSQGLRIDGRRNYELKNFESSLTTTSNFNnfsrnsqSNTTYLQMGQNKILVNIDGPKEPtnANRSRIDQDKAVLDININVTKFSKVNRQVST-NSnnlpDKQTQEWEFEIQKLFEKIIILETYPKSVINVSVTVLQQDGGILASIINCVSIALMNNSIQVYDIVSACSVGIVDQkHYLLDLNHLEEQFLTSGTIAIIGNSSlqniedaNVCLLSLKDIFPLDLLDGFMMIGIKGCNTLKEIMVKQVKDMNINKLIEIQ--
>SAMEA103904984:k141_247917_5
---AGGRIEFLSPEGLRVDGRRPNELRSYRAQLAVIPQA-DGSALFSLGNTTVIATVYGPRDNNNHNSSNTECSINTkIHAAAFSSTTGDRRKagSS-NTDRRLQDWSETVSHTISGVLLHDLFPRTSLDIFVEVLSADGAVLAASINAVSLALVDAGVPMRDPVVALQGVIIREHLLLDGNRLEERAGAPTTLAFTPRNGKIVGVMVDPKYPQHRFQDVCTMLQPHSESVFAHLDSEVirprLKHLYSMLK-----
... rest of the sequences ...
</code></pre>
</details>

<details>
<summary>Example `sto` for PDB entry 5k36 protein chain B ...</summary>
<pre><code>
# STOCKHOLM 1.0

#=GS MGYP003365344427/1-246   DE [subseq from] FL=1
#=GS MGYP003366480418/1-243   DE [subseq from] FL=1
#=GS MGYP002782847914/1-245   DE [subseq from] FL=1
#=GS MGYP001343290792/1-246   DE [subseq from] FL=1
#=GS MGYP003180110455/28-272  DE [subseq from] FL=1
... rest of the annotation field ...

5k36_2|B|B|PROTEIN               GPDHMSRLEIYSP-EG-L-RLDG-RR-W-NE-LR--RF--E------SS-I-N--T---------H--------P-------H-----------A--------A------D-------GSSYMEQGN-N-K---I---I---T--L--V--------K------G----P--K--E-----P----R-----L----K--
MGYP003365344427/1-246           ----MSRLEIYSP-EG-L-RLDG-RR-W-NE-LR--RF--E------TS-I-N--T---------H--------P-------H-----------A--------A------D-------GSSYLEQGN-N-K---I---I---T--L--V--------K------G----P--K--E-----P----R-----L----K--
MGYP003366480418/1-243           ----MSRLEIYSP-EG-L-RLDG-RR-W-NE-LR--RF--E------SS-I-N--T---------H--------P-------H-----------A--------S------D-------GSSYLEQGN-N-K---I---I---T--L--V--------K------G----P--K--E-----P----N-----L----R--
MGYP002782847914/1-245           ----MSRVEIYSP-EG-L-RLDG-RR-W-NE-LR--RF--E------SA-I-N--T---------H--------P-------H-----------A--------A------D-------GSSYLEQGN-N-K---V---I---T--L--V--------K------G----P--K--E-----P----T-----L----K--
MGYP001343290792/1-246           ----MSRLEIYSP-EG-L-RLDG-RR-W-NE-LR--RF--E------CS-I-N--T---------H--------S-------H-----------A--------A------D-------GSSYLEQGN-N-K---V---I---T--L--V--------K------G----P--Q--E-----P----S-----S----R--
MGYP003180110455/28-272          ----MSRLEIYSP-EG-L-RLDG-RR-W-NE-LR--RF--D------CS-I-N--T---------H--------P-------N-----------A--------A------D-------GSSYLEQGN-N-K---I---I---T--L--V--------N------G----P--Q--E-----P----A-----L----R--
... rest of the sequences ...
</code></pre>
</details>

<details>
<summary>Example `sto` for PDB entry 7oxa RNA chain A ...</summary>
<pre><code>
# STOCKHOLM 1.0

#=GS 7oxa_A                           AC 7oxa_A

#=GS 7oxa_A                           DE 7oxa_A

7oxa_A                                   .GGGGCCACUAGGGACAGGAUGUUUUAGAGCUAGAAAUAGCAAGUUAAAAUAAGGCUAGUCCGUUAUCAACUUGAAAAAGUG
BA000034.2/1153816-1153905/22-68         c-----------------------------------AUAGCAAGUUAAAAUAAGGCUAGUCCGUUAUCAACUUGAAAAAGUG
#=GR BA000034.2/1153816-1153905/22-68 PP 8...................................689****************************************986
CP003068.1/774585-774496/23-68           .-----------------------------------AUAGCAAGUUAAAAUAAGGCUAGUCCGUUAUCAACUUGAAAAAGUG
#=GR CP003068.1/774585-774496/23-68   PP ....................................589****************************************986
AP011114.1/1173965-1174054/23-68         g------------------------------------UAGCAAGUUAAAAUAAGGCUAGUCCGUUAUCAACUUGAAAAAGUG
#=GR AP011114.1/1173965-1174054/23-68 PP 7....................................689***************************************986
#=GC PP_cons                             ....................................6799***************************************986
#=GC RF                                  .xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
//
</code></pre>
</details>

### 1.2. Providing Species Information for Online Pairing

Certain main MSAs can be designated to be used for cross-chain pairing in heteromeric protein complexes (see `msas_to_pair` [here](precomputed_msa_how_to.md#5-modifying-msa-settings-for-custom-precomputed-msas)). In addition to the formatting requirements above, these MSA files need to have sequence headers for all sequences **except the first sequence** in the following format:

```
<str><sep><str><sep><str><sep><species_id><sep><str><sep><str>
```

where 
- `<sep>` can be any of `|`, `_`, `/`, `:`, or `-`
- `<str>` can be any arbitrary string **not containing** the above delimiters - these fields are not used by our pairing algorithm
- `<species_id>` is the species-specific identifier - this can also be any arbitrary string as long as sequences that come from identical species have the exact same string identifier values across all alignments

So an `a3m` example with most minimal headers that looks like this

```
>query
GDPHMACNFQFPEIAYPGKLICP...
>|||HUMAN||
----LEAITATL-VGTV-RC---...
>|||HUMAN||
----LEAITATL-VGTV-RC---...
>|||7227||
----LEAITATL-VGTV-RC---...
>|||7227||
----LEAITATL-VGTV-RC---...
```

is parsed to associate the first two aligned sequences with the species identified by `HUMAN` and the last two with `7227`. See the [Online Pairing](precomputed_msa_explanation#3-online-msa-pairing) section in the precomputed MSA explanatory document for more information on how pairing is done in the inference pipeline on-the-fly.

## 2. Precomputed MSA Directory Structure and File Name Conventions

The MSA inference pipeline expects
1. the MSA files for each chain to be separated into **per-chain directories**; the names of these directories can be arbitrary strings. 
2. the MSA files generated by searching the query sequence against specific databases should have the **same filenames across chain-level directories**; the names of these files can be arbitrary strings but need to be provided in the `runner.yml` if different from the [OF3-style MSA file names](../../openfold3/projects/of3_all_atom/config/dataset_config_components.py#L30). See [Modifying MSA Settings](precomputed_msa_how_to.md#5-modifying-msa-settings-for-custom-precomputed-msas) below.

For example, if you have alignments for 3 distinct protein chains each with alignments generated using MGnify, Uniprot and a custom sequence database, the directory structure and filenames should look like below. Note that for all chains the MGnify alignments, for example, are all named identically as `mgnify_hits.a3m`.
```
alignments/
├── example_chain_A/
│   ├── mgnify_hits.a3m
│   ├── uniprot_hits.a3m
│   └── custom_database_hits.a3m
├── example_chain_B/
│   ├── mgnify_hits.a3m
│   ├── uniprot_hits.a3m
│   └── custom_database_hits.a3m
└── example_chain_C/
    ├── mgnify_hits.a3m
    ├── uniprot_hits.a3m
    └── custom_database_hits.a3m
```

## 3. Preparsing Raw MSAs into NPZ Format

For main MSAs, we provide support for preparsing MSA files from `a3m` and `sto` into `npz` numpy array format. This optional step reduces overall MSA parsing and processing times in the inference pipeline for large batch jobs with redundant sets of sequences as well as the disk space required to store these MSAs (see [this](precomputed_msa_explanation.md#5-preparsing-raw-msas-into-npz-format) section for details). The preparsed, compressed MSA files can be generated from raw MSA data with the directory and file structure specified above using [this preparsing script](../../scripts/data_preprocessing/preparse_alignments_af3.py). 

Output for the example above should look like this
```
preparsed_alignments/
├── example_chain_A.npz
├── example_chain_B.npz
└── example_chain_C.npz
```

## 4. Specifying Paths in the Inference Query File

The data pipeline needs to know which MSA to use for which protein chain. This information is provided by specifying the [paths to the MSAs](input_format.md#31-protein-chains) for each chain in the inference query json file. There are 3 equivalent ways of specifying these paths.

### 4.1. Direct File Paths

You can list the paths for all alignments for each chain. For our example of 3 chains with MGnify, Uniprot and custom database MSAs, you would specify the main_msa_paths as follows:

<details>
<summary>List of file paths example ...</summary>
<pre><code>
{
    "queries": {
        "example_query": {
            "chains": [
                {
                    "molecule_type": "protein",
                    "chain_ids": "A",
                    "sequence": "GCTLSAEDKAAVERSKMIDRNLREDGEKAAREVKLLLLGAGESGKSTIVKQMKIIHEAGYSEEECKQYKAVVYSNTIQSIIAIIRAMGRLKIDFGDAARADDARQLFVLAGAAEEGFMTAELAGVIKRLWKDSGVQACFNRSREYQLNDSAAYYLNDLDRIAQPNYIPTQQDVLRTRVKTTGIVETHFTFKDLHFKMFDVGAQRSERKKWIHCFEGVTAIIFCVALSDYDLVLAEDEEMNRMHESMKLFDSICNNKWFTDTSIILFLNKKDLFEEKIKKSPLTICYPEYAGSNTYEEAAAYIQCQFEDLNKRKDTKEIYTHFTCATDTKNVQFVFDAVTDVIIKNNLKDCGLF",
                    "main_msa_file_paths": [
                        "alignments/example_chain_A/mgnify_hits.a3m",
                        "alignments/example_chain_A/uniprot_hits.a3m",
                        "alignments/example_chain_A/custom_database_hits.a3m",
                    ]
                },
                {
                    "molecule_type": "protein",
                    "chain_ids": "B",
                    "sequence": "MSELDQLRQEAEQLKNQIRDARKACADATLSQITNNIDPVGRIQMRTRRTLRGHLAKIYAMHWGTDSRLLVSASQDGKLIIWDSYTTNKVHAIPLRSSWVMTCAYAPSGNYVACGGLDNICSIYNLKTREGNVRVSRELAGHTGYLSCCRFLDDNQIVTSSGDTTCALWDIETGQQTTTFTGHTGDVMSLSLAPDTRLFVSGACDASAKLWDVREGMCRQTFTGHESDINAICFFPNGNAFATGSDDATCRLFDLRADQELMTYSHDNIICGITSVSFSKSGRLLLAGYDDFNCNVWDALKADRAGVLAGHDNRVSCLGVTDDGMAVATGSWDSFLKIWN",
                    "main_msa_file_paths": [
                        "alignments/example_chain_B/mgnify_hits.a3m",
                        "alignments/example_chain_B/uniprot_hits.a3m",
                        "alignments/example_chain_B/custom_database_hits.a3m",
                    ]
                },
                {
                    "molecule_type": "protein",
                    "chain_ids": "C",
                    "sequence": "MASNNTASIAQARKLVEQLKMEANIDRIKVSKAAADLMAYCEAHAKEDPLLTPVPASENPFREKKFFSAIL",
                    "main_msa_file_paths": [
                        "alignments/example_chain_C/mgnify_hits.a3m",
                        "alignments/example_chain_C/uniprot_hits.a3m",
                        "alignments/example_chain_C/custom_database_hits.a3m",
                    ]
                },
            ],
        }
    }
}
</code></pre>
</details>

### 4.2. Folder Containing Alignments per Chain

You may also pass in the chain-level directory containing the alignments relevant to the chain. In this case, the contents of the directory should still contain individual files as above.

<details>
<summary>Directory path example ...</summary>
<pre><code>
{
    "queries": {
        "example_query": {
            "chains": [
                {
                    "molecule_type": "protein",
                    "chain_ids": "A",
                    "sequence": "GCTLSAEDKAAVERSKMIDRNLREDGEKAAREVKLLLLGAGESGKSTIVKQMKIIHEAGYSEEECKQYKAVVYSNTIQSIIAIIRAMGRLKIDFGDAARADDARQLFVLAGAAEEGFMTAELAGVIKRLWKDSGVQACFNRSREYQLNDSAAYYLNDLDRIAQPNYIPTQQDVLRTRVKTTGIVETHFTFKDLHFKMFDVGAQRSERKKWIHCFEGVTAIIFCVALSDYDLVLAEDEEMNRMHESMKLFDSICNNKWFTDTSIILFLNKKDLFEEKIKKSPLTICYPEYAGSNTYEEAAAYIQCQFEDLNKRKDTKEIYTHFTCATDTKNVQFVFDAVTDVIIKNNLKDCGLF",
                    "main_msa_file_paths": "alignments/example_chain_A"
                },
                {
                    "molecule_type": "protein",
                    "chain_ids": "B",
                    "sequence": "MSELDQLRQEAEQLKNQIRDARKACADATLSQITNNIDPVGRIQMRTRRTLRGHLAKIYAMHWGTDSRLLVSASQDGKLIIWDSYTTNKVHAIPLRSSWVMTCAYAPSGNYVACGGLDNICSIYNLKTREGNVRVSRELAGHTGYLSCCRFLDDNQIVTSSGDTTCALWDIETGQQTTTFTGHTGDVMSLSLAPDTRLFVSGACDASAKLWDVREGMCRQTFTGHESDINAICFFPNGNAFATGSDDATCRLFDLRADQELMTYSHDNIICGITSVSFSKSGRLLLAGYDDFNCNVWDALKADRAGVLAGHDNRVSCLGVTDDGMAVATGSWDSFLKIWN",
                    "main_msa_file_paths": "alignments/example_chain_B"
                },
                {
                    "molecule_type": "protein",
                    "chain_ids": "C",
                    "sequence": "MASNNTASIAQARKLVEQLKMEANIDRIKVSKAAADLMAYCEAHAKEDPLLTPVPASENPFREKKFFSAIL",
                    "main_msa_file_paths": "alignments/example_chain_C"
                },
            ],
        }
    }
}
</code></pre>
</details>

### 4.3. Preparsed NPZ File Containing Contents of All Alignment Files

If you opted to preparse the raw alignment files into NPZ files, you should specify the NPZ file paths in the query json file.

<details>
<summary>NPZ file example ...</summary>
<pre><code>
{
    "queries": {
        "example_query": {
            "chains": [
                {
                    "molecule_type": "protein",
                    "chain_ids": "A",
                    "sequence": "GCTLSAEDKAAVERSKMIDRNLREDGEKAAREVKLLLLGAGESGKSTIVKQMKIIHEAGYSEEECKQYKAVVYSNTIQSIIAIIRAMGRLKIDFGDAARADDARQLFVLAGAAEEGFMTAELAGVIKRLWKDSGVQACFNRSREYQLNDSAAYYLNDLDRIAQPNYIPTQQDVLRTRVKTTGIVETHFTFKDLHFKMFDVGAQRSERKKWIHCFEGVTAIIFCVALSDYDLVLAEDEEMNRMHESMKLFDSICNNKWFTDTSIILFLNKKDLFEEKIKKSPLTICYPEYAGSNTYEEAAAYIQCQFEDLNKRKDTKEIYTHFTCATDTKNVQFVFDAVTDVIIKNNLKDCGLF",
                    "main_msa_file_paths": "preparsed_alignments/example_chain_A.npz"
                },
                {
                    "molecule_type": "protein",
                    "chain_ids": "B",
                    "sequence": "MSELDQLRQEAEQLKNQIRDARKACADATLSQITNNIDPVGRIQMRTRRTLRGHLAKIYAMHWGTDSRLLVSASQDGKLIIWDSYTTNKVHAIPLRSSWVMTCAYAPSGNYVACGGLDNICSIYNLKTREGNVRVSRELAGHTGYLSCCRFLDDNQIVTSSGDTTCALWDIETGQQTTTFTGHTGDVMSLSLAPDTRLFVSGACDASAKLWDVREGMCRQTFTGHESDINAICFFPNGNAFATGSDDATCRLFDLRADQELMTYSHDNIICGITSVSFSKSGRLLLAGYDDFNCNVWDALKADRAGVLAGHDNRVSCLGVTDDGMAVATGSWDSFLKIWN",
                    "main_msa_file_paths": "preparsed_alignments/example_chain_B.npz"
                },
                {
                    "molecule_type": "protein",
                    "chain_ids": "C",
                    "sequence": "MASNNTASIAQARKLVEQLKMEANIDRIKVSKAAADLMAYCEAHAKEDPLLTPVPASENPFREKKFFSAIL",
                    "main_msa_file_paths": "preparsed_alignments/example_chain_C.npz"
                },
            ],
        }
    }
}
</code></pre>
</details>

### 4.4. Pre-Paired MSAs in the Inference Query

If you want to use your own pre-paired MSAs, perhaps pre-paired using a custom pairing algorithm, you can specify paths using any of the previous three methods to the per-chain paired MSAs using the `paired_msa_file_paths` field in your input query json file.

## 5. Modifying MSA Settings for Custom Precomputed MSAs

In the inference pipeline, we use the [`MSASettings`](../../openfold3/projects/of3_all_atom/config/dataset_config_components.py#L18) class to control MSA processing and featurization. You can update it using the dataset_config_kwargs section in the `runner.yml`. Updates to `MSASettings` via the `runner.yml` **overwrite the corresponding default fields**. The MSASettings do **not** need to be updated when using OF3-style protein MSAs.

For our running example of 3 chains with alignments stored under `uniprot_hits`, `mgnify_hits` and `custom_database_hits` files, an `MSASettings` update via the `runner.yml` could look like this (refer to the [main inference document](Inference.md#33-customized-inference-settings-using-runneryml) for more details):

```
dataset_config_kwargs:
  msa:
    max_seq_counts:  
      uniprot_hits: 50000
      mgnify_hits: 5000
      custom_database_hits: 10000
    msas_to_pair: ["uniprot_hits"]
    aln_order:   
      - uniprot_hits
      - mgnify_hits
      - custom_database_hits
```

Where `max_seq_counts` specifies the maximum number of sequences to use from each file, `msas_to_pair` specifies which files to use for online pairing and `aln_order` instructs the pipeline to vertically concatenate the MSAs in the order `uniprot_hits`-`mgnify_hits`-`custom_database_hits` from top to bottom. Refer to the [Precomputed MSA Explanation Document](precomputed_msa_explanation.md#2-msasettings-reference) for further details on modifying MSASettings.

The `runner.yml` can then be passed as a command line argument to `run_openfold.py`:

```
python run_openfold.py predict \
--query_json query_precomputed_full_path.json \
--use_msa_server=False \
--inference_ckpt_path=of3_v14_79-32000_converted.ckpt.pt \
--output_dir=precomputed_prediction_output/ \
--runner_yaml=inference_precomputed.yml 
```
