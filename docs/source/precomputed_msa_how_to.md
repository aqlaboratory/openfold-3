# A How-To Guide for Precomputed MSAs in the OF3 Inference Pipeline

In this document, we intend to provide a guide on how to format and organize precomputed multiple sequence alignments (MSAs) and how to provide settings for the Openfold3 inference pipeline to use these MSAs correctly for creating MSA features for Openfold3. Use this guide if you already generated MSAs using your own or our internal OF3-style pipeline. If you have yet to generate MSAs and would like to use our workflow, refer to our [MSA Generation Guide](msa_generation_how_to.md). If you need further clarifications on how some of the MSA components of our inference pipeline work, refer to [this explanatory document](precomputed_msas_explanation.md).

The main steps detailed in this guide are:
1. [Providing the MSAs in the expected format](precomputed_msa_how_to.md#1-precomputed-msa-file-format)
2. [Organizing the MSAs in the expected directory structure](precomputed_msa_how_to.md#2-precomputed-msa-directory-structure-and-file-name-conventions)
3. [Preparsing MSAs into NPZ format](precomputed_msa_how_to.md#3-preparsing-raw-msas-into-npz-format)
4. [Adding the MSA file/directory paths to the inference query json](precomputed_msa_how_to.md#4-specifying-paths-in-the-inference-query-file)
5. [Updating the MSA pipeline settings](precomputed_msa_how_to.md#5-modifying-msa-settings-for-custom-precomputed-msas)

If you intend to use your own, custom pipeline for generating MSAs, we advise consulting steps 1 and 2 beforehand. Steps 1, 2 and 5 can be skipped if using our OF3-style MSA generation pipeline. Step 3 is optional.

TODO: add paired MSA steps, update links

## 1. Precomputed MSA Files

TODO: clarify main vs paired MSA here

### 1.1. General MSA Format

This section details the format of the MSA files provided by our Snakemake pipeline for generating protein MSAs. MSAs generated with a different method should follow the same format:
- the MSAs can be in either `a3m` or `sto` format
- the first sequence in the MSA must be the query sequence i.e.: the protein or RNA sequence for which the structure is to be predicted

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

```
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
```
</code></pre>
</details>

<details>
<summary>Example `sto` for PDB entry 7oxa RNA chain A ...</summary>
<pre><code>

```
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
```
</code></pre>
</details>

### 1.2. Providing Species Information for Online Pairing

TODO: write

## 2. Precomputed MSA Directory Structure and File Name Conventions

The MSA inference pipeline expects 
1. the MSA files for each chain to be separated into **per-chain directories**; the names of these directories can be arbitrary strings. 
2. the MSA files generated by searching the query sequence against specific databases should have the **same filenames across chain-level directories**; the names of these files can be arbitrary strings but need to be provided in the `runner.yml` if different from the [OF3-style MSA file names](../../openfold3/projects/of3_all_atom/config/dataset_config_components.py#L30). See [Modifying MSA Settings](precomputed_msas.md#4-modifying-msa-settings-for-custom-precomputed-msas) below.

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

TODO: finish, add link to explanation

The compressed, preparsed MSA files can be generated from raw MSA data with the directory and file structure specified above using [this preparsing script](../../scripts/data_preprocessing/preparse_alignments_af3.py). More detailed documentation on use cases for npz files and how to create them will be provided in the next internal release. 

Output for the example above should look like this
```
preparsed_alignments/
├── example_chain_A.npz
├── example_chain_B.npz
└── example_chain_C.npz
```

## 4. Specifying Paths in the Inference Query File

The data pipeline needs to know which MSA to use for which protein chain. This information is provided by specifying the [paths to the MSAs](input_format.md#L106) for each chain in the inference query json file. There are 3 equivalent ways of specifying these paths.

### 4.1. Direct File Paths

The direct paths for all alignments for each chain can be passed into the query.json. For our example of 3 chains with MGnify, Uniprot and custom database MSAs, you would specify the main_msa_paths as follows:

```
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
```

### 4.2. Folder Containing Alignments per Chain

You may also pass in the chain-level directory containing the alignments relevant to the chain. In this case, the contents of the directory should still contain individual files that contain the msa database name.

```
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
```

### 4.3. Preparsed NPZ File Containing Contents of All Alignment Files

If you opted to preparse the raw alignment files into NPZ files, you can also specify the NPZ file paths in the query json file.

```
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
```

## 5. Modifying MSA Settings for Custom Precomputed MSAs

In the inference pipeline, we use the [MSASettings](../../openfold3/projects/of3_all_atom/config/dataset_config_components.py#L18) class to control MSA processing and featurization. You can update it using the dataset_config_kwargs section in the `runner.yml`. Updates to `MSASettings` via the `runner.yml` **overwrite the corresponding default fields**. The MSASettings do **not** need to be updated when using OF3-style protein MSAs.

For our example, an `MSASettings` update could look like this:

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

Where `max_seq_counts` specifies the maximum number of sequences to use from each file, `msas_to_pair` specifyies which files to use for online pairing and `aln_order` instructs the pipeline to vertically concatenate the MSAs in the order `uniprot_hits`-`mgnify_hits`-`custom_database_hits` from top to bottom. Refer to the [Precomputed MSA Explanation Document](precomputed_msa_explanation.md#1-msasettings) for further details on modifying MSASettings.

The `runner.yml` can then be passed as a command line argument to `run_openfold.py`:

```
python run_openfold.py predict \
--query_json query_precomputed_full_path.json \
--use_msa_server=False \
--inference_ckpt_path=of3_v14_79-32000_converted.ckpt.pt \
--output_dir=precomputed_prediction_output/ \
--runner_yaml=inference_precomputed.yml 
```
