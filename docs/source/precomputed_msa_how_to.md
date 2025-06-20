# A How-To Guide for Precomputed MSAs in the OF3 Inference Pipeline

In this document, we intend to provide a guide on how to format and organize precomputed multiple sequence alignments (MSAs) and provide settings for the Openfold3 inference pipeline to use them. Use this guide if you already generated MSAs using your own or our internal OF3-style pipeline. If you have yet to generate MSAs and would like to use our workflow, refer to our [MSA Generation Guide](msa_generation_how_to.md). If you need further clarifications on how the MSA components of our inference pipeline, refer to [this explanatory document](precomputed_msas_explanation.md).

Once all MSAs are generated, you need to
1. [Provide the MSAs in the expected format](precomputed_msas_how_to.md#1-precomputed-msa-file-format)
2. [Organize the MSAs in the expected directory structure](precomputed_msas_how_to.md#2-precomputed-msa-directory-structure-and-file-name-conventions)
3. [Add the MSA file/directory paths to the inference query json](precomputed_msas_how_to.md#3-specifying-paths-in-the-inference-query-file)
4. [Update the MSA pipeline settings](precomputed_msas_how_to.md#4-modifying-msa-settings-for-custom-precomputed-msas)

Steps 1, 2 and 4 can be skipped if using our OF3-style MSA generation pipeline. All of these steps are detailed below.

## 1. Precomputed MSA File Format

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

The MSA inference pipeline expects the MSA files for each chain to be separated into **per-chain directories**; the names of these directories can be arbitrary strings. Further, the MSA files generated by searching the query sequence against specific databases should have the **same filename across chain-level directories**; the names of these files can be arbitrary strings but need to be provided in the `runner.yml` if different from the OF3-style MSA file names. See [Modifying MSA Settings](precomputed_msas.md#4-modifying-msa-settings-for-custom-precomputed-msas) below.

For example, for 3 distinct protein chains each with MGnify and Uniprot alignments:
```
msas/
├── 5k36_B/
│   ├── mgnify_hits.a3m
│   └── uniprot_hits.a3m
├── 5k36_D/
│   ├── mgnify_hits.a3m
│   └── uniprot_hits.a3m
└── 5k36_E/
    ├── mgnify_hits.a3m
    └── uniprot_hits.a3m
```

## 3. Specifying Paths in the Inference Query File

The data pipeline needs to know which MSA to use for which protein chain. This information is provided by specifying the [paths to the MSAs](input_format.md#L106) for each chain in the inference query json file. There are 3 equivalent ways of specifying these paths.

### 3.1. Direct File Paths

The direct paths for all alignments for each chain can be passed into the query.json. In this case, you would specify the main_msa_paths as follows:

```
{
    "queries": {
        "7cnx": {
            "query_name": "7cnx",
            "chains": [
                {
                    "molecule_type": "protein",
                    "chain_ids": [
                        "A",
                        "C"
                    ],
                    "sequence": "MLNSFKLSLQY...",
                    "main_msa_file_paths": [
                        "alignments/7cnw_A/cfdb_uniref30.a3m",
                        "alignments/7cnw_A/mgnify_hits.sto",
                        "alignments/7cnw_A/uniprot_hits.sto",
                        "alignments/7cnw_A/uniref90_hits.sto"
                    ]
                },
                ...
            ],
	        "use_msas": true,
            "use_paired_msas": true,
            "use_main_msas": true
        }
    }
}
```

The names of these files do need to match the msa names provided to the inference dataset pipelines. [Below](precomputed_msas.md#4-modifying-msa-settings-for-custom-precomputed-msas) is an example of how to provide these msa file names to the inference dataset by overwriting the values in the runner.json.

### 3.2. Folder Containing Alignments per Chain

You may also pass in the directory containing the alignments relevant to the chain. In this case, the contents of the directory should still contain individual files that contain the msa database name.

```
{
    "queries": {
        "7cnx": {
            "query_name": "7cnx",
            "chains": [
                {
                    "molecule_type": "protein",
                    "chain_ids": [
                        "A",
                        "C"
                    ],
                    "sequence": "MLNSFKLSLQY...",
                    "main_msa_file_paths": [
                        "alignments/7cnw_A/",
                    ]
                },
                ...
            ],
			"use_msas": true,
            "use_paired_msas": true,
            "use_main_msas": true
        }
    }
}
```

### 3.3. Compressed NPZ File Containing Contents of All Alignment Files

It is also an option to pass in compressed npz files containing the alignment data above. This is the default way alignments from the ColabFold MSA server are stored.

```
{
    "queries": {
        "7cnx": {
            "query_name": "7cnx",
            "chains": [
                {
                    "molecule_type": "protein",
                    "chain_ids": [
                        "A",
                        "C"
                    ],
                    "sequence": "MLNSFKLSLQY...",
                    "main_msa_file_paths": [
                        "alignments/7cnw_A.npz",
                    ]
                },
                ...
            ],
	        "use_msas": true,
            "use_paired_msas": true,
            "use_main_msas": true
        }
    }
}
```

The compressed, preparsed MSA files can be generated from raw MSA data with the directory and file structure specified above using [this preparsing script](../../scripts/data_preprocessing/preparse_alignments_af3.py). More detailed documentation on use cases for npz files and how to create them will be provided in the next internal release. 

## 4. Modifying MSA Settings for Custom Precomputed MSAs

In the inference pipeline, we use the [MSASettings](../../openfold3/projects/of3_all_atom/config/dataset_config_components.py) class to control MSA processing and featurization. 

More details on the MSASettings will be shared in the next internal release. Brief notes on settings that need to be updated when working with custom precomputed alignments are below. 

It is possible to pass in an update to the MSASettings to match your alignments using the dataset_config_kwargs section in the `runner.yml`, for example:

```
dataset_config_kwargs:
  msa:
    max_seq_counts:  
      uniref90_hits: 10000
      uniprot_hits: 50000
      cfdb_uniref30: 10000000
      mgnify_hits: 5000
    msas_to_pair: ["uniprot_hits", "uniprot"]
    aln_order:   
      - uniref90_hits
      - cfdb_uniref30
      - mgnify_hits
      - concat_cfdb_uniref100_filtered
```

This runner yaml can then be passed as a command line argument to our run_openfold.py command:

```
python run_openfold.py predict \
--query_json query_precomputed_full_path.json \
--use_msa_server=False \
--inference_ckpt_path=of3_v14_79-32000_converted.ckpt.pt \
--output_dir=precomputed_prediction_output/ \
--runner_yaml=inference_precomputed.yml 
```

📝  *Note:*
- The MSASettings do NOT need to be updated when using OF3-style protein MSAs.

The 3 main settings to update are:
1. *max_seq_counts*: A dictionary specifying how many sequences to read from each MSA file with the associated name. MSA files whose names are not provided in this dictionary *will not be parsed*. For example, if one wants `uniparc_hits.a3m` MSA files to be parsed, the following field should be specified:

```
dataset_config_kwargs:
  msa:
    max_seq_counts:  
      uniparc_hits: 10000 
```

where the up to the first 10000 sequences will be read from each `uniparc_hits.a3m` file.

2. *msas_to_pair*: The list of MSA filenames that contain species information that can be used for online pairing. See the [Online MSA Pairing](precomputed_msas.md#6-online-msa-pairing-from-precomputed-msas) section for details.

3. *aln_order*: The order in which to vertically concatenate MSA files for each chain for main MSA features. MSA files whose names are not provided in this list *will not be used*. For example, if one has MSA files named `mgnify_hits`, `uniprot_hits` and `uniparc_hits` and want to vertically concatenate them for each chain in this order, they should update the `runner.yml` as follows:

```
dataset_config_kwargs:
  msa:
    aln_order:   
     - mgnify_hits
     - uniprot_hits
     - uniparc_hits
```