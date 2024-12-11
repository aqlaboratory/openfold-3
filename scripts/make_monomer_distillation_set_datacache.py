#%%
import subprocess as sp 
from copy import deepcopy
import json 
import click 
import boto3

@click.command()
@click.option(
    "--s3_profile",
    type=str,
    default="openfold",
    help="AWS profile to use for S3 access."
    "Must be set up following instructions pinned in RCP slack"
)

@click.option(
    "--s3_bucket",
    default="rcp-openfold-dataset-prd-361769554172",
    type=str,
    help="S3 bucket name to search for monomer distillation data."
)

@click.option(
    "--s3_prefix",
    default="monomer_distill_clean_data_v3",
    type=str,
    help="S3 prefix to search for monomer distillation data."
)

@click.option(
    "--max_seq_counts",
    default=100000,
    type=int,
    help="Maximum number of sequences to include in the MSA."
)

@click.option(
    "--out_file",
    default="monomer_distillation_set_datacache.json",
    type=str,
    help="Output file to save the datacache."
)

def main(
    s3_profile: str, 
    s3_bucket: str,
    s3_prefix: str,
    max_seq_counts: int,
    out_file: str
):
    base_dict ={
                "chains":{
                    "1":{
                        "label_asym_id": "1",
                        "auth_asym_id": "A",
                        "entity_id": "1",
                        "molecule_type": "PROTEIN",
                        "reference_mol_id": None ,
                        "alignment_representative_id":"",
                        "template_ids": []
                    }
                },
                "release_date": "1999-09-09",
                "resolution": 100000.0 ## no confidence losses on monomer distillation set, so just make it really high so trip the filter. 
            }
    
    avail_mgy_ids = get_sample_ids(s3_bucket, s3_prefix, s3_profile)
    avail_mgy_ids = [mgy_id for mgy_id in avail_mgy_ids if mgy_id.startswith("M")]

    out_dict = {
        "_type": "MonomerDistillationDataCache",
        "name" : "LongMonomerDistillationSet",
        "s3_data":{
            "profile": s3_profile,
            "bucket": s3_bucket,
            "prefix": s3_prefix        
        },
        "max_seq_counts": {
            "concat_cfdb_uniref100_filtered": max_seq_counts
        },
        "structure_data":{

        }
    }
    for mgy_id in avail_mgy_ids[:5]:
        new_dict = deepcopy(base_dict)
        new_dict["chains"]["1"]["alignment_representative_id"] = mgy_id
        out_dict["structure_data"][mgy_id] = new_dict
    
    import json 
    with open(out_file, "w") as f: 
        json.dump(out_dict, f, indent = 4)


def get_sample_ids(bucket_name, prefix, profile):
    session = boto3.Session(profile_name=profile)
    s3_client = session.client('s3')
    paginator = s3_client.get_paginator('list_objects_v2')
    operation_parameters = {'Bucket': bucket_name, 'Prefix': prefix}

    sample_ids = set()

    for page in paginator.paginate(**operation_parameters):
        if 'Contents' in page:
            for obj in page['Contents']:
                key = obj['Key']
                # Extract sample_id from the key
                parts = key.split('/')
                if len(parts) > 2:  # Ensure the key has the structure
                    sample_id = parts[-2]
                    sample_ids.add(sample_id)

    return list(sample_ids)


if __name__ == "__main__":
    main()
