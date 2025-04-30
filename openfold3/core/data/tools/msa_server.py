import glob
import json
import logging
import os
import random
import tarfile
import time
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


TQDM_BAR_FORMAT = (
    "{l_bar}{bar}| {n_fmt}/{total_fmt} [elapsed: {elapsed} remaining: {remaining}]"
)


### This function is copied directly from the colabfold codebase:
### https://github.com/sokrypton/ColabFold/blob/main/colabfold/colabfold.py
### with a few minor modifications.
def query_msa_server(
    x: list[str],
    prefix: str,
    use_env: bool = True,
    use_filter: bool = True,
    use_templates: bool = False,
    filter: Optional[bool] = None,
    use_pairing: bool = False,
    pairing_strategy: str = "greedy",
    host_url: str = "https://api.colabfold.com",
    user_agent: str = "openfold/1.0.0 vss2134@cumc.columbia.edu",
) -> tuple[list[str], list[str]]:
    """_summary_

    Args:
        x (List[str]): list of amino acid sequences to run
        prefix (str): output directory to save the results
        use_env (bool, optional): Whether to align against env db(BFD/cfdb)
            Defaults to True.
        use_filter (bool, optional): Whether to apply diversity filter.
            Defaults to True.
        use_templates (bool, optional): Whether to run template search.
            Defaults to False.
        filter (Optional[bool], optional): Legacy option to enable diversity filter.
            Defaults to None.
        use_pairing (bool, optional): Whether to generate a single paired MSA
            Defaults to False.
        pairing_strategy (str, optional): Pairing method, one of ["complete", "greedy"].
            For pairing of more than 2 chains, "complete" requires a taxonomic group to
            be present in all chains, where greedy requires it to be in only 2
        host_url (_type_, optional): host url for MSA server.
            Defaults to "https://api.colabfold.com".
        user_agent (str, optional): User associated with API call.
            Defaults to "openfold/1.0.0 vss2134@cumc.columbia.edu".

    Returns:
        Tuple[List[str], List[str]]: A tuple containing the list of MSA lines
            and the list of template paths
    """

    submission_endpoint = "ticket/pair" if use_pairing else "ticket/msa"

    headers = {}
    if user_agent != "":
        headers["User-Agent"] = user_agent
    else:
        logger.warning(
            "No user agent specified. Please set a user agent"
            "(e.g., 'toolname/version contact@email') to help"
            "us debug in case of problems. This warning will become an error"
            "in the future."
        )

    def submit(seqs, mode, N=101):
        n, query = N, ""
        for seq in seqs:
            query += f">{n}\n{seq}\n"
            n += 1

        while True:
            error_count = 0
            try:
                # https://requests.readthedocs.io/en/latest/user/advanced/#advanced
                # "good practice to set connect timeouts to slightly larger
                # than a multiple of 3"
                res = requests.post(
                    f"{host_url}/{submission_endpoint}",
                    data={"q": query, "mode": mode},
                    timeout=6.02,
                    headers=headers,
                )
            except requests.exceptions.Timeout:
                logger.warning("Timeout while submitting to MSA server. Retrying...")
                continue
            except Exception as e:
                error_count += 1
                logger.warning(
                    f"Error while fetching result from MSA server."
                    f"Retrying... ({error_count}/5)"
                )
                logger.warning(f"Error: {e}")
                time.sleep(5)
                if error_count > 5:
                    raise
                continue
            break

        try:
            out = res.json()
        except ValueError:
            logger.error(f"Server didn't reply with json: {res.text}")
            out = {"status": "ERROR"}
        return out

    def status(ID):
        while True:
            error_count = 0
            try:
                res = requests.get(
                    f"{host_url}/ticket/{ID}", timeout=6.02, headers=headers
                )
            except requests.exceptions.Timeout:
                logger.warning(
                    "Timeout while fetching status from MSA server. Retrying..."
                )
                continue
            except Exception as e:
                error_count += 1
                logger.warning(
                    f"Error while fetching result from MSA server."
                    f"Retrying... ({error_count}/5)"
                )
                logger.warning(f"Error: {e}")
                time.sleep(5)
                if error_count > 5:
                    raise
                continue
            break
        try:
            out = res.json()
        except ValueError:
            logger.error(f"Server didn't reply with json: {res.text}")
            out = {"status": "ERROR"}
        return out

    def download(ID, path):
        error_count = 0
        while True:
            try:
                res = requests.get(
                    f"{host_url}/result/download/{ID}", timeout=6.02, headers=headers
                )
            except requests.exceptions.Timeout:
                logger.warning(
                    "Timeout while fetching result from MSA server. Retrying..."
                )
                continue
            except Exception as e:
                error_count += 1
                logger.warning(
                    f"Error while fetching result from MSA server."
                    f"Retrying... ({error_count}/5)"
                )
                logger.warning(f"Error: {e}")
                time.sleep(5)
                if error_count > 5:
                    raise
                continue
            break
        with open(path, "wb") as out:
            out.write(res.content)

    # process input x
    seqs = [x] if isinstance(x, str) else x

    # compatibility to old option
    if filter is not None:
        use_filter = filter

    # setup mode
    if use_filter:
        mode = "env" if use_env else "all"
    else:
        mode = "env-nofilter" if use_env else "nofilter"

    if use_pairing:
        use_templates = False
        mode = ""
        # greedy is default, complete was the previous behavior
        if pairing_strategy == "greedy":
            mode = "pairgreedy"
        elif pairing_strategy == "complete":
            mode = "paircomplete"
        if use_env:
            mode = mode + "-env"

    ## VS: put everything in the same folder
    path = f"{prefix}"
    if not os.path.isdir(path):
        os.mkdir(path)

    # call mmseqs2 api
    tar_gz_file = f"{path}/out.tar.gz"
    N, REDO = 101, True

    # deduplicate and keep track of order
    seqs_unique = []
    # TODO this might be slow for large sets
    [seqs_unique.append(x) for x in seqs if x not in seqs_unique]
    Ms = [N + seqs_unique.index(seq) for seq in seqs]
    # lets do it!
    if not os.path.isfile(tar_gz_file):
        TIME_ESTIMATE = 150 * len(seqs_unique)
        with tqdm(total=TIME_ESTIMATE, bar_format=TQDM_BAR_FORMAT) as pbar:
            while REDO:
                pbar.set_description("SUBMIT")

                # Resubmit job until it goes through
                out = submit(seqs_unique, mode, N)
                while out["status"] in ["UNKNOWN", "RATELIMIT"]:
                    sleep_time = 5 + random.randint(0, 5)
                    logger.error(f"Sleeping for {sleep_time}s. Reason: {out['status']}")
                    # resubmit
                    time.sleep(sleep_time)
                    out = submit(seqs_unique, mode, N)

                if out["status"] == "ERROR":
                    raise Exception(
                        "MMseqs2 API is giving errors."
                        "Please confirm your input is a valid protein sequence."
                        "If error persists, please try again an hour later."
                    )

                if out["status"] == "MAINTENANCE":
                    raise Exception(
                        "MMseqs2 API is undergoing maintenance."
                        "Please try again in a few minutes."
                    )

                # wait for job to finish
                ID, TIME = out["id"], 0
                pbar.set_description(out["status"])
                while out["status"] in ["UNKNOWN", "RUNNING", "PENDING"]:
                    t = 5 + random.randint(0, 5)
                    logger.error(f"Sleeping for {t}s. Reason: {out['status']}")
                    time.sleep(t)
                    out = status(ID)
                    pbar.set_description(out["status"])
                    if out["status"] == "RUNNING":
                        TIME += t
                        pbar.update(n=t)
                    # if TIME > 900 and out["status"] != "COMPLETE":
                    #  # something failed on the server side, need to resubmit
                    #  N += 1
                    #  break

                if out["status"] == "COMPLETE":
                    if TIME < TIME_ESTIMATE:
                        pbar.update(n=(TIME_ESTIMATE - TIME))
                    REDO = False

                if out["status"] == "ERROR":
                    REDO = False
                    raise Exception(
                        "MMseqs2 API is giving errors."
                        "Please confirm your input is a valid protein sequence."
                        "If error persists, please try again an hour later."
                    )

            # Download results
            download(ID, tar_gz_file)

    # prep list of a3m files
    if use_pairing:
        a3m_files = [f"{path}/pair.a3m"]
    else:
        a3m_files = [f"{path}/uniref.a3m"]
        if use_env:
            a3m_files.append(f"{path}/bfd.mgnify30.metaeuk30.smag30.a3m")

    # extract a3m files
    if any(not os.path.isfile(a3m_file) for a3m_file in a3m_files):
        with tarfile.open(tar_gz_file) as tar_gz:
            tar_gz.extractall(path)

    # templates
    if use_templates:
        templates = {}
        # print("seq\tpdb\tcid\tevalue")
        for line in open(f"{path}/pdb70.m8"):  # noqa: SIM115
            p = line.rstrip().split()
            M, pdb, qid, e_value = p[0], p[1], p[2], p[10]  # noqa: F841
            M = int(M)
            if M not in templates:
                templates[M] = []
            templates[M].append(pdb)
            # if len(templates[M]) <= 20:
            #  print(f"{int(M)-N}\t{pdb}\t{qid}\t{e_value}")

        template_paths = {}
        for k, TMPL in templates.items():
            TMPL_PATH = f"{prefix}/templates_{k}"
            if not os.path.isdir(TMPL_PATH):
                os.mkdir(TMPL_PATH)
                TMPL_LINE = ",".join(TMPL[:20])
                response = None
                while True:
                    error_count = 0
                    try:
                        # "good practice to set connect timeouts to slightly
                        # larger than a multiple of 3"
                        response = requests.get(
                            f"{host_url}/template/{TMPL_LINE}",
                            stream=True,
                            timeout=6.02,
                            headers=headers,
                        )
                    except requests.exceptions.Timeout:
                        logger.warning(
                            "Timeout while submitting to template server. Retrying..."
                        )
                        continue
                    except Exception as e:
                        error_count += 1
                        logger.warning(
                            f"Error while fetching result from template server."
                            f"Retrying... ({error_count}/5)"
                        )
                        logger.warning(f"Error: {e}")
                        time.sleep(5)
                        if error_count > 5:
                            raise
                        continue
                    break
                with tarfile.open(fileobj=response.raw, mode="r|gz") as tar:
                    tar.extractall(path=TMPL_PATH)
                os.symlink("pdb70_a3m.ffindex", f"{TMPL_PATH}/pdb70_cs219.ffindex")
                with open(f"{TMPL_PATH}/pdb70_cs219.ffdata", "w") as f:
                    f.write("")
            template_paths[k] = TMPL_PATH

    # gather a3m lines
    a3m_lines = {}
    for a3m_file in a3m_files:
        update_M, M = True, None
        for line in open(a3m_file):  # noqa: SIM115
            if len(line) > 0:
                if "\x00" in line:
                    line = line.replace("\x00", "")
                    update_M = True
                if line.startswith(">") and update_M:
                    M = int(line[1:].rstrip())
                    update_M = False
                    if M not in a3m_lines:
                        a3m_lines[M] = []
                a3m_lines[M].append(line)

    # return results

    a3m_lines = ["".join(a3m_lines[n]) for n in Ms]

    if use_templates:
        template_paths_ = []
        for n in Ms:
            if n not in template_paths:
                template_paths_.append(None)
                # print(f"{n-N}\tno_templates_found")
            else:
                template_paths_.append(template_paths[n])
        template_paths = template_paths_

    return (a3m_lines, template_paths) if use_templates else a3m_lines


def run_alignments_msaserver(
    seq_dict: dict[str, str],
    structure_id: str,
    output_directory: Path,
    use_templates: bool,
    prelim_dataset_cache: dict,
) -> None:
    """Run alignments using the colabfold MSA server

    Args:
        seq_dict (dict[str, str]): A dictionary of that maps representative ID
            to amino acid sequence
        structure_id (str): The ID of the structure. This is used to create the
            output directory and to update the dataset cache accordingly
        output_directory (Path): Root Path to save the output files. Assumed to be the
            name of the structure. chain MSAs will be written to separate output
            directories ie path/to/<pdb_id>/msas/chain_1/...
        use_templates (bool): Whether or not to run template search
        prelim_dataset_cache (dict): A dictionary containing the dataset cache
            for the structure. This function will only update the "chain" field of the
            dataset cache
    Notes:
    - parsing templates to structures arrays not implemented
    - need to update the dataset cache to use the appropriate cache Class
    - optionally silence msa server logger.
    """
    ## generate representative mapping
    useq2repr = {}
    repr2useq = {}
    chainid2repr = {}
    for rep_id, seq in seq_dict.items():
        if seq not in useq2repr:
            useq2repr[seq] = rep_id
            repr2useq[rep_id] = seq
            chainid2repr[rep_id] = rep_id
        else:
            chainid2repr[rep_id] = useq2repr[seq]

    output_directory = (
        output_directory / structure_id
    )  ## only ever handling data from one structure at a time
    (output_directory / "msas").mkdir(parents=True, exist_ok=True)
    ## run msa server
    ### if more than 1 unique chain, run in paired mode.
    if len(repr2useq.keys()) > 1:
        _ = query_msa_server(
            list(repr2useq.values()),
            f"{output_directory}/msas",
            use_pairing=True,
            pairing_strategy="greedy",
            use_templates=False,
        )
        ## delete extra files since we write both paired and unpaired MSAs
        ## to the same output directory for now.
        Path(f"{output_directory}/msas/out.tar.gz").unlink(missing_ok=True)
        Path(f"{output_directory}/msas/pair.sh").unlink(missing_ok=True)
    ## generate MSAs for each chain
    ### Note that alignmetns for each are written to the same
    ### file and are separated by a null character.
    msa_lines_unpaired, template_path = query_msa_server(
        list(repr2useq.values()),
        f"{output_directory}/msas",
        use_pairing=False,
        use_templates=use_templates,
    )
    # template path will contain None for chains w/o templates
    template_path = [p for p in template_path if p is not None]
    ## mmseqs uses an internal numbering scheme for sequences -
    # we need to map this back to the original chain IDs
    mmseqs_cid2repr = {}
    repr2mmseqs_cid = {}
    for alignment in msa_lines_unpaired:
        alignment = alignment.split("\n")
        mmseqs_chain_id, chain_aaseq = alignment[:2]
        for repr_id, aa_seq in repr2useq.items():
            if aa_seq == chain_aaseq:
                mmseqs_chain_id = mmseqs_chain_id.strip(">")
                mmseqs_cid2repr[mmseqs_chain_id] = repr_id
                repr2mmseqs_cid[repr_id] = mmseqs_chain_id
                break
        else:
            ## This should never happen
            raise ValueError(
                "Unable to match mmseqs assigned chain ID to original chain ID"
            )
    ## parse the returned templates
    ### NOTE: Currently we just take all templates returned by the server in any order.
    ### In the case where the number of templates returned is larger than the max
    ### number of templates we want to use, we should implement a strategy to select
    ### the top N templates. Evalues/seq id is available in the returned pdb70.m8 file
    repr2templates = {rep_id: [] for rep_id in repr2useq}
    all_template_paths = []
    for tpath in template_path:
        tpath = Path(tpath)
        mmseqs_cid = tpath.name.split("_")[-1]
        template_hits = glob.glob(f"{tpath}/*.cif")
        repr2templates[mmseqs_cid2repr[mmseqs_cid]] = [
            Path(th).stem for th in template_hits
        ]
        all_template_paths.extend(template_hits)
    ## create a new folder for templates and move the cif files there
    template_dir = output_directory / "templates"
    template_dir.mkdir(exist_ok=True)
    for tpath in all_template_paths:
        tpath = Path(tpath)
        tpath.replace(template_dir / tpath.name)  ## only need unique set of templates

    ## delete the template directories
    for tpath in template_path:
        tpath = Path(tpath)
        for f in tpath.glob("*"):
            f.unlink()
        tpath.rmdir()

    ## TODO: parse templates to structure arrays

    ## update the dataset cache with the new chain information

    dataset_cache = {}
    for chain_id in chainid2repr:
        repr_id = chainid2repr[chain_id]
        templates = repr2templates[repr_id]
        dataset_cache[chain_id] = {
            "chain_id": chain_id,
            "alignment_representative_id": repr2mmseqs_cid[repr_id],
            "template_ids": templates,
            "molecule_type": "PROTEIN",
        }
    prelim_dataset_cache["structure_data"]["chains"].update(dataset_cache)
    ## write the dataset cache to file
    with open(output_directory / "dataset_cache.json", "w") as f:
        json.dump(prelim_dataset_cache, f, indent=4)

    return
