# Copyright 2026 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Loading and preflight operations for inference query sets."""

import json
import logging
from pathlib import Path

from biotite.structure.io import pdbx
from pydantic import FilePath, ValidationError

from openfold3.core.data.primitives.structure.query import (
    CovalentQueryError,
    infer_ccd_leaving_atoms,
    normalize_manual_leaving_atoms,
    structure_with_ref_mols_from_query,
)
from openfold3.projects.of3_all_atom.config.inference_query_format import (
    InferenceQuerySet,
    Query,
)

logger = logging.getLogger(__name__)


def load_inference_query_set_with_query_errors(
    json_path: FilePath,
) -> tuple[InferenceQuerySet, dict[str, ValidationError]]:
    """Load valid queries while retaining query-scoped schema errors.

    Top-level syntax, type, and extra-field errors still fail the entire input. Each
    entry under ``queries`` is otherwise validated independently so one malformed
    query does not prevent valid siblings from running.
    """
    with open(json_path) as file:
        data = json.load(file)

    # Delegate malformed/missing ``queries`` and all top-level validation to the
    # ordinary model. Replacing only the mapping contents keeps ``extra=forbid`` and
    # seed validation intact without validating every query as one unit.
    if not isinstance(data, dict) or not isinstance(data.get("queries"), dict):
        return InferenceQuerySet.model_validate(data), {}
    raw_queries = data["queries"]
    top_level_data = dict(data)
    top_level_data["queries"] = {}
    top_level_query_set = InferenceQuerySet.model_validate(top_level_data)

    valid_queries: dict[str, Query] = {}
    query_errors: dict[str, ValidationError] = {}
    for query_name, raw_query in raw_queries.items():
        try:
            valid_queries[query_name] = Query.model_validate(raw_query)
        except ValidationError as error:
            query_errors[query_name] = error

    return (
        InferenceQuerySet(seeds=top_level_query_set.seeds, queries=valid_queries),
        query_errors,
    )


def preflight_covalent_query_set(
    query_set: InferenceQuerySet,
    *,
    structure_format: str,
    infer_leaving_atoms: bool = False,
    ccd_file_path: Path | None = None,
) -> InferenceQuerySet:
    """Validate and normalize query-level covalent-bond settings.

    Returns the original query set when preparation makes no changes, otherwise a new
    set containing the effective queries. Invalid individual queries are logged and
    omitted; an entirely invalid query set raises ``ValueError``.
    """
    bonded_query_ids = [
        query_id
        for query_id, query in query_set.queries.items()
        if query.covalent_bonds
    ]
    if bonded_query_ids and structure_format == "pdb":
        raise ValueError(
            "PDB output does not preserve query-defined covalent connectivity. "
            "Select CIF or CIF.GZ output for queries containing covalent_bonds. "
            f"Affected queries: {bonded_query_ids}"
        )

    has_query_edits = any(
        query.covalent_bonds or query.leaving_atoms
        for query in query_set.queries.values()
    )
    if not infer_leaving_atoms and not has_query_edits:
        return query_set

    # A custom text CCD can be hundreds of megabytes. Parse it once for the query set
    # rather than once per query. Query molecule construction intentionally continues
    # to use Biotite's separate preprocessed BinaryCIF source.
    inference_ccd = (
        pdbx.CIFFile.read(ccd_file_path)
        if infer_leaving_atoms and ccd_file_path is not None
        else None
    )

    effective_queries = {}
    failed_queries = []
    for query_id, query in query_set.queries.items():
        try:
            effective_query = normalize_manual_leaving_atoms(query)
            effective_query = (
                infer_ccd_leaving_atoms(effective_query, ccd=inference_ccd)
                if infer_leaving_atoms
                else effective_query
            )
            if effective_query.covalent_bonds or effective_query.leaving_atoms:
                # Preflight against the production structure builder so invalid
                # selectors do not enter the data module or effective query log.
                # Production featurization rebuilds this structure later.
                structure_with_ref_mols_from_query(effective_query)
            effective_queries[query_id] = effective_query
        except CovalentQueryError as error:
            failed_queries.append(query_id)
            logger.error(
                "Skipping query %s because covalent query preflight failed: %s",
                query_id,
                error,
            )

    if not effective_queries:
        raise ValueError(
            "No valid queries remain after covalent query preflight. "
            f"Failed queries: {failed_queries}"
        )

    if not failed_queries and all(
        effective_queries[query_id] == query
        for query_id, query in query_set.queries.items()
    ):
        return query_set

    return type(query_set)(seeds=query_set.seeds, queries=effective_queries)


__all__ = [
    "load_inference_query_set_with_query_errors",
    "preflight_covalent_query_set",
]
