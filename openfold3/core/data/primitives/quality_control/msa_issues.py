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

"""Problems found in the MSAs of an inference query, and how they are reported.

The checks of the MSA inference pipeline do not warn or raise themselves. Each
returns a list of `MsaIssue`s, and `report_msa_issues` warns about the warnings and
raises one `MsaValidationError` listing all errors. Every kind of problem, its
severity and its message are defined once, in `MsaIssueKind`; these are the entries
of the troubleshooting table in docs/source/precomputed_msa_how_to.md.
"""

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum

from openfold3.core.config.msa_pipeline_configs import MsaKind


class Severity(Enum):
    """Whether a problem stops the query or only warns."""

    WARNING = "warning"
    ERROR = "error"


# Hints shared by several kinds of issue
_ALIGNMENT_NAMES_HINT = (
    "Alignments are selected by name: the file stem for .sto/.a3m files, the array "
    "name for .npz files. Rename them, or add the names to the MSA settings under "
    "dataset_config_kwargs.msa in the runner yaml."
)
_QUERY_ROW_HINT = (
    "Residues are given as <chain residue><position><MSA residue>. Check that each "
    "chain's MSA paths point at alignments of that chain's sequence. The MSA of a "
    "related sequence is accepted if it differs by a single residue or in at most "
    "{tolerance} of residues."
)


class MsaIssueKind(Enum):
    """The kinds of problem the MSA checks report, with their message templates.

    Templates are formatted with the fields of the `MsaIssue`: `query_name`,
    `chains` (the chain IDs, comma-separated), `msa_kind`, and its `details`.

    Attributes:
        severity (Severity):
            Whether the problem is an error or a warning.
        template (str):
            The message.
        hint (str | None):
            Advice on fixing the problem. Hints are formatted like the template and
            shown once per error report, however many errors share them.
    """

    # Found while mapping chains to representatives
    DIFFERENT_FILES_SAME_SEQUENCE = (
        Severity.WARNING,
        "Chains {rep_chain} and {chain} in query {query_name} have the same sequence "
        "but different {msa_kind} MSA files. Both will use the {msa_kind} MSAs of "
        "chain {rep_chain}: {rep_paths}. Ignoring {paths}.",
        None,
    )
    CHAIN_WITHOUT_MSAS = (
        Severity.WARNING,
        "Expected MSA file for chain {chains} of type {molecule_type} in query "
        "{query_name}, but no MSA files found. No MSA features will be computed for "
        "this chain.{copy_note}",
        None,
    )

    # Found from the alignment names, before parsing
    NO_USABLE_ALIGNMENT = (
        Severity.ERROR,
        "None of the alignments in the {msa_kind} MSA files {paths} of chain(s) "
        "{chains} in query {query_name} can be used as a {msa_kind} MSA. Found "
        "alignments: {found}. {msa_kind_title} MSAs are made from the alignments "
        "named in {required_setting} {required_names}{notes}.",
        _ALIGNMENT_NAMES_HINT,
    )
    NO_READABLE_ALIGNMENT = (
        Severity.ERROR,
        "None of the alignments in the {msa_kind} MSA files {paths} of chain(s) "
        "{chains} in query {query_name} is read, but these chains' query sequence "
        "is taken from them. Found alignments: {found}{notes}.",
        _ALIGNMENT_NAMES_HINT,
    )
    IGNORED_ALIGNMENTS = (
        Severity.WARNING,
        "Ignoring alignments in the {msa_kind} MSA files {paths} of chain(s) {chains} "
        "in query {query_name}: {ignored}.",
        _ALIGNMENT_NAMES_HINT,
    )
    UNPAIRABLE_CHAINS = (
        Severity.WARNING,
        "Protein chain(s) {chains} in query {query_name} have no main MSA alignment "
        "named in msas_to_pair {msas_to_pair}, {consequence}. Provide such an "
        "alignment with species information in its headers, or precomputed paired "
        "MSAs.",
        None,
    )

    # Found from the parsed alignments
    QUERY_ROW_LENGTH = (
        Severity.ERROR,
        "The first row of an MSA must be the sequence of the chain it is used for, "
        "but in query {query_name} for chain(s) {chains}, {alignments} from {paths}: "
        "the query row has {row_length} residues, the chain's sequence "
        "{sequence_length}.",
        _QUERY_ROW_HINT,
    )
    QUERY_ROW_MISMATCH = (
        Severity.ERROR,
        "The first row of an MSA must be the sequence of the chain it is used for, "
        "but in query {query_name} for chain(s) {chains}, {alignments} from {paths}: "
        "{differences}.",
        _QUERY_ROW_HINT,
    )
    QUERY_ROW_TOLERATED = (
        Severity.WARNING,
        "The query rows of MSAs for chain(s) {chains} in query {query_name} differ "
        "from the chain's sequence, given as <chain residue><position><MSA "
        "residue>: {differences}. The MSAs are used as they are. This is expected "
        "when reusing the MSA of a related sequence, such as a point mutant's; "
        "otherwise check the chain's MSA paths.",
        None,
    )

    def __init__(self, severity: Severity, template: str, hint: str | None):
        self.severity = severity
        self.template = template
        self.hint = hint


@dataclass(frozen=True)
class MsaIssue:
    """One problem found in the MSAs of a query.

    Attributes:
        kind (MsaIssueKind):
            What the problem is, which sets its severity and message.
        query_name (str):
            The query the problem was found in.
        chain_ids (tuple[str, ...]):
            The chains affected.
        msa_kind (MsaKind | None):
            The kind of MSA affected, if the problem concerns one.
        details (Mapping[str, object]):
            The other fields of the message template.
    """

    kind: MsaIssueKind
    query_name: str
    chain_ids: tuple[str, ...]
    msa_kind: MsaKind | None = None
    details: Mapping[str, object] = field(default_factory=dict)

    @property
    def severity(self) -> Severity:
        return self.kind.severity

    def _fields(self) -> dict[str, object]:
        msa_kind = str(self.msa_kind) if self.msa_kind is not None else ""
        return {
            "query_name": self.query_name,
            "chains": ", ".join(self.chain_ids),
            "msa_kind": msa_kind,
            "msa_kind_title": msa_kind.capitalize(),
            **self.details,
        }

    @property
    def message(self) -> str:
        """The message, without the hint."""
        return self.kind.template.format(**self._fields())

    @property
    def hint(self) -> str | None:
        """The hint on fixing the problem, if any."""
        if self.kind.hint is None:
            return None
        return self.kind.hint.format(**self._fields())


class MsaValidationError(ValueError):
    """Raised for the MSA errors of a query, all of which it lists.

    Attributes:
        issues (tuple[MsaIssue, ...]):
            The errors.
    """

    def __init__(self, query_name: str, issues: Sequence[MsaIssue]):
        self.issues = tuple(issues)
        hints = dict.fromkeys(i.hint for i in self.issues if i.hint is not None)
        lines = [
            f"Invalid MSAs in query {query_name}:",
            *(f" - {issue.message}" for issue in self.issues),
            *hints,
        ]
        super().__init__("\n".join(lines))


def report_msa_issues(query_name: str, issues: Sequence[MsaIssue]) -> None:
    """Warns about each warning, then raises for all errors at once.

    Args:
        query_name (str):
            The query the issues were found in.
        issues (Sequence[MsaIssue]):
            The issues found by the MSA checks.

    Raises:
        MsaValidationError:
            If any issue is an error.
    """
    for issue in issues:
        if issue.severity is Severity.WARNING:
            message = issue.message
            if issue.hint is not None:
                message = f"{message} {issue.hint}"
            warnings.warn(message, stacklevel=3)
    errors = [issue for issue in issues if issue.severity is Severity.ERROR]
    if errors:
        raise MsaValidationError(query_name, errors)
