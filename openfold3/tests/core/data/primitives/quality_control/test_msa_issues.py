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

"""Tests for rendering and reporting MSA issues."""

import warnings

import pytest

from openfold3.core.config.msa_pipeline_configs import MsaKind
from openfold3.core.data.primitives.quality_control.msa_issues import (
    MsaIssue,
    MsaIssueKind,
    MsaValidationError,
    Severity,
    report_msa_issues,
)


def _no_usable_alignment(chain_id: str) -> MsaIssue:
    return MsaIssue(
        kind=MsaIssueKind.NO_USABLE_ALIGNMENT,
        query_name="q",
        chain_ids=(chain_id,),
        msa_kind=MsaKind.MAIN,
        details={
            "paths": [f"msas/{chain_id}"],
            "found": ["my_db"],
            "required_setting": "aln_order",
            "required_names": ["uniref90_hits"],
            "notes": "",
        },
    )


IGNORED = MsaIssue(
    kind=MsaIssueKind.IGNORED_ALIGNMENTS,
    query_name="q",
    chain_ids=("A", "B"),
    msa_kind=MsaKind.MAIN,
    details={"paths": ["msas/A"], "ignored": "my_db (not in aln_order)"},
)


def test_message_fills_the_template_from_the_issue_fields():
    assert IGNORED.message == (
        "Ignoring alignments in the main MSA files ['msas/A'] of chain(s) A, B in "
        "query q: my_db (not in aln_order)."
    )
    assert IGNORED.severity is Severity.WARNING


def test_message_capitalizes_the_msa_kind_where_it_starts_a_sentence():
    assert "Main MSAs are made from the alignments named in aln_order" in (
        _no_usable_alignment("A").message
    )


def test_warnings_are_warned_with_their_hint():
    with pytest.warns(UserWarning) as record:
        report_msa_issues("q", [IGNORED])

    (warning,) = record
    assert str(warning.message) == f"{IGNORED.message} {IGNORED.hint}"


def test_errors_are_raised_together_with_shared_hints_once():
    errors = [_no_usable_alignment("A"), _no_usable_alignment("B")]

    with pytest.raises(MsaValidationError) as excinfo:
        report_msa_issues("q", errors)

    assert excinfo.value.issues == tuple(errors)
    assert str(excinfo.value) == "\n".join(
        [
            "Invalid MSAs in query q:",
            f" - {errors[0].message}",
            f" - {errors[1].message}",
            str(errors[0].hint),
        ]
    )


def test_warnings_are_given_before_errors_are_raised():
    with (
        pytest.warns(UserWarning, match="Ignoring alignments"),
        pytest.raises(MsaValidationError),
    ):
        report_msa_issues("q", [_no_usable_alignment("A"), IGNORED])


def test_no_issues_report_nothing():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        report_msa_issues("q", [])


def test_validation_error_is_a_value_error():
    """Callers catching ValueError, as before these issues, still catch it."""
    assert issubclass(MsaValidationError, ValueError)
