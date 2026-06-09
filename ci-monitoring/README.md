# CI Monitoring

Persistent records and operating instructions for the daily CI failure scan on
`aqlaboratory/openfold-3`'s `integration-test.yml` workflow.

## Purpose

Distinguish flaky AWS GPU outages from real OpenFold-3 regressions:

- **AWS outage** — record silently in [`aws-outage-failures.md`](./aws-outage-failures.md).
- **Code regression** — send a notification to the maintainer (no silent record).
- **All passing** — no output, no notification.

## How it runs

The scan is driven by a Claude Code on the web scheduled trigger. The trigger
fires once per day and re-invokes a prompt equivalent to:

> Scan the integration-test workflow on aqlaboratory/openfold-3 for failures
> since the last scan. For each failure, fetch the failed job logs, classify
> against the AWS-outage signatures in `ci-monitoring/aws-outage-failures.md`,
> append AWS outages to that file (no notification), and notify only on
> failures that look like code regressions. End the turn silently if every
> in-window run passed.

To create or edit the trigger:

1. Open <https://code.claude.com/> → this project → Triggers.
2. Add a "Schedule" trigger with cadence `daily`.
3. Paste the prompt above as the trigger prompt.
4. Set the development branch to whatever branch you want any record-file
   updates committed to (this directory was first populated on
   `claude/ecstatic-faraday-kdx091`).

Reference: <https://code.claude.com/docs/en/claude-code-on-the-web>

## Weekly report

The user asks roughly once a week; the assistant reads
[`aws-outage-failures.md`](./aws-outage-failures.md) plus the latest 7 days of
runs from the `integration-test.yml` workflow and reports:

- Total runs vs. failed runs in the window.
- Share of failures attributed to AWS outage.
- Any non-outage failures (these should already have triggered a notification
  when they happened, but the weekly report confirms nothing slipped).
