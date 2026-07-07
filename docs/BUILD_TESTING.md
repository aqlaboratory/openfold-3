# Building and Testing a Local sdist

This document describes how to build an `openfold3` sdist/wheel locally and
verify it installs and runs correctly. This is a pre-release sanity check —
it does **not** cover tagging, versioning, or publishing. Actual releases go
through the release pipeline once local testing here passes.

## 1. Build in a clean environment

```bash
cd ~/workspace/openfold-3
# make sure you're on the branch/commit you want to test

uv build --python=3.14
```

uv build builds in an isolated, ephemeral virtual environment by
default — it installs whatever build backend/deps pyproject.toml
declares into a throwaway env, builds the distributions, and discards the
env. There's nothing to create or activate beforehand, and it can't pick up
a stale openfold3 install from your shell, since it never touches your
active environment.

This produces a `dist/openfold3-<version>.tar.gz` (sdist) and a
`dist/openfold3-<version>-py3-none-any.whl` (wheel) in the repo's `dist/`
directory. Since the working tree isn't tagged for a real release,
`setuptools_scm` will typically generate a local dev version like
`0.4.2.dev146+gf4063eb56.d20260701` — that's expected and fine for this kind
of local testing.

Optionally check the sdist metadata is well-formed before installing it:

```bash
uvx twine check dist/*
```

## 2. Install the sdist into a fresh venv with uv

Testing from a clean venv (rather than your dev environment) is what
actually catches packaging bugs — missing files, incorrect package data,
wrong dependency pins — that `python -m build` alone won't surface.

```bash
mkdir test-sdist && cd test-sdist
uv venv
source .venv/bin/activate

uv pip install ../dist/openfold3-<version>.tar.gz[dev]
```

(Adjust the path/version to match whatever landed in `dist/`.)

## 3. Run the test suite against the installed package

```bash
pytest --pyargs openfold3
```

Running with `--pyargs` is important here — it runs tests against the
**installed** package in `.venv/`, not against the source tree, which is the
point of this exercise (confirming the packaged artifact actually works).

