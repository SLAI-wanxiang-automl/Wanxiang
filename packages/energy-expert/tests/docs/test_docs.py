from pathlib import Path

import pytest
from mktestdocs import check_md_file

# Docstring examples in these modules construct LLM agents and would require a
# live LLM provider to execute.
REQUIRES_LLM = {"agent.py"}


@pytest.mark.docs
@pytest.mark.parametrize(
    "fpath",
    Path("docs").rglob("*.md"),
    ids=str,
)
def test_docs(fpath):
    check_md_file(fpath=fpath, memory=True)


@pytest.mark.docs
def test_readme():
    check_md_file("README.md", memory=True)


@pytest.mark.docs
@pytest.mark.parametrize(
    "fpath",
    [
        p
        for p in Path("eneygyexpert").glob("**/*.py")
        if p.name not in REQUIRES_LLM
    ],
    ids=str,
)
def test_py_examples(fpath):
    check_md_file(fpath=fpath, memory=True)
