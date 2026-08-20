"""CogniGate does not write to the ledger.

A worker does not mint obligations. AsyncGate holds the lease and proposes
acceptance and completion on the worker's behalf, which is what makes the
obligation one thing with one identity.

CogniGate used to emit its own `accepted` and `complete` receipts straight to
ReceiptGate, with an obligation_id derived from the *lease*:

    obligation_id = derive_ulid("cognigate.lease", lease.lease_id)

while AsyncGate derived one from the *task* for the same work. One task
therefore carried two obligations: the principal's, which nothing ever closed,
and a lease-scoped one that opened and closed beside it. A lease is operational
and is re-granted; the obligation persists across leases, and the constitution
already says so -- `LEASE_EXPIRED` declares `changes_custody: false`.

`worker.contract.md` had required workers to emit `accepted` while also
prohibiting them from minting obligations, which are the same act described
twice. The contradiction was resolved in favour of the prohibition; this is the
code side of that.

CogniGate still reports to AsyncGate. That is the correct path: it tells the
component holding the lease what happened, and that component notarises.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "cognigate"


def _sources() -> list[Path]:
    return sorted(SRC.rglob("*.py"))


def test_no_module_talks_to_receiptgate():
    offenders = {
        path.name: line.strip()
        for path in _sources()
        for line in path.read_text(encoding="utf-8").splitlines()
        if re.search(r"receiptgate\.submit_receipt|ReceiptGateClient", line)
    }
    assert not offenders, (
        f"{offenders} write to the ledger directly. A worker does not mint "
        "obligations; it reports to AsyncGate, which notarises on its behalf."
    )


def test_no_module_mints_an_obligation_id():
    """Deriving one locally is minting one, whatever it is derived from."""
    offenders = {
        path.name: line.strip()
        for path in _sources()
        for line in path.read_text(encoding="utf-8").splitlines()
        if "obligation_id" in line and not line.strip().startswith("#")
    }
    assert not offenders, (
        f"{offenders} construct an obligation_id. The obligation is opened by "
        "the acceptance AsyncGate proposes, and every later receipt repeats it."
    )


def test_the_receipt_builder_is_gone():
    assert not (SRC / "legivellum_receipts.py").exists()
    assert not (SRC / "receiptgate_client.py").exists()


def test_no_receiptgate_settings_remain():
    """A setting for an endpoint the component must not call is a footgun.

    Leaving it would let a deployment point CogniGate at the ledger and expect
    that to mean something.
    """
    config = (SRC / "config.py").read_text(encoding="utf-8")
    declared = [
        line.strip()
        for line in config.splitlines()
        if re.match(r"\s*receiptgate_\w+\s*:", line)
    ]
    assert not declared, declared


def test_the_manifest_does_not_bind_a_receiptgate_endpoint():
    """Bootstrap must not hand the worker a capability it must not have."""
    client = (SRC / "metagate_client.py").read_text(encoding="utf-8")
    specs = re.search(r"_BINDING_SPECS[^=]*=\s*\((.*?)\)\s*\n", client, re.S)
    assert specs, "binding specs not found"
    assert "receiptgate" not in specs.group(1)


def test_cognigate_still_reports_to_asyncgate():
    """The other half. Silence would be worse than the duplicate obligation.

    Removing the ledger path must not remove the worker's obligation to say what
    happened -- it just says it to the component that holds the lease.
    """
    leasing = (SRC / "leasing.py").read_text(encoding="utf-8")
    assert "send_receipt" in leasing, (
        "CogniGate no longer reports outcomes to AsyncGate at all"
    )
