"""Helpers for building LegiVellum receipt payloads."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from legivellum.ulid import derive_ulid

from .models import Lease

# Hard dependency, imported unguarded. The parent-directory walk this replaces
# found LegiVellum/shared in a checkout and nothing in a container, so
# CanonicalReceipt was None in every deployment and this module posted
# unvalidated dictionaries whose rejections were logged and dropped.
from legivellum.models import Receipt as CanonicalReceipt


def _normalize_artifacts(
    artifacts: list[dict[str, Any]] | list[Any] | None,
) -> list[dict[str, Any]]:
    if not artifacts:
        return []
    normalized: list[dict[str, Any]] = []
    for artifact in artifacts:
        if isinstance(artifact, dict):
            normalized.append(artifact)
        elif hasattr(artifact, "model_dump"):
            normalized.append(artifact.model_dump())
    return normalized


def _extract_artifact_fields(artifacts: list[dict[str, Any]] | list[Any] | None) -> dict[str, Any]:
    normalized = _normalize_artifacts(artifacts)
    artifact = normalized[0] if normalized else None
    if not isinstance(artifact, dict):
        return {
            "artifact_location": "NA",
            "artifact_pointer": "NA",
            "artifact_checksum": "NA",
            "artifact_size_bytes": 0,
            "artifact_mime": "NA",
        }

    pointer = (
        artifact.get("uri")
        or artifact.get("url")
        or artifact.get("pointer")
        or artifact.get("output_path")
        or "NA"
    )
    location = artifact.get("sink_id") or artifact.get("type") or "NA"
    metadata = artifact.get("metadata") or {}

    return {
        "artifact_location": location,
        "artifact_pointer": pointer,
        "artifact_checksum": metadata.get("checksum", "NA"),
        "artifact_size_bytes": metadata.get("size_bytes", 0),
        "artifact_mime": metadata.get("mime", "NA"),
    }


def _coerce_principal(lease: Lease) -> str:
    principal_ai = lease.principal_ai or lease.payload.get("principal_ai") if isinstance(lease.payload, dict) else None
    if principal_ai:
        return principal_ai
    return "unknown"


def build_receipt(
    *,
    lease: Lease,
    phase: str,
    status: str,
    worker_id: str,
    summary: str | None = None,
    artifact_pointers: list[dict[str, Any]] | None = None,
    error_metadata: dict[str, Any] | None = None,
    receipt_id: str | None = None,
    started_at: datetime | None = None,
    completed_at: datetime | None = None,
) -> dict[str, Any]:
    """Build a LegiVellum receipt payload."""
    now = datetime.now(timezone.utc)
    principal_ai = _coerce_principal(lease)
    owner_principal = principal_ai if principal_ai != "unknown" else worker_id

    task_type = lease.task_type or (lease.payload.get("task_type") if isinstance(lease.payload, dict) else None)
    task_type = task_type or "cognitive"

    task_summary = lease.payload.get("task_summary") if isinstance(lease.payload, dict) else None
    task_summary = task_summary or summary or task_type

    inputs: dict[str, Any] = {}
    task_body = "TBD"
    if lease.payload_pointer:
        inputs["payload_pointer"] = lease.payload_pointer
    if lease.payload:
        inputs["payload"] = lease.payload
        task_body = json.dumps(lease.payload)
    elif lease.payload_pointer:
        task_body = lease.payload_pointer

    outcome_kind = "NA"
    outcome_text = "NA"
    if phase == "complete":
        has_artifacts = bool(artifact_pointers)
        has_summary = bool(summary)
        has_error = bool(error_metadata)

        if has_artifacts and (has_summary or has_error):
            outcome_kind = "mixed"
        elif has_artifacts:
            outcome_kind = "artifact_pointer"
        elif has_summary or has_error:
            outcome_kind = "response_text"
        else:
            outcome_kind = "none"

        if summary:
            outcome_text = summary
        elif error_metadata:
            outcome_text = error_metadata.get("message", "NA")

    normalized_artifacts = _normalize_artifacts(artifact_pointers)
    artifact_fields = _extract_artifact_fields(normalized_artifacts)

    caused_by_receipt_id = lease.caused_by_receipt_id or "NA"
    recipient_ai = worker_id if phase == "accepted" else owner_principal

    body_payload: dict[str, Any] = {
        "phase": phase,
        "status": status,
        "worker_id": worker_id,
        "lease_id": lease.lease_id,
        "summary": summary,
        "artifacts": normalized_artifacts or None,
        "error": error_metadata,
        "started_at": started_at.isoformat() if started_at else None,
        "completed_at": completed_at.isoformat() if completed_at else None,
    }

    payload = {
        "schema_version": "1.0",
        "tenant_id": lease.tenant_id or "default",
        "receipt_id": receipt_id or str(uuid4()),
        "task_id": lease.task_id,
        # One lease is one obligation. accepted and complete are built from
        # different call sites with no shared state, so the id is derived from
        # the lease rather than minted twice -- otherwise the closing receipt
        # would name an obligation that was never opened.
        "obligation_id": derive_ulid("cognigate.lease", lease.lease_id),
        "parent_task_id": "NA",
        "caused_by_receipt_id": caused_by_receipt_id,
        "dedupe_key": lease.lease_id,
        "attempt": 0,
        "from_principal": owner_principal,
        "for_principal": owner_principal,
        "source_system": "cognigate",
        "recipient_ai": recipient_ai,
        "trust_domain": "default",
        "phase": phase,
        "status": status,
        "realtime": False,
        "task_type": task_type,
        "task_summary": task_summary,
        "task_body": task_body,
        "inputs": inputs,
        "expected_outcome_kind": lease.expected_outcome_kind or "NA",
        "expected_artifact_mime": lease.expected_artifact_mime or "NA",
        "outcome_kind": outcome_kind,
        "outcome_text": outcome_text,
        **artifact_fields,
        "escalation_class": "NA",
        "escalation_reason": "NA",
        "escalation_to": "NA",
        "retry_requested": False,
        "body": body_payload,
        "artifact_refs": normalized_artifacts,
        "created_at": now.isoformat(),
        "stored_at": now.isoformat(),
        "started_at": started_at.isoformat() if started_at else None,
        "completed_at": completed_at.isoformat() if completed_at else None,
        "read_at": None,
        "archived_at": None,
        "metadata": {
            "lease_id": lease.lease_id,
            "worker_id": worker_id,
        },
    }

    return CanonicalReceipt.model_validate(payload).model_dump(mode="json")
