"""Runtime drift alarms for attested PromptABI contracts."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from ._version import __version__
from .bundles import _stable_json_hash
from .lockfiles import Lockfile, LockfileArtifact, load_lockfile
from .runtime_attestation import (
    RUNTIME_CONTRACT_FAMILIES,
    RuntimeAttestationReport,
    RuntimeContract,
)


RUNTIME_ALARM_REPORT_VERSION = "promptabi.runtime-alarms.v1"


class RuntimeAlarmError(ValueError):
    """Raised when runtime alarm inputs cannot be loaded or compared."""


class RuntimeAlarmSeverity(StrEnum):
    """Severity assigned to a runtime drift alarm."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class RuntimeAlarmSource(StrEnum):
    """Evidence source that produced a runtime alarm."""

    ATTESTATION = "attestation"
    LOCKFILE = "lockfile"
    POLICY_PACK = "policy-pack"
    CORPUS_BASELINE = "corpus-baseline"
    KNOWN_BAD = "known-bad"


@dataclass(frozen=True, slots=True)
class RuntimeAlarm:
    """One concrete runtime-attestation drift alarm."""

    rule_id: str
    severity: RuntimeAlarmSeverity
    source: RuntimeAlarmSource
    message: str
    contract: str | None = None
    expected: object | None = None
    actual: object | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "rule_id": self.rule_id,
            "severity": self.severity.value,
            "source": self.source.value,
            "message": self.message,
        }
        for key in ("contract", "expected", "actual", "reason"):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        return payload

    @property
    def sort_key(self) -> tuple[str, str, str]:
        return (self.source.value, self.rule_id, self.contract or "")


@dataclass(frozen=True, slots=True)
class RuntimeAlarmReport:
    """Deterministic report comparing runtime attestation against current evidence."""

    service: str
    environment: str
    attestation_sha256: str
    lockfile: str | None
    policy_pack: str | None
    corpus_baseline: str | None
    known_bad: str | None
    alarms: tuple[RuntimeAlarm, ...]

    @property
    def ok(self) -> bool:
        return not any(alarm.severity is RuntimeAlarmSeverity.ERROR for alarm in self.alarms)

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "report_version": RUNTIME_ALARM_REPORT_VERSION,
            "promptabi_version": __version__,
            "service": self.service,
            "environment": self.environment,
            "ok": self.ok,
            "attestation_sha256": self.attestation_sha256,
            "alarm_counts": _alarm_counts(self.alarms),
            "alarms": [alarm.to_dict() for alarm in self.alarms],
        }
        for key in ("lockfile", "policy_pack", "corpus_baseline", "known_bad"):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        payload["report_sha256"] = _stable_json_hash(payload)
        return payload


def build_runtime_alarm_report(
    attestation: str | Path | Mapping[str, Any] | RuntimeAttestationReport,
    *,
    lockfile: str | Path | Lockfile | None = None,
    policy_pack: str | Path | Mapping[str, Any] | None = None,
    corpus_baseline: str | Path | Mapping[str, Any] | None = None,
    known_bad: str | Path | Mapping[str, Any] | None = None,
) -> RuntimeAlarmReport:
    """Compare one runtime attestation with lockfile, policy, corpus, and bad-version evidence."""

    attestation_payload, attestation_path = _attestation_payload(attestation)
    contracts = _contracts_from_payload(attestation_payload)
    alarms: list[RuntimeAlarm] = []
    if attestation_payload.get("ok") is not True:
        alarms.append(
            RuntimeAlarm(
                "runtime-attestation-not-ok",
                RuntimeAlarmSeverity.ERROR,
                RuntimeAlarmSource.ATTESTATION,
                "runtime attestation reports a non-passing verification gate",
                expected=True,
                actual=attestation_payload.get("ok"),
            )
        )
    missing_families = tuple(
        family for family in RUNTIME_CONTRACT_FAMILIES if _family_count(attestation_payload, family) == 0
    )
    if missing_families:
        alarms.append(
            RuntimeAlarm(
                "runtime-attestation-family-missing",
                RuntimeAlarmSeverity.ERROR,
                RuntimeAlarmSource.ATTESTATION,
                "runtime attestation is missing required contract families",
                expected=list(RUNTIME_CONTRACT_FAMILIES),
                actual={family: _family_count(attestation_payload, family) for family in RUNTIME_CONTRACT_FAMILIES},
                reason=", ".join(missing_families),
            )
        )

    lock_path: str | None = None
    if lockfile is not None:
        loaded_lockfile, lock_path = _load_lockfile_input(lockfile)
        alarms.extend(_lockfile_alarms(contracts, loaded_lockfile))

    policy_path: str | None = None
    if policy_pack is not None:
        policy, policy_path = _load_json_input(policy_pack, "policy pack")
        alarms.extend(_policy_alarms(attestation_payload, policy))

    corpus_path: str | None = None
    if corpus_baseline is not None:
        corpus, corpus_path = _load_json_input(corpus_baseline, "corpus baseline")
        alarms.extend(_baseline_alarms(contracts, corpus, source=RuntimeAlarmSource.CORPUS_BASELINE))

    known_bad_path: str | None = None
    if known_bad is not None:
        bad, known_bad_path = _load_json_input(known_bad, "known-bad manifest")
        alarms.extend(_known_bad_alarms(contracts, bad))

    return RuntimeAlarmReport(
        service=_required_str(attestation_payload, "service"),
        environment=_required_str(attestation_payload, "environment"),
        attestation_sha256=_stable_json_hash(attestation_payload),
        lockfile=lock_path,
        policy_pack=policy_path,
        corpus_baseline=corpus_path,
        known_bad=known_bad_path,
        alarms=tuple(sorted(alarms, key=lambda alarm: alarm.sort_key)),
    )


def render_runtime_alarm_json(report: RuntimeAlarmReport) -> str:
    """Render runtime alarms as deterministic JSON."""

    return json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"


def render_runtime_alarm_text(report: RuntimeAlarmReport) -> str:
    """Render a compact runtime alarm summary."""

    lines = [
        "PromptABI runtime alarms",
        f"status: {'PASS' if report.ok else 'FAIL'}",
        f"service: {report.service}",
        f"environment: {report.environment}",
        f"attestation_sha256: {report.attestation_sha256}",
        "alarms: "
        + ", ".join(f"{severity}={count}" for severity, count in sorted(_alarm_counts(report.alarms).items())),
    ]
    for source_name, path in (
        ("lockfile", report.lockfile),
        ("policy_pack", report.policy_pack),
        ("corpus_baseline", report.corpus_baseline),
        ("known_bad", report.known_bad),
    ):
        if path is not None:
            lines.append(f"{source_name}: {path}")
    for alarm in report.alarms:
        subject = f" ({alarm.contract})" if alarm.contract else ""
        lines.append(f"- {alarm.severity.value} {alarm.rule_id}{subject}: {alarm.message}")
        if alarm.reason:
            lines.append(f"  reason: {alarm.reason}")
        if alarm.expected is not None or alarm.actual is not None:
            lines.append(f"  expected: {alarm.expected!r}; actual: {alarm.actual!r}")
    return "\n".join(lines) + "\n"


def _attestation_payload(
    attestation: str | Path | Mapping[str, Any] | RuntimeAttestationReport,
) -> tuple[dict[str, Any], str | None]:
    if isinstance(attestation, RuntimeAttestationReport):
        return dict(attestation.to_dict()), None
    if isinstance(attestation, Mapping):
        return dict(attestation), None
    payload, path = _load_json_input(attestation, "runtime attestation")
    return payload, path


def _contracts_from_payload(payload: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    raw = payload.get("contracts")
    if not isinstance(raw, list):
        raise RuntimeAlarmError("runtime attestation field 'contracts' must be a list")
    contracts: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise RuntimeAlarmError(f"runtime attestation contract {index} must be an object")
        artifact = item.get("artifact")
        if not isinstance(artifact, dict):
            raise RuntimeAlarmError(f"runtime attestation contract {index} is missing artifact summary")
        name = item.get("name")
        kind = item.get("kind")
        if not isinstance(name, str) or not isinstance(kind, str):
            raise RuntimeAlarmError(f"runtime attestation contract {index} must include name and kind")
        contracts.append(dict(item))
    return tuple(sorted(contracts, key=lambda item: (_required_str(item, "kind"), _required_str(item, "name"))))


def _lockfile_alarms(contracts: tuple[dict[str, Any], ...], lockfile: Lockfile) -> tuple[RuntimeAlarm, ...]:
    alarms: list[RuntimeAlarm] = []
    locked = {(artifact.kind, artifact.name): artifact for artifact in lockfile.artifacts}
    seen: set[tuple[str, str]] = set()
    for contract in contracts:
        key = (_required_str(contract, "kind"), _required_str(contract, "name"))
        seen.add(key)
        artifact = _artifact_summary(contract)
        locked_artifact = locked.get(key)
        label = f"{key[1]}:{key[0]}"
        if locked_artifact is None:
            alarms.append(
                RuntimeAlarm(
                    "runtime-lockfile-contract-missing",
                    RuntimeAlarmSeverity.ERROR,
                    RuntimeAlarmSource.LOCKFILE,
                    "runtime contract is not present in the latest lockfile",
                    contract=label,
                )
            )
            continue
        alarms.extend(_artifact_field_alarms(label, artifact, locked_artifact))
    for kind, name in sorted(set(locked) - seen):
        alarms.append(
            RuntimeAlarm(
                "runtime-lockfile-contract-not-attested",
                RuntimeAlarmSeverity.WARNING,
                RuntimeAlarmSource.LOCKFILE,
                "latest lockfile contains a contract not reported by runtime attestation",
                contract=f"{name}:{kind}",
            )
        )
    return tuple(alarms)


def _artifact_field_alarms(
    label: str,
    artifact: Mapping[str, Any],
    locked: LockfileArtifact,
) -> tuple[RuntimeAlarm, ...]:
    alarms: list[RuntimeAlarm] = []
    for field_name in ("sha256", "version", "revision"):
        expected = getattr(locked, field_name)
        actual = artifact.get(field_name)
        if expected is not None and actual is not None and actual != expected:
            alarms.append(
                RuntimeAlarm(
                    "runtime-lockfile-artifact-drift",
                    RuntimeAlarmSeverity.ERROR,
                    RuntimeAlarmSource.LOCKFILE,
                    f"runtime artifact {field_name} differs from latest lockfile",
                    contract=label,
                    expected=expected,
                    actual=actual,
                )
            )
    return tuple(alarms)


def _policy_alarms(attestation: Mapping[str, Any], policy: Mapping[str, Any]) -> tuple[RuntimeAlarm, ...]:
    runtime_policy = policy.get("runtime_alarms", policy)
    if not isinstance(runtime_policy, Mapping):
        raise RuntimeAlarmError("policy pack field 'runtime_alarms' must be an object")
    alarms: list[RuntimeAlarm] = []
    required_families = _string_sequence(runtime_policy.get("required_contract_families", ()), "required_contract_families")
    missing = tuple(family for family in required_families if _family_count(attestation, family) == 0)
    if missing:
        alarms.append(
            RuntimeAlarm(
                "runtime-policy-required-family-missing",
                RuntimeAlarmSeverity.ERROR,
                RuntimeAlarmSource.POLICY_PACK,
                "runtime attestation does not satisfy policy-required contract families",
                expected=list(required_families),
                actual=dict(_mapping(attestation.get("contract_families", {}), "contract_families")),
                reason=", ".join(missing),
            )
        )
    allowed_environments = _string_sequence(runtime_policy.get("allowed_environments", ()), "allowed_environments")
    if allowed_environments and attestation.get("environment") not in set(allowed_environments):
        alarms.append(
            RuntimeAlarm(
                "runtime-policy-environment-denied",
                RuntimeAlarmSeverity.ERROR,
                RuntimeAlarmSource.POLICY_PACK,
                "runtime environment is not allowed by the policy pack",
                expected=list(allowed_environments),
                actual=attestation.get("environment"),
            )
        )
    required_key_ids = _string_sequence(runtime_policy.get("required_signing_key_ids", ()), "required_signing_key_ids")
    bundle = _mapping(attestation.get("bundle", {}), "bundle")
    signing_key_id = bundle.get("signing_key_id")
    if required_key_ids and signing_key_id not in set(required_key_ids):
        alarms.append(
            RuntimeAlarm(
                "runtime-policy-signing-key-denied",
                RuntimeAlarmSeverity.ERROR,
                RuntimeAlarmSource.POLICY_PACK,
                "runtime attestation was signed with an unapproved key id",
                expected=list(required_key_ids),
                actual=signing_key_id,
            )
        )
    return tuple(alarms)


def _baseline_alarms(
    contracts: tuple[dict[str, Any], ...],
    baseline: Mapping[str, Any],
    *,
    source: RuntimeAlarmSource,
) -> tuple[RuntimeAlarm, ...]:
    baseline_artifacts = _artifact_entries(baseline)
    if not baseline_artifacts:
        return ()
    alarms: list[RuntimeAlarm] = []
    contracts_by_key = {
        (_required_str(contract, "kind"), _required_str(contract, "name")): _artifact_summary(contract)
        for contract in contracts
    }
    for expected in baseline_artifacts:
        key = (_required_str(expected, "kind"), _required_str(expected, "name"))
        actual = contracts_by_key.get(key)
        label = f"{key[1]}:{key[0]}"
        if actual is None:
            alarms.append(
                RuntimeAlarm(
                    "runtime-corpus-baseline-contract-missing",
                    RuntimeAlarmSeverity.WARNING,
                    source,
                    "corpus baseline references a contract not reported at runtime",
                    contract=label,
                )
            )
            continue
        for field_name in ("sha256", "version", "revision"):
            if field_name in expected and actual.get(field_name) != expected[field_name]:
                alarms.append(
                    RuntimeAlarm(
                        "runtime-corpus-baseline-drift",
                        RuntimeAlarmSeverity.WARNING,
                        source,
                        f"runtime artifact {field_name} differs from corpus baseline",
                        contract=label,
                        expected=expected[field_name],
                        actual=actual.get(field_name),
                    )
                )
    return tuple(alarms)


def _known_bad_alarms(contracts: tuple[dict[str, Any], ...], known_bad: Mapping[str, Any]) -> tuple[RuntimeAlarm, ...]:
    bad_artifacts = _artifact_entries(known_bad, keys=("known_bad_artifacts", "artifacts"))
    alarms: list[RuntimeAlarm] = []
    for contract in contracts:
        artifact = _artifact_summary(contract)
        label = f"{_required_str(contract, 'name')}:{_required_str(contract, 'kind')}"
        for bad in bad_artifacts:
            if not _same_artifact_identity(artifact, bad):
                continue
            matched_fields = tuple(
                field_name
                for field_name in ("sha256", "version", "revision")
                if field_name in bad and artifact.get(field_name) == bad[field_name]
            )
            if matched_fields:
                alarms.append(
                    RuntimeAlarm(
                        "runtime-known-bad-artifact",
                        RuntimeAlarmSeverity.ERROR,
                        RuntimeAlarmSource.KNOWN_BAD,
                        "runtime artifact matches a known-bad artifact version",
                        contract=label,
                        actual={field: artifact.get(field) for field in matched_fields},
                        reason=str(bad.get("reason", "known-bad artifact version")),
                    )
                )
    return tuple(alarms)


def _artifact_summary(contract: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(contract.get("artifact"), "contract artifact")


def _same_artifact_identity(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    for field_name in ("name", "kind"):
        if field_name in expected and actual.get(field_name) != expected[field_name]:
            return False
    return True


def _artifact_entries(
    payload: Mapping[str, Any],
    *,
    keys: Sequence[str] = ("artifact_baseline", "artifacts"),
) -> tuple[dict[str, Any], ...]:
    for key in keys:
        raw = payload.get(key)
        if raw is None:
            continue
        if not isinstance(raw, list):
            raise RuntimeAlarmError(f"manifest field {key!r} must be a list")
        entries: list[dict[str, Any]] = []
        for index, item in enumerate(raw):
            if not isinstance(item, dict):
                raise RuntimeAlarmError(f"manifest field {key!r} entry {index} must be an object")
            if not isinstance(item.get("name"), str) or not isinstance(item.get("kind"), str):
                raise RuntimeAlarmError(f"manifest field {key!r} entry {index} must include name and kind")
            entries.append(dict(item))
        return tuple(entries)
    return ()


def _load_lockfile_input(lockfile: str | Path | Lockfile) -> tuple[Lockfile, str | None]:
    if isinstance(lockfile, Lockfile):
        return lockfile, None
    path = Path(lockfile)
    return load_lockfile(path), path.as_posix()


def _load_json_input(value: str | Path | Mapping[str, Any], context: str) -> tuple[dict[str, Any], str | None]:
    if isinstance(value, Mapping):
        return dict(value), None
    path = Path(value)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeAlarmError(f"{context} not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeAlarmError(f"{context} is not valid JSON at {path}:{exc.lineno}:{exc.colno}: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise RuntimeAlarmError(f"{context} root must be a JSON object")
    return payload, path.as_posix()


def _alarm_counts(alarms: tuple[RuntimeAlarm, ...]) -> dict[str, int]:
    return {severity.value: sum(1 for alarm in alarms if alarm.severity is severity) for severity in RuntimeAlarmSeverity}


def _family_count(attestation: Mapping[str, Any], family: str) -> int:
    families = _mapping(attestation.get("contract_families", {}), "contract_families")
    value = families.get(family, 0)
    return value if isinstance(value, int) else 0


def _required_str(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise RuntimeAlarmError(f"runtime alarm input field {key!r} is required")
    return value


def _mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeAlarmError(f"{context} must be an object")
    return value


def _string_sequence(value: object, field_name: str) -> tuple[str, ...]:
    if value in (None, ()):
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise RuntimeAlarmError(f"policy pack field {field_name!r} must be a list of non-empty strings")
    return tuple(value)
