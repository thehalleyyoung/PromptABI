"""Offline training data-loader adapter fixture normalization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


REQUIRED_DATA_LOADER_FAMILIES: tuple[str, ...] = (
    "huggingface-datasets",
    "jsonl",
    "parquet-metadata",
    "axolotl",
    "llama-factory",
    "trl",
    "openrlhf",
    "internal",
)

_FAMILY_ALIASES = {
    "hf": "huggingface-datasets",
    "hf-datasets": "huggingface-datasets",
    "huggingface": "huggingface-datasets",
    "huggingface-datasets": "huggingface-datasets",
    "json-lines": "jsonl",
    "jsonl": "jsonl",
    "parquet": "parquet-metadata",
    "parquet-metadata": "parquet-metadata",
    "axolotl": "axolotl",
    "llama-factory": "llama-factory",
    "llamafactory": "llama-factory",
    "lmsys-llama-factory": "llama-factory",
    "trl": "trl",
    "openrlhf": "openrlhf",
    "internal": "internal",
    "internal-json": "internal",
}


class DataLoaderAdapterError(ValueError):
    """A malformed data-loader adapter fixture."""


@dataclass(frozen=True, slots=True)
class DataLoaderAdapterFixture:
    """A normalized, non-sensitive training data-loader adapter fixture."""

    name: str
    family: str
    dataset_format: str
    path: str
    kind: str
    split: str | None = None
    example_count: int | None = None
    content_fields: tuple[str, ...] = ()
    preference_fields: tuple[str, ...] = ()
    private_material: bool = False
    fingerprint: str | None = None

    def __post_init__(self) -> None:
        _require_non_empty("data-loader adapter name", self.name)
        _require_non_empty("data-loader adapter family", self.family)
        _require_non_empty("data-loader adapter dataset_format", self.dataset_format)
        _require_non_empty("data-loader adapter path", self.path)
        _require_non_empty("data-loader adapter kind", self.kind)
        if self.family not in REQUIRED_DATA_LOADER_FAMILIES:
            raise DataLoaderAdapterError(f"unsupported data-loader adapter family: {self.family}")
        _optional_non_empty("data-loader adapter split", self.split)
        _optional_non_negative("data-loader adapter example_count", self.example_count)
        _optional_non_empty("data-loader adapter fingerprint", self.fingerprint)
        object.__setattr__(self, "content_fields", _unique_strings(self.content_fields, "content_fields"))
        object.__setattr__(self, "preference_fields", _unique_strings(self.preference_fields, "preference_fields"))
        if self.kind == "preference" and not self.preference_fields:
            raise DataLoaderAdapterError(f"preference adapter '{self.name}' must declare preference_fields")


@dataclass(frozen=True, slots=True)
class DataLoaderAdapterProbe:
    """Evidence gathered by reading one local adapter fixture."""

    fixture: DataLoaderAdapterFixture
    resolved_path: str
    observed_count: int | None = None
    observed_fields: tuple[str, ...] = ()
    source_type: str = "metadata"

    def __post_init__(self) -> None:
        object.__setattr__(self, "observed_fields", _unique_strings(self.observed_fields, "observed_fields"))


@dataclass(frozen=True, slots=True)
class DataLoaderAdapterReport:
    """Deterministic summary of all data-loader adapter fixtures in a manifest."""

    probes: tuple[DataLoaderAdapterProbe, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "probes", tuple(sorted(self.probes, key=lambda probe: probe.fixture.name)))

    @property
    def families(self) -> tuple[str, ...]:
        present = {probe.fixture.family for probe in self.probes}
        return tuple(family for family in REQUIRED_DATA_LOADER_FAMILIES if family in present)

    @property
    def formats(self) -> tuple[str, ...]:
        return tuple(sorted({probe.fixture.dataset_format for probe in self.probes}))

    @property
    def sample_count(self) -> int:
        return sum(probe.observed_count or 0 for probe in self.probes)

    @property
    def complete_required_families(self) -> bool:
        return set(REQUIRED_DATA_LOADER_FAMILIES).issubset(self.families)

    def to_metadata(self) -> tuple[tuple[str, object], ...]:
        if not self.probes:
            return ()
        return (
            ("data_loader_adapter_count", len(self.probes)),
            ("data_loader_adapter_families", self.families),
            ("data_loader_adapter_formats", self.formats),
            ("data_loader_adapter_names", tuple(probe.fixture.name for probe in self.probes)),
            ("data_loader_adapter_required_families_complete", self.complete_required_families),
            ("data_loader_adapter_sample_count", self.sample_count),
            ("data_loader_adapter_private_material", any(probe.fixture.private_material for probe in self.probes)),
        )


def analyze_data_loader_adapters(raw_manifest: Mapping[str, object], *, base_dir: Path) -> DataLoaderAdapterReport:
    """Normalize and validate local data-loader adapter fixtures from manifest metadata."""

    metadata = _mapping(raw_manifest.get("metadata"))
    raw_adapters = metadata.get("data_loader_adapters", metadata.get("loader_adapters", ()))
    fixtures = _adapter_fixtures(raw_adapters)
    probes = tuple(_probe_fixture(fixture, base_dir=base_dir) for fixture in fixtures)
    return DataLoaderAdapterReport(probes=probes)


def _adapter_fixtures(raw_adapters: object) -> tuple[DataLoaderAdapterFixture, ...]:
    if raw_adapters in (None, ()):
        return ()
    if isinstance(raw_adapters, Mapping):
        adapters = []
        for name, value in sorted(raw_adapters.items()):
            if not isinstance(name, str) or not name:
                raise DataLoaderAdapterError("data_loader_adapters keys must be non-empty strings")
            if not isinstance(value, Mapping):
                raise DataLoaderAdapterError(f"data_loader_adapters.{name} must be an object")
            adapters.append({"name": name, **dict(value)})
    elif isinstance(raw_adapters, list):
        adapters = raw_adapters
    else:
        raise DataLoaderAdapterError("metadata.data_loader_adapters must be an object or list")

    fixtures: list[DataLoaderAdapterFixture] = []
    for index, item in enumerate(adapters, start=1):
        if not isinstance(item, Mapping):
            raise DataLoaderAdapterError("data-loader adapter entries must be objects")
        family = _canonical_family(_first_string(item, ("family", "adapter", "type"), default="jsonl"))
        preference_fields = _strings(item.get("preference_fields", ()), "preference_fields")
        kind = _string(item, "kind", default="preference" if preference_fields else "supervised")
        fixtures.append(
            DataLoaderAdapterFixture(
                name=_string(item, "name", default=f"adapter-{index}"),
                family=family,
                dataset_format=_string(item, "format", default=_default_format(family)),
                path=_first_string(item, ("path", "fixture")),
                kind=kind,
                split=_optional_string(item.get("split"), "split"),
                example_count=_optional_int(item.get("example_count"), "example_count"),
                content_fields=_strings(item.get("content_fields", ()), "content_fields"),
                preference_fields=preference_fields,
                private_material=_bool(item.get("private_material", False), "private_material"),
                fingerprint=_optional_string(item.get("fingerprint"), "fingerprint"),
            )
        )
    return tuple(fixtures)


def _probe_fixture(fixture: DataLoaderAdapterFixture, *, base_dir: Path) -> DataLoaderAdapterProbe:
    path = Path(fixture.path)
    if not path.is_absolute():
        path = base_dir / path
    if not path.is_file():
        raise DataLoaderAdapterError(f"data-loader adapter fixture '{fixture.name}' does not exist: {path}")

    if path.suffix.lower() == ".jsonl":
        rows = _read_jsonl(path)
        observed_fields = tuple(sorted({key for row in rows for key in row}))
        _require_fields(fixture, observed_fields)
        _check_count(fixture, len(rows))
        return DataLoaderAdapterProbe(
            fixture=fixture,
            resolved_path=str(path),
            observed_count=len(rows),
            observed_fields=observed_fields,
            source_type="jsonl-sample",
        )

    if path.suffix.lower() == ".json":
        payload = _read_json(path)
        observed_count, observed_fields, source_type = _probe_json_fixture(fixture, payload)
        _require_fields(fixture, observed_fields)
        _check_count(fixture, observed_count)
        return DataLoaderAdapterProbe(
            fixture=fixture,
            resolved_path=str(path),
            observed_count=observed_count,
            observed_fields=observed_fields,
            source_type=source_type,
        )

    raise DataLoaderAdapterError(
        f"data-loader adapter fixture '{fixture.name}' must be JSON or JSONL, got {path.suffix or '<none>'}"
    )


def _probe_json_fixture(fixture: DataLoaderAdapterFixture, payload: Mapping[str, object]) -> tuple[int | None, tuple[str, ...], str]:
    if fixture.family == "huggingface-datasets":
        splits = _mapping(payload.get("splits"))
        split_payload = _mapping(splits.get(fixture.split or "train"))
        count = _optional_int(split_payload.get("num_examples"), "splits.num_examples")
        features = _field_names(payload.get("features"))
        return count, features, "huggingface-datasets-metadata"

    if fixture.family == "parquet-metadata":
        count = _optional_int(payload.get("num_rows"), "num_rows")
        fields = _field_names(payload.get("columns"))
        if not fields:
            raise DataLoaderAdapterError(f"parquet metadata adapter '{fixture.name}' must declare columns")
        return count, fields, "parquet-metadata"

    if fixture.family == "axolotl":
        datasets = payload.get("datasets")
        if not isinstance(datasets, list) or not datasets:
            raise DataLoaderAdapterError(f"Axolotl adapter '{fixture.name}' must declare datasets")
        fields = _field_names(payload.get("field_mapping")) or _field_names(datasets[0] if isinstance(datasets[0], Mapping) else {})
        return _optional_int(payload.get("sample_count"), "sample_count"), fields, "axolotl-config"

    if fixture.family == "llama-factory":
        dataset_info = _mapping(payload.get("dataset_info"))
        fields = _llama_factory_fields(dataset_info)
        if not dataset_info:
            raise DataLoaderAdapterError(f"LLaMA-Factory adapter '{fixture.name}' must declare dataset_info")
        return _optional_int(payload.get("sample_count"), "sample_count"), fields, "llama-factory-dataset-info"

    if fixture.family == "internal":
        schema = _mapping(payload.get("schema"))
        fields = _field_names(schema.get("columns", payload.get("columns")))
        if not fields:
            raise DataLoaderAdapterError(f"internal adapter '{fixture.name}' must declare schema columns")
        return _optional_int(payload.get("sample_count"), "sample_count"), fields, "internal-schema"

    rows = payload.get("rows")
    if isinstance(rows, list) and all(isinstance(row, Mapping) for row in rows):
        fields = tuple(sorted({key for row in rows for key in row}))
        return len(rows), fields, "json-row-sample"
    return _optional_int(payload.get("sample_count"), "sample_count"), _field_names(payload.get("fields")), "json-metadata"


def _read_jsonl(path: Path) -> tuple[Mapping[str, object], ...]:
    rows: list[Mapping[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DataLoaderAdapterError(f"{path}:{line_number} is not valid JSONL: {exc.msg}") from exc
        if not isinstance(row, Mapping):
            raise DataLoaderAdapterError(f"{path}:{line_number} JSONL row must be an object")
        rows.append(row)
    if not rows:
        raise DataLoaderAdapterError(f"JSONL adapter fixture is empty: {path}")
    return tuple(rows)


def _read_json(path: Path) -> Mapping[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DataLoaderAdapterError(f"{path} is not valid JSON: {exc.msg}") from exc
    if not isinstance(payload, Mapping):
        raise DataLoaderAdapterError(f"{path} root must be a JSON object")
    return payload


def _require_fields(fixture: DataLoaderAdapterFixture, observed_fields: tuple[str, ...]) -> None:
    required = set(fixture.content_fields).union(fixture.preference_fields)
    missing = sorted(required.difference(observed_fields))
    if missing:
        raise DataLoaderAdapterError(
            f"data-loader adapter '{fixture.name}' fixture is missing declared fields: {', '.join(missing)}"
        )


def _check_count(fixture: DataLoaderAdapterFixture, observed_count: int | None) -> None:
    if fixture.example_count is not None and observed_count is not None and fixture.example_count != observed_count:
        raise DataLoaderAdapterError(
            f"data-loader adapter '{fixture.name}' expected {fixture.example_count} examples, observed {observed_count}"
        )


def _canonical_family(value: str) -> str:
    family = value.strip().lower().replace("_", "-")
    try:
        return _FAMILY_ALIASES[family]
    except KeyError as exc:
        raise DataLoaderAdapterError(f"unsupported data-loader adapter family: {value}") from exc


def _default_format(family: str) -> str:
    if family == "huggingface-datasets":
        return "hf-datasets"
    if family == "parquet-metadata":
        return "parquet"
    if family == "llama-factory":
        return "llama-factory-dataset-info"
    if family == "internal":
        return "internal-json"
    return family


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _field_names(value: object) -> tuple[str, ...]:
    if isinstance(value, Mapping):
        return tuple(sorted(key for key in value if isinstance(key, str) and key))
    if isinstance(value, list):
        fields: list[str] = []
        for item in value:
            if isinstance(item, str) and item:
                fields.append(item)
            elif isinstance(item, Mapping):
                name = item.get("name")
                if isinstance(name, str) and name:
                    fields.append(name)
        return tuple(sorted(dict.fromkeys(fields)))
    return ()


def _llama_factory_fields(dataset_info: Mapping[str, object]) -> tuple[str, ...]:
    fields: list[str] = []
    for spec in dataset_info.values():
        if not isinstance(spec, Mapping):
            continue
        columns = spec.get("columns")
        if isinstance(columns, Mapping):
            for key, value in columns.items():
                if isinstance(key, str) and key:
                    fields.append(key)
                if isinstance(value, str) and value:
                    fields.append(value)
        fields.extend(_field_names(spec.get("tags")))
    return tuple(sorted(dict.fromkeys(fields)))


def _string(mapping: Mapping[str, object], key: str, *, default: str | None = None) -> str:
    value = mapping.get(key, default)
    if not isinstance(value, str) or not value:
        raise DataLoaderAdapterError(f"data-loader adapter field '{key}' must be a non-empty string")
    return value


def _first_string(mapping: Mapping[str, object], keys: tuple[str, ...], *, default: str | None = None) -> str:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            if not isinstance(value, str) or not value:
                raise DataLoaderAdapterError(f"data-loader adapter field '{key}' must be a non-empty string")
            return value
    if default is not None:
        return default
    joined = ", ".join(keys)
    raise DataLoaderAdapterError(f"data-loader adapter must define one of: {joined}")


def _optional_string(value: object, key: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise DataLoaderAdapterError(f"data-loader adapter field '{key}' must be a non-empty string")
    return value


def _strings(value: object, key: str) -> tuple[str, ...]:
    if value in (None, ()):
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise DataLoaderAdapterError(f"data-loader adapter field '{key}' must be a list of non-empty strings")
    return tuple(value)


def _optional_int(value: object, key: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise DataLoaderAdapterError(f"data-loader adapter field '{key}' must be an integer")
    return value


def _bool(value: object, key: str) -> bool:
    if not isinstance(value, bool):
        raise DataLoaderAdapterError(f"data-loader adapter field '{key}' must be a boolean")
    return value


def _require_non_empty(field_name: str, value: str) -> None:
    if not value:
        raise DataLoaderAdapterError(f"{field_name} must be non-empty")


def _optional_non_empty(field_name: str, value: str | None) -> None:
    if value is not None and not value:
        raise DataLoaderAdapterError(f"{field_name} must be non-empty")


def _optional_non_negative(field_name: str, value: int | None) -> None:
    if value is not None and value < 0:
        raise DataLoaderAdapterError(f"{field_name} must be non-negative")


def _unique_strings(values: tuple[str, ...], field_name: str) -> tuple[str, ...]:
    if not all(isinstance(value, str) and value for value in values):
        raise DataLoaderAdapterError(f"{field_name} values must be non-empty strings")
    return tuple(sorted(dict.fromkeys(values)))
