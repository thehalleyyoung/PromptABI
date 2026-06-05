from promptabi.provider_migration_dryrun import (
    ParamSpec,
    PatchOpKind,
    TargetSchema,
    dry_run_migration,
    render_migration_patch_text,
)


def _target() -> TargetSchema:
    return TargetSchema(
        accepted={
            "max_output_tokens": ParamSpec("max_output_tokens", max_value=4096),
            "temperature": ParamSpec("temperature", max_value=2.0),
        },
        renames={"max_tokens": "max_output_tokens"},
    )


def test_rename_and_keep():
    patch = dry_run_migration(
        {"max_tokens": 100, "temperature": 0.5}, _target()
    )
    kinds = {op.kind for op in patch.ops}
    assert PatchOpKind.RENAME in kinds
    assert PatchOpKind.KEEP in kinds
    assert patch.target_request["max_output_tokens"] == 100
    assert not patch.lossy


def test_clamp_is_lossy():
    patch = dry_run_migration({"temperature": 5.0}, _target())
    kinds = {op.kind for op in patch.ops}
    assert PatchOpKind.CLAMP in kinds
    assert patch.lossy
    assert patch.target_request["temperature"] == 2.0


def test_drop_unsupported_is_lossy():
    patch = dry_run_migration({"top_k": 40}, _target())
    kinds = {op.kind for op in patch.ops}
    assert PatchOpKind.DROP in kinds
    assert patch.lossy
    assert "top_k" not in patch.target_request


def test_render_smoke():
    out = render_migration_patch_text(dry_run_migration({"temperature": 1.0}, _target()))
    assert out.endswith("\n")
