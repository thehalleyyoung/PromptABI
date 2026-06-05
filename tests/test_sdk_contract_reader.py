from promptabi.sdk_contract_reader import (
    IRFindingKind,
    ProviderContract,
    render_ir_validation_text,
    to_ir,
    validate_ir,
)


def _contract() -> ProviderContract:
    return ProviderContract(
        contract_id="c1",
        obligations=("stop-terminates", "role-non-forgeable"),
        required_capabilities=("json_schema",),
        max_total_tokens=8000,
        stop_sequences=("\n\n",),
    )


def test_ir_roundtrips_and_validates():
    ir = to_ir(_contract())
    result = validate_ir(ir)
    assert result.readable
    assert ir["contract_id"] == "c1"


def test_missing_field_flagged():
    ir = to_ir(_contract())
    del ir["obligations"]
    result = validate_ir(ir)
    kinds = {f.kind for f in result.findings}
    assert IRFindingKind.MISSING_FIELD in kinds


def test_empty_obligations_flagged():
    ir = to_ir(_contract())
    ir["obligations"] = []
    result = validate_ir(ir)
    kinds = {f.kind for f in result.findings}
    assert IRFindingKind.EMPTY_OBLIGATIONS in kinds


def test_bad_version_flagged():
    ir = to_ir(_contract())
    ir["version"] = "wrong"
    result = validate_ir(ir)
    kinds = {f.kind for f in result.findings}
    assert IRFindingKind.BAD_VERSION in kinds


def test_render_smoke():
    out = render_ir_validation_text(validate_ir(to_ir(_contract())))
    assert out.endswith("\n")
