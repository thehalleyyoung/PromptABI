# Air-gapped installation

PromptABI's core checks are local and CPU-only, so an offline deployment is a
packaging problem rather than a degraded verification mode. Build the bundle on a
connected staging host, review it, move it into the isolated network, then prove
the installed copy against the same fixture and reproducibility checks used by
CI.

## Connected staging bundle

Create a wheelhouse that contains PromptABI, optional checker dependencies, and
the pinned Z3 wheel used by the paper artifact:

```bash
python -m pip wheel --wheel-dir vendor/wheelhouse . ".[grammars,solver,tokenizers]" z3-solver==4.15.4
python -m pip hash vendor/wheelhouse/*.whl > vendor/wheelhouse/SHASUMS.txt
promptabi paper reproducibility --output-dir paper_artifact --benchmark-iterations 1 --force
promptabi corpus provider-fixture-manifest --output vendor/provider-fixture-manifest.json
promptabi prompt-pack mirror build --config examples/prompt-packs/promptabi.json --mirror-dir vendor/prompt-pack-mirror --format json
```

Include these directories in the reviewed transfer artifact:

| Path | Offline purpose |
| --- | --- |
| `vendor/wheelhouse/` | Vendored Python wheels for PromptABI, optional tokenizer/grammar backends, and `z3-solver` |
| `fixtures/seed_corpus/` | Tokenizer/template seed corpus used by corpus and drift checks |
| `fixtures/structured_schemas/` | Structured-output, grammar, parser, and tool-schema fixtures |
| `fixtures/provider_fixture_packs/` | Secret-free provider request/response, tool, stop, streaming, error, and limit mirrors |
| `fixtures/real_bug_benchmarks/` and `fixtures/evaluation/` | Labeled real-bug and evaluation corpora |
| `paper_artifact/` | Fixture hashes, expected tables, solver pin, evaluator guide, and regeneration script |
| `vendor/prompt-pack-mirror/` | Content-addressed private prompt-pack mirror plus `prompt-pack-mirror.json` |

The provider fixture pack command validates that recorded provider mirrors are
download-free, anonymized, license-tagged, hashed, and free of credential-like
fields before they enter the offline bundle.

## Offline install

On the isolated host, install only from the reviewed wheelhouse:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --no-index --find-links vendor/wheelhouse promptabi z3-solver==4.15.4
promptabi doctor --config examples/minimal/promptabi.json
```

If the target platform cannot use the staged `z3-solver` wheel, vendor an
approved platform wheel or an internally built Z3 binary package with the same
review record. Do not let pip resolve Z3 from the network inside the air-gapped
environment; the solver version is part of the verification provenance.

## Offline verification gates

Run a small deterministic gate after transfer:

```bash
promptabi verify --config examples/minimal/promptabi.json --require-lockfile
promptabi corpus provider-fixture-manifest --root fixtures/provider_fixture_packs --output provider-fixture-manifest.json
promptabi prompt-pack mirror verify --manifest vendor/prompt-pack-mirror/prompt-pack-mirror.json --format json
bash paper_artifact/reproduction_commands.sh
```

For private applications, point configs at local artifact paths, local prompt-pack
mirrors, and reviewed provider fixture packs. Keep `artifact-provenance` enabled
so every tokenizer, template, schema, provider fixture, and prompt pack remains
hash-pinned, source-reviewed, and license-tagged.

## Reproducibility checklist

1. Build wheels and fixture mirrors on a connected staging host.
2. Review `vendor/wheelhouse/SHASUMS.txt`, `paper_artifact/fixture_hashes.json`,
   `vendor/provider-fixture-manifest.json`, and
   `vendor/prompt-pack-mirror/prompt-pack-mirror.json`.
3. Transfer the reviewed archive into the isolated network.
4. Install with `--no-index --find-links` only.
5. Run `promptabi doctor`, provider fixture manifest generation, prompt-pack
   mirror verification, and the paper reproduction script.
6. Store the resulting manifests with the release evidence or audit bundle.
