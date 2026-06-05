# Diagnostic localization

PromptABI diagnostics are English-first today, but their public shape is ready
for translation without requiring downstream tools to parse prose.

Each diagnostic has a stable `rule_id`, severity, check modes, witness data, and
a localization key. If a checker provides an explicit `message_id`, that key is
used. Otherwise PromptABI derives a deterministic key from the rule, such as
`promptabi.diagnostic.repository.skeleton`.

```python
from promptabi import Diagnostic, DiagnosticSeverity

diagnostic = Diagnostic(
    rule_id="artifact-missing",
    severity=DiagnosticSeverity.ERROR,
    message="artifact {name} does not exist",
    message_id="promptabi.diagnostic.artifact_missing",
    message_args=(("name", "schema"),),
)
```

Message templates use strict Python-format placeholders with simple identifier
names only. Catalog generation rejects missing, duplicate, unused, or invalid
placeholders so translations can be checked before release.

```bash
promptabi diagnostics catalog --config examples/minimal/promptabi.json --format json
```

The command runs real verification, then emits a deterministic English catalog:
message ID, locale, default message, related rule IDs, severities, and
placeholders. The regular text, JSON, SARIF, HTML, and GitHub annotation renders
continue to display the same English messages, preserving existing CI behavior.
