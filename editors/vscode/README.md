# PromptABI VS Code extension

This minimal extension turns the real `promptabi diagnostics lsp` command into
inline VS Code diagnostics. It also registers quick commands to rerun checks,
open `promptabi explain` for the selected finding, and preview rendered or
tokenized witness steps from PromptABI diagnostics.

Run it from this directory with VS Code's extension host. The extension has no
runtime npm dependencies; it shells out to the configured `promptabi`
executable, so local behavior matches CI and the CLI exactly.
