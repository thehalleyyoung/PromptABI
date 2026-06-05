# Checker design guide

PromptABI checks are small proof-producing programs over discrete LLM interface
artifacts. A checker should name the unsafe or impossible state precisely,
choose a `CheckMode`, and emit diagnostics that distinguish proof, bounded
evidence, heuristic evidence, and abstention.

## Required design notes

Each checker proposal should document:

1. The artifacts consumed and the exact source spans used for diagnostics.
2. The supported fragment and the cases that must produce an abstention instead
   of a silent pass.
3. The witness format, including rendered strings, token IDs, parser states,
   solver assignments, truncation decisions, or minimized fields when relevant.
4. Differential evidence against real libraries or providers when the checker
   models external semantics.
5. Property tests covering safe, unsafe, ambiguous, unsupported, satisfiable,
   unsatisfiable, and timeout-prone cases where applicable.

Checks should fail closed on malformed artifacts, keep diagnostic ordering
deterministic, and avoid broad fallbacks that hide unsupported behavior.
