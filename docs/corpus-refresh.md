# Annual corpus refresh procedure

A read-only refresh plan that updates PromptABI's corpus without losing longitudinal evidence.

## Metrics

- **seed model families:** 10
- **structured schema cases:** 5
- **provider fixture packs:** 6
- **real bug cases:** 7

## Commands

- `promptabi corpus verify --format text`
- `promptabi release compatibility-audit ...`
- `promptabi corpus leaderboard --format text`
- `promptabi maintain refresh --output-dir ... --force`
