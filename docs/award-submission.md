# Award submission brief

**Claim.** PromptABI is a CPU-only static verifier for LLM interface contracts that finds structural bugs before inference, fine-tuning, evaluation publication, provider migration, and deployment.

## Evidence

- Comparative cases: 16 with PromptABI detecting 16.
- Real-bug benchmark cases: 7.
- Stage-ready demos: 4.
- Leaderboard precision/recall: 1.0/1.0; solver reliability 1.0.

## Limitations

- does not prove model intent or sampled behavior
- claims are scoped to explicit supported fragments
- unsupported artifacts produce visible abstentions instead of false proofs

## Reproduction

- `promptabi corpus comparative-study --format markdown`
- `promptabi corpus leaderboard --benchmark-iterations 1 --format text`
- `promptabi conference-demo --format text`
- `promptabi paper reproducibility --output-dir paper_artifact --force`
