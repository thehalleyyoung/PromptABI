# RAG truncation boundary

The buggy app packs retrieved chunks with missing citation metadata, tokenizer
drift, and a retrieval payload limit that is smaller than the rendered chunk.
The fixed contract pins serving-tokenizer-compatible chunk sizes and preserves
required citation labels.

```bash
promptabi verify --config examples/end-to-end/rag-truncation/buggy.promptabi.json
promptabi verify --config examples/end-to-end/rag-truncation/fixed.promptabi.json
```

