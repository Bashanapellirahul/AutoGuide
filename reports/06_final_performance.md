# Development validation (not final assignment results)

Initial collected corpus: 24 recall documents, five NHTSA Federal Register
feature publications, and 1,169 structure-strategy chunks in ChromaDB.

The 18-question development set contains 12 supported retrieval questions
and six refusal questions. With hybrid retrieval and cross-encoder reranking:

| Metric | Observed value |
|---|---:|
| Document hit rate at 5 | 1.0000 |
| Document recall at 5 | 0.9583 |
| Document precision at 5 | 0.8944 |
| Mean reciprocal rank at 5 | 1.0000 |
| Refusal accuracy | 1.0000 |

No-keyword-match and low-semantic-relevance refusals both count as appropriate
insufficient-evidence refusals. The detailed case results and query latencies
are in ignored local `data/evaluation/hybrid.json`. Latency was measured on
the development computer and is not a controlled benchmark.

These results measure retrieval and refusal behavior, not answer faithfulness
or completeness. The set was manually labelled from this small development
corpus, and is not an independent final test set. Paid OpenAI generation has
not been exercised; mocked tests validate the API result/reference flow.
The larger required webpage/PDF/table corpus and final evaluation are pending.
