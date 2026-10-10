# Answers and evidence

Source-passages mode works without a paid API. Recall answers quote normalized
summary, safety consequence, and remedy fields. Feature answers select a few
substantive explanatory sentences across retrieved chunks, excluding bare
headings, with a reference for each sentence. They remain extractive answers,
not free-form generated paraphrases. Source titles, URLs, and dates appear in
the widget.

Optional AI-summary mode uses the configured OpenAI model and API key. The
prompt instructs it to use evidence only, cite numbered references, and avoid
unsupported predictions, VIN claims, or instructions embedded in documents.
API errors and nonexistent/absent citation numbers produce visible failures.

The evidence layer refuses missing data, unsupported vehicles/components,
conflicting vehicle filters, failure predictions, VIN lookup, vehicle feature
fitment, and current legal requirements not verified by the dated corpus.
Keyword overlap is a heuristic relevance gate, not proof of sufficiency.
Citation validation verifies reference IDs, not semantic entailment of every
claim. Human answer-faithfulness evaluation is still required. Automated LLM
tests use a mock client; a paid generation request has not been tested without
a user-provided API key.
