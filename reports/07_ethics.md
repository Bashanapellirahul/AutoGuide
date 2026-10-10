# Data provenance and limits

Official recall responses retain URLs, retrieval timestamps, and SHA-256
hashes. NHTSA feature publications are collected through the Federal Register
API, with full raw text, publication dates, and provenance preserved. The
NHTSA website's HTTP 403 restriction is not bypassed.

Model-year recall matching does not establish an open recall for a particular
VIN. The app links to the official VIN lookup and refuses VIN verification.
It cannot predict component failure or verify equipment on an individual
vehicle. Dated rule publications are not a current legal compliance service.

API keys are optional, entered as password fields or loaded from ignored
`.env` files. They are not written to chat history or source records. Source
passages run locally; AI summaries send the question and retrieved evidence
to OpenAI. Local models and embeddings are downloaded from Hugging Face.

The required 150 webpages, five PDFs, and three table-containing pages are
not yet collected/validated. This corpus should not be described as meeting
those assignment requirements. Final human-reviewed evaluation is pending.
