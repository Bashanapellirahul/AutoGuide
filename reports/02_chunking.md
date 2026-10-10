# Chunking implementation

Both strategies count cl100k tokens, target 500 tokens, and use 100-token
overlap within long text windows. Fixed chunking windows the complete document.
Structure-aware chunking packs known headings/recall sections into bounded
chunks; short recall documents stay intact. Oversized sections use windows.
Publications without heading metadata currently use fixed windows.

Chunk IDs include document identity, strategy, size, overlap, position, and
text content. URLs, campaign numbers, vehicle fields, retrieval/publication
dates, and recall summary/consequence/remedy metadata survive chunking.
`chunks.jsonl` is atomically replaced; the semantic index removes obsolete IDs.

Current initial corpus: 24 recall documents and five NHTSA Federal Register
publications. Structure strategy produced 1,169 chunks. This count is a
development snapshot and will change with source refreshes.

Tests cover size, overlap, identity stability, provenance, short section
preservation, invalid settings, and empty-corpus handling. A controlled
fixed-versus-structure retrieval study remains to be conducted.
