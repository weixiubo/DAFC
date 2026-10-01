# Validation and migration boundary

Six original post-check tests pass unchanged. Five portable atomic protocol
tests and four attempt-lifecycle tests retain their assertions. Four integral
coordinate repair tests run against the extracted normalizer. Three additional
regressions cover all original seeded geometry receipts (identity, translation,
blank target), source-image tampering and append-only ledger integrity.

The original integral-repair test could not collect because its frozen V2
analyzer was absent from the active tree; its source exists in preserved Git.
The portable normalizer uses the same two-actor protocol and leaves inputs
unchanged. Remote lifecycle and pinned workstation asset tests remain archived;
they are not reported as passing tests of this clean environment.

The original C587/C588 source, 1121 attempts and 3360 screenshots are preserved.
The full local raw verifier and result reconstruction are recorded separately.
Legacy module mappings are in `MODULE_MIGRATION.json`; old CLI names import the
installed package. Optional Android imports are delayed until actual use.
