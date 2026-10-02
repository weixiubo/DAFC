# Baseline validation

The baseline public commit is `8b0eae11468195ce86ccb33fea980334c72ac2d6`.
Its 22 tests and five subtests remain in the maintained suite, with import/owner
mapping documented in TEST_MIGRATION.json. The selected original implementations
used for regression comparisons are hash-bound in tests/reference/MANIFEST.json. See VALIDATION.md for current counts,
deterministic controller comparisons and checks that require external resources.

The historical pilot-source mappings in MODULE_MIGRATION.json describe the
earlier archive-to-public-tree transition. For this release's installed commands
and module ownership use MIGRATION.md. Unused wrappers are removed from the
working tree; earlier versions remain in normal Git history. Required original
comparison functions are explicit test inputs rather than installation entrypoints.
