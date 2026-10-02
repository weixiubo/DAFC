# 0.2.0

Portable installed commands replace pilot wrappers. Coordinate logic, registration,
transition state, device operations, model strategies, experiment phases and
receipt storage now have explicit boundaries. Model loading imports torch where
used and only loads local checkpoints. Runtime paths no longer bind to the shell
working directory during import. Services own their backend rather than changing
another module's globals.

Current run recovery is bound to source, input, configuration and service receipts;
atomic ledger recovery refuses ambiguous action replay. Old signed research
artifacts and verification source remain frozen with their original identities.

This is a locally validated source release, with device/GPU integrations pending
resource restoration. Nothing has been published automatically. See VALIDATION.md.
