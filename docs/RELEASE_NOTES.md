# Version 0.2.0

- Installed `dafc` commands provide diagnostics, examples, configured execution
  and evidence reproduction.
- Coordinate logic, registration, transition state, device operations, model
  adapters and receipt storage have explicit component boundaries.
- Model loading uses local checkpoints and explicit framework imports.
- Paths resolve from configuration files; services own their backend instances.
- Run recovery checks source, inputs, configuration and service receipts.
  Atomic recovery requires a durable action outcome before continuing.

See [migration](MIGRATION.md) for command changes and
[validation](VALIDATION.md) for regression and distribution checks.
