## I/O policy for this procedure (MANDATORY)

This procedure contains Fortran I/O (PRINT/READ/WRITE/OPEN/CLOSE/etc).

Rules:
- `_core` MUST contain **NO I/O**. No `print`, no `input`, no file operations.
- All I/O must be moved into the Python wrapper `<PROC>(...)` only.
- If this Fortran procedure is *purely I/O* (e.g., prints a matrix and returns nothing),
  you MUST generate **wrapper-only** code:
  - Provide `<PROC>(...)` with Python I/O (printing/logging).
  - OMIT `<PROC>_core(...)` entirely (do not generate a fake core).
- If the procedure mixes compute + I/O, keep compute in `_core`, I/O in wrapper.
