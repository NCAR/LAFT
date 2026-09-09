## Error handling policy (MANDATORY)
Many Fortran routines signal errors like this:
- `WRITE(errmsg, ...)`
- `errflg = 1`
- `RETURN`
Rules:
- `<PROC>_core(...)` MUST NOT print, write files, or build strings.
- Detect errors in `<PROC>_core(...)` by computing `errflg` as a JAX integer
  (0 = OK, nonzero = error). Use `jnp.where` or `lax.cond` — never Python `if`.
- `<PROC>_core(...)` RETURNS `errflg` (and updated arrays) but NOT `errmsg`.
- The wrapper `<PROC>(...)` handles `errmsg`:
  - Check `if int(errflg) != 0` (host conversion is fine in the wrapper).
  - Construct the message with an f-string and return early.
- Do not attempt Fortran `WRITE` formatting inside `<PROC>_core(...)`.