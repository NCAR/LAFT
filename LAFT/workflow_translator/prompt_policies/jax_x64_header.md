## Required imports (MANDATORY - every generated file must start with this)
```python
import os
os.environ["JAX_ENABLE_X64"] = "1"

import functools
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import lax
```

The `os.environ` line MUST come before `import jax`. Without this, float64 silently truncates to float32.
`import functools` is required for the `@functools.partial(jax.jit, ...)` decorator on `_core`.
