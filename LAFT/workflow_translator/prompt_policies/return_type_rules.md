## Return Type Rules (CRITICAL)

**Rule 1: Match the return count to the signature**

If signature policy says:
```
Return updated values: theta, field_a, field_b, field_c, accum, scalar_out, scheme_name, errmsg, errflg
```

Then BOTH wrapper and core must return EXACTLY these values:
```python
def proc_core(...):
    return theta, field_a, field_b, field_c, accum, scalar_out, scheme_name, errmsg, errflg  # ✅ 9 returns

def proc(...):
    return theta, field_a, field_b, field_c, accum, scalar_out, scheme_name, errmsg, errflg  # ✅ 9 returns
```

**Rule 2: Scalar outputs are RETURNED, not passed as parameters**

```python
# ❌ Wrong
def proc(n, result):      # result is an output, should not be a parameter
    result = compute(n)
    return None

# ✅ Correct
def proc(n):
    result = compute(n)
    return result
```

**Rule 3: Arrays modified in-place must be returned**

Fortran INOUT arrays must be returned (JAX does not allow mutation):
```python
def proc(a, n):
    a = a.at[0].set(n)    # functional update
    return a              # ✅ return the new array
```

**Rule 4: Core returns ONLY JAX-compatible types**

Core can return:   ✅ JAX arrays, ✅ Python numbers, ✅ JAX scalars
Core CANNOT return: ❌ Strings, ❌ None (if signature says return something)

Wrapper can return strings; core cannot.
