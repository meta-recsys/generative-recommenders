---
oncalls: ['generative_ranking']
description: "Prevent non-ASCII characters in code to avoid AOTI UnicodeEncodeError"
apply_to_regex: '\.(py|cpp|h|cu|cuh)$'
---
# No Non-ASCII Characters in Code

## Rule

When writing or modifying code files in the `generative_recommenders/` directory (including comments, docstrings, and string literals):

1. **NEVER use non-ASCII characters** such as:
   - Em-dash (`—` U+2014) - use regular hyphen (`-`)
   - En-dash (`–` U+2013) - use regular hyphen (`-`)
   - Smart/curly quotes (`"` `"` `'` `'`) - use straight quotes (`"` `'`)
   - Ellipsis (`…`) - use three periods (`...`)
   - Non-breaking space (U+00A0) - use regular space
   - Any other Unicode characters outside ASCII range (0-127)

2. **All text in code files must be ASCII-only** (characters 0x00-0x7F)

3. This rule applies to ALL text in code files:
   - Code comments (both inline `#` and block comments)
   - Docstrings (`"""..."""` and `'''...'''`)
   - String literals
   - Variable names and identifiers
   - Any text in `.py`, `.cpp`, `.h`, `.cu`, `.cuh` files

## Reason

Non-ASCII characters cause `UnicodeEncodeError` during AOT Inductor (AOTI) compilation. When PyTorch Inductor generates C++/CUDA source code, it writes files using ASCII encoding. If docstrings, comments, or other text containing non-ASCII characters are captured in the FX graph metadata, the file write operation fails with errors like:

```
UnicodeEncodeError: 'ascii' codec can't encode character '\u2014' in position 69453: ordinal not in range(128)
```

This breaks AOTI model publishing workflows even though the same code works fine in eager/JIT mode.

## Examples

### Bad (contains em-dash)
```python
# Config for this group — extract once, not per candidate
def process_config():
    """Process the config — returns a dict."""
    pass
```

### Good (uses regular hyphen)
```python
# Config for this group - extract once, not per candidate
def process_config():
    """Process the config - returns a dict."""
    pass
```

## Enforcement

Before committing code changes, verify no non-ASCII characters exist:
```bash
grep -rP '[^\x00-\x7F]' --include="*.py" fbcode/generative_recommenders/
```
