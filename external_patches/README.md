# external_patches

`external/` (EasyOCR, OpenOCR, parseq clones) is not committed. This folder holds everything we changed or added there.

- `UPSTREAM.txt` — upstream URL + exact commit each clone was checked out at.
- `<repo>.diff` — `git diff` of tracked upstream files we modified.
- `<repo>/...` — files we added (configs, training/eval scripts, charsets).

Restore:

```bash
git clone <url> external/<repo> && cd external/<repo> && git checkout <commit>
git apply ../../external_patches/<repo>.diff
cp -r ../../external_patches/<repo>/* .
```
