# HOMcontainer.py — Wishlist & Roadmap

This document tracks planned improvements, requested features, and known
limitations of `HOMcontainer.py`. The script is under active development
and was submitted in an incomplete state, so this file will evolve rapidly.

Users are encouraged to open GitHub Issues or submit pull requests to
discuss or propose additional features.

---

## 1. Planned Features

- [ ] **Standalone command-line mode**  
      Allow usage like:  
      `HOMcontainer.py input.star output.star`

- [ ] **Batch processing mode**  
      Process multiple input STAR files and write merged output.

- [ ] **Optional “silent” / “verbose” levels** for debugging.

- [ ] **Flag to output only selected helical metadata columns**  
      E.g., twist, rise, angle priors, origins.

- [ ] **Add automatic detection of optics groups**  
      (Autofill optics table or warn if missing.)

- [ ] **Add basic validation** of STAR files before processing  
      (Are required tables present? Are column names present?)

- [ ] **Support for writing compressed output**  
      e.g., `output.star.gz`.

---

## 2. Improvements to Existing Code

- [ ] **Revisit `_read_star` design**  
      Possibly replace with direct calls to `starfile.read()` or abstract
      the logic more cleanly.

- [ ] **Simplify or remove `_write_star`**  
      If `starfile.write()` is sufficient, drop the custom writer.

- [ ] **Cleaner handling of `convert_path_fn`**  
      Currently ignored; either implement or document as not supported.

- [ ] **Ensure consistent DataFrame schema**  
      Some tables propagate extra or missing columns depending on input.

- [ ] **Improve docstrings**  
      Add clear examples showing usage and expected input/output.

---

## 3. Known Issues / Incomplete Areas

- [ ] **Incomplete treatment of helical metadata**  
      Some columns (twist, rise, angle priors) are not fully merged or
      validated across segments.

- [ ] **No unit tests yet**  
      Add tests for:  
        - STAR read/write  
        - Merging particle + optics tables  
        - Column validation  
        - Error handling

- [ ] **Error handling is minimal**  
      Exceptions should be caught and surfaced with meaningful messages.

- [ ] **Insufficient type checking**  
      Many parameters assume DataFrames or paths without checking.

---

## 4. User Requests / Open Questions

(Users may add here via Pull Request)

- [ ] <User-defined request goes here>

---

## Contributing

To request a feature:

1. Open a GitHub Issue **OR**
2. Submit a PR that adds an item to this wishlist.

Please describe the scientific use-case when possible; this helps prioritize
development in the context of helical metadata workflows.

