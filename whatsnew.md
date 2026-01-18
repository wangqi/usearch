# USearch Upgrade: v2.21.3 to v2.23.0

**Upgrade Date:** 2026-01-14
**Previous Version:** v2.21.3 (branch: v20251129)
**New Version:** v2.23.0 (branch: tag-v2.23.0)

---

## Summary

This upgrade brings **3 critical bug fixes** in the core HNSW index implementation that improve stability and correctness, plus **1 performance optimization** for read-heavy workloads. No Swift API changes were made.

---

## Changes Affecting iOS/macOS

### Bug Fixes (Core Library)

| Fix | Impact | Risk |
|-----|--------|------|
| **Conditional lock release without acquire** (#681) | Prevents potential mutex state corruption in `index_gt::update` when updating existing vectors | Medium |
| **Skip key rewrites in update** (#679) | Avoids unnecessary key writes when key hasn't changed, reducing write amplification | Low |
| **Static thread safety fix** (#8938d6e) | Fixes thread-unsafe static variable access | Medium |

### Performance Improvements

| Improvement | Impact | Risk |
|-------------|--------|------|
| **Fewer conditional locks for immutable index views** (#689) | Search operations on immutable indexes no longer acquire unnecessary locks, improving read throughput | Low |

### Other Changes

| Change | Description | Impact on iOS |
|--------|-------------|---------------|
| SimSIMD bump | Updated to match v2.23.0 | None - SIMD optimizations |
| Filtered search callback typedef | Added `usearch_filtered_search_callback_t` type | None - GoLang only |
| Result set assertion | Added safety assertion in `back()` for empty results | None - C++ only |

---

## Changes NOT Affecting iOS/macOS

- **GoLang bindings**: Filtered search, missing APIs, vector lifetime fixes
- **Python**: 3.14 support
- **Windows JNI**: ISA extension fixes
- **CI/CD**: Workflow updates

---

## Risk Assessment

### Overall Risk: **LOW**

| Category | Assessment |
|----------|------------|
| **API Compatibility** | No breaking changes. Swift API unchanged. |
| **Binary Compatibility** | Compatible. Version bump only (2.21.3 -> 2.23.0) |
| **Behavioral Changes** | Bug fixes improve correctness, no semantic changes |
| **Performance** | Improved (fewer locks on read path) |

### Detailed Risk Analysis

#### Low Risk Factors
1. **No Swift file changes** - The `swift/` directory has no modifications
2. **Bug fixes are defensive** - They fix edge cases, not core algorithms
3. **Performance change is opt-in** - Only affects immutable index views

#### Medium Risk Factors
1. **Mutex/lock behavior changed** - Could expose latent bugs if code relied on incorrect behavior
2. **Thread safety fix** - May change timing characteristics in multithreaded scenarios

#### Mitigation Steps
1. Run existing USearch test suite (`testcases/libs/USearchRelevanceScorerTests.swift`)
2. Test concurrent search operations
3. Verify index persistence (save/load cycle)

---

## Commits Included

```
a4557ae Merge remote-tracking branch 'refs/remotes/origin/main'
f5e597e Update simsimd submodule to upstream v2.23.0 version
50ff651 Merge tag 'v2.23.0'
7306bb4 Release: v2.23.0 [skip ci]
43ee8b7 Add: Python 3.14 support (#693)
5fd2d3a Release: v2.22.0 [skip ci]
9f281e5 Merge: GoLang, Perf, & Refreshed CI (#690)
4f02166 Improve: Fewer conditional locks for immutable index views (#689)
8938d6e Fix: static is not thread safe
a2f1759 Release: v2.21.4 [skip ci]
c604a90 Fix: Progress callback passing & minor checks
6387134 Fix: Skip key rewrites in `index_gt::update` (#679)
c001112 Fix: Correct conditional lock release w/out acquire (#681)
```

---

## Testing Recommendations

1. **Unit Tests**: Run `./run_tests.sh --no-build USearchRelevanceScorerTests`
2. **Concurrency Test**: Verify parallel search operations work correctly
3. **Persistence Test**: Save index, reload, verify search results match
4. **Memory Test**: Monitor for leaks during heavy usage

---

## Float16 Patch Status

The custom Float16 removal patch for macOS (commit `3f9d408`) needs to be **reapplied** after merge, as upstream does not include this fix. The patch affects:
- `swift/USearchIndex.swift`
- `swift/USearchIndex+Sugar.swift`

This patch wraps Float16-specific methods with `#if !os(macOS)` to avoid build failures on macOS where Float16 has limited hardware support.
