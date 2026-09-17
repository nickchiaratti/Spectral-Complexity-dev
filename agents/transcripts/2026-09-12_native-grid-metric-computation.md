# 2026-09-12 — Native-Grid Metric Computation Architecture Discussion

## Summary
Discussion about refactoring the Spectral Complexity pipeline to compute neighborhood-aware metrics on each sensor's native coordinate grid before warping only the scalar metric results to a harmonized display grid.

## Key Technical Points Established

### 1. NN Resampling Preserves Radiometry but Corrupts Spatial Neighborhoods
- Nearest Neighbor preserves the exact spectral vector (critical for spectral analysis)
- However, NN duplicates and drops pixels to resolve grid offsets
- The 3x3 spectral complexity kernel is a spatial neighborhood metric
- NN duplication creates artificial zero-variance boundaries within the kernel window

### 2. Quantified Geometric Error
- col_offset/row_offset math introduces up to +/-15m shift
- On homogeneous terrain: near 0% radiometric error
- On heterogeneous terrain: up to 100%+ error at land-cover boundaries
- The +/-15m jitter is absorbed by the 90m (3x3 at 30m) spatial averaging footprint

### 3. Pipeline Architecture
- Two pipeline paths: MGRS (default) and Dynamic Albers (legacy)
- For HLS on MGRS path, offsets evaluate to 0 (no actual shift)
- Warping error applies primarily to EnMAP, Tanager, Dragonette

### 4. Proposed Architecture
- Step 1: Stack imagery per-sensor in source coordinate grid (exists)
- Step 2: Compute metrics natively on source grid arrays (new)
- Step 3: Warp only scalar metric results to display grid (new)
- Downstream consumers require minimal changes

## Open Questions
1. Apply to all sensors uniformly or only non-HLS?
2. Cover both MGRS and Albers paths?
3. On-the-fly SR extraction from native files acceptable for viewer?
4. Persistent water mask handling?
