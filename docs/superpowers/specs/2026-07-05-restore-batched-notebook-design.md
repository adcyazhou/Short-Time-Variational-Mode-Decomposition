# Restore the batched STVMD notebook

## Goal

Add the last batch-optimized blast-analysis notebook from commit `b20ece7` to
the current repository without replacing the exact-original-STVMD notebook.

## Design

- Read `blast_multichannel_stvmd.ipynb` exactly from commit `b20ece7`.
- Add that historical content as `blast_multichannel_stvmd_batched.ipynb`.
- Keep `blast_multichannel_stvmd.ipynb` unchanged.
- Do not modify or stage the user's `5m.TXT`, `10m.TXT`, or `15m.TXT` files.
- Preserve the historical notebook's algorithm, parameters, batching behavior,
  figures, and output path without modernization.

## Verification

- Confirm the restored file is byte-for-byte identical to the historical blob.
- Confirm both notebook files are valid JSON.
- Confirm the restored notebook contains `BATCH_WINDOWS` and the batched STVMD
  implementation.
- Run the focused notebook tests without executing the expensive full analysis.
