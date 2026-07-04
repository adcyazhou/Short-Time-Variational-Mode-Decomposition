# Restore Batched STVMD Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore the batch-optimized blast STVMD notebook from commit `b20ece7` under a new filename without changing the current notebook.

**Architecture:** Treat the historical notebook as an immutable artifact. Extract the exact Git blob into a temporary directory, move it to the new filename, then verify byte identity, JSON validity, and the expected batching markers.

**Tech Stack:** Git, PowerShell, Jupyter Notebook JSON, pytest

---

### Task 1: Restore and verify the historical notebook

**Files:**
- Create: `blast_multichannel_stvmd_batched.ipynb`
- Preserve: `blast_multichannel_stvmd.ipynb`
- Test: `tests/test_blast_multichannel_stvmd_notebook.py`

- [ ] **Step 1: Record the current notebook hash and confirm the destination is absent**

Run:

```powershell
$currentHash = (Get-FileHash -Algorithm SHA256 -LiteralPath 'blast_multichannel_stvmd.ipynb').Hash
if (Test-Path -LiteralPath 'blast_multichannel_stvmd_batched.ipynb') {
    throw 'Destination already exists'
}
Write-Output "CURRENT_HASH=$currentHash"
```

Expected: one `CURRENT_HASH` line and no exception.

- [ ] **Step 2: Extract the exact historical notebook to the new filename**

Run:

```powershell
$workspace = (Resolve-Path -LiteralPath '.').Path
$tempDir = Join-Path $workspace '.restore-batched-notebook'
if ([System.IO.Path]::GetFullPath($tempDir) -notlike "$workspace*") {
    throw 'Temporary path escaped the workspace'
}
New-Item -ItemType Directory -Path $tempDir -Force | Out-Null
git archive --format=tar --output="$tempDir\historical.tar" b20ece7 blast_multichannel_stvmd.ipynb
tar -xf "$tempDir\historical.tar" -C $tempDir
Move-Item -LiteralPath "$tempDir\blast_multichannel_stvmd.ipynb" -Destination "$workspace\blast_multichannel_stvmd_batched.ipynb"
```

Expected: `blast_multichannel_stvmd_batched.ipynb` exists at the repository root.

- [ ] **Step 3: Verify byte identity and preserve the current notebook**

Run:

```powershell
$historicalBlob = (git rev-parse 'b20ece7:blast_multichannel_stvmd.ipynb').Trim()
$restoredBlob = (git hash-object 'blast_multichannel_stvmd_batched.ipynb').Trim()
$currentHashAfter = (Get-FileHash -Algorithm SHA256 -LiteralPath 'blast_multichannel_stvmd.ipynb').Hash
Write-Output "HISTORICAL_BLOB=$historicalBlob"
Write-Output "RESTORED_BLOB=$restoredBlob"
Write-Output "BLOB_MATCH=$($historicalBlob -eq $restoredBlob)"
Write-Output "CURRENT_UNCHANGED=$($currentHash -eq $currentHashAfter)"
```

Expected: `BLOB_MATCH=True` and `CURRENT_UNCHANGED=True`.

- [ ] **Step 4: Validate notebook structure and batching markers**

Run:

```powershell
python -m json.tool blast_multichannel_stvmd_batched.ipynb *> $null
rg -n '"BATCH_WINDOWS = 64 if QUICK_TEST else 256"|def run_dynamic_stvmd_batched|batch_windows=BATCH_WINDOWS' blast_multichannel_stvmd_batched.ipynb
```

Expected: JSON validation exits with code 0 and all three batching markers are found.

- [ ] **Step 5: Run the focused notebook test suite**

Run:

```powershell
python -m pytest tests/test_blast_multichannel_stvmd_notebook.py -q
```

Expected: all tests pass without running the expensive full STVMD analysis.

- [ ] **Step 6: Remove the verified temporary directory**

Run:

```powershell
$resolvedTemp = [System.IO.Path]::GetFullPath($tempDir)
if ($resolvedTemp -notlike "$workspace*") {
    throw 'Refusing to remove a path outside the workspace'
}
Remove-Item -LiteralPath $resolvedTemp -Recurse -Force
```

Expected: `.restore-batched-notebook` no longer exists.

- [ ] **Step 7: Commit only the restored notebook**

Run:

```powershell
git add -- blast_multichannel_stvmd_batched.ipynb
git commit -m "feat: restore batched STVMD notebook"
git status -sb
```

Expected: the commit contains only the restored notebook; the user's TXT files remain untracked.
