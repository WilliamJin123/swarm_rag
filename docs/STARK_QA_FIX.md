# StarkQA Package Fix (Windows/Path Compat)

**Issue:** `stark_qa` crashes on Windows because it compares Windows paths (`\`) against Hugging Face paths (`/`). It also raises an `UnboundLocalError` if no files are found.
**File Location:** `.venv\lib\site-packages\stark_qa\tools\download_hf.py`

## Instructions to Re-Apply Fix

If you delete or recreate your `.venv`, you must re-apply this fix to `download_hf.py`:

1.  **Locate the function** `download_hf_folder`.
2.  **Add Path Normalization** at the very start of the function:
    ```python
    # Normalize Windows backslashes to forward slashes for HF compatibility
    folder = folder.replace("\\", "/")
    ```
3.  **Initialize the variable** before the `for` loop:
    ```python
    file_path = None
    ```
4.  **Guard the print statement** at the end of the function:
    ```python
    if file_path:
        print(f"Use file from {file_path}.")
    ```

## Why this is safe for Linux
The fix `folder.replace("\\", "/")` is system-agnostic. 
* **On Windows:** Converts `path\to\folder` -> `path/to/folder` (Matches HF format).
* **On Linux:** `path/to/folder` remains `path/to/folder` (No change).