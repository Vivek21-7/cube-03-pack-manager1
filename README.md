# Pack Manager (CUBE Track 03)

Outbound open-package verification. **This README is a stub** until the full
eval report and documentation pass; the working demo is already runnable.

## Demo (correct-order case)

```bash
python -m venv .venv
.venv\Scripts\pip install -e ".[dev]"
.venv\Scripts\python -m pack_manager.demo --write-images
.venv\Scripts\pytest
```
