# turf-efi

Open research library proposing and validating algorithms that approximate
concepts from FIFA's Enhanced Football Intelligence (EFI) system — an
independent effort to produce EFI-style metrics from raw tracking data using
open, paper-grounded methods.

**EFI is FIFA's closed-source system.** This repo does not implement FIFA's
definitions; it proposes research approximations and validates them against
tracking data. Divergence from FIFA's internal system is expected and is itself
informative.

Part of the **turf** ecosystem. Installable as `pip install turf-efi`;
importable as `turf.efi.*`.

```python
from turf.efi.line import detect_lines_frame, analyze_lines
from turf.efi.label import label_event, detect_line_break
from turf.efi.risk import build_expected_xt
```

## Extras

Pitch visualization requires the `viz` dependency group (mplsoccer + matplotlib):

```bash
uv sync --group viz
```
