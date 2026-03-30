# fiicode-2026-ai-ml
bank telemarketing prediction

## Borrowed Top-3 Path

This repo now includes `borrowed_top3_submission.py`, which stages two submission
candidates using artifacts borrowed from
`C:\Users\dbxdr_iytiz92\Dropbox\fiicode`.

Run:

```bash
python borrowed_top3_submission.py
```

Outputs:

- `submission_top3_borrowed_exact_source/submission.csv`
- `submission_top3_borrowed_hybrid_rank3/submission.csv`
- `borrowed_from_costin/borrowed_submission_summary.json`

## Submission Layout

Submissions now follow a folder-based layout:

```text
submission_name/
  submission.csv
```

The scripts in this repo were updated to read and write submissions using that
layout.
