# Data folder

Full instructions are in docs/TRAINING.md.

- `raw/` holds downloads exactly as received. Git ignores it, because each source
  keeps its own licence.
- `observations/` holds what training reads, one folder per source. Each folder has
  `subjects.csv`, `events.csv`, and `observations.csv`.

| Folder | Created by | Committed to git |
|---|---|---|
| `observations/literature/` | Typed in from published sources, listed in `notes` | Yes |
| `observations/custom/` | You | Your choice |
| `observations/pkdb/` | `humanmock data pkdb` | No |
| `observations/synthetic/` | `humanmock data synthetic`, and never treated as evidence | No |
