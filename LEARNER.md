# Learner profile

Read at the start of every session. Keep it short and current; replace stale lines instead of appending.

## Background
- Has substantial existing knowledge; wants the opening refresher short and high level.
- Details to be filled in from `refresher-diagnostic`.

## Setup
- Works in Google Colab; local machine runs Claude Code only (Python available, no NumPy or PyTorch installed).
- Compute: free T4 by default, sometimes larger. Notebooks default to `SCALE = "t4"`.
- Python only.

## Strong areas
- (from diagnostic and sessions)

## Weak spots to revisit
- (from diagnostic and sessions)

## How to run sessions
- (pace, scaffold level adjustments, preferences that emerge)

## Colab and GitHub quirks found
- Round trip confirmed 2026-11-18: clone cell works, outputs survive "Save a copy in GitHub", solution hash matches. Colab gave CPU runtime (python 3.13, torch cpu).
- Local pushes: GitHub fine-grained PAT in git-ignored `token.txt`, used by `git_sync.py`. SSH key not registered on GitHub.
