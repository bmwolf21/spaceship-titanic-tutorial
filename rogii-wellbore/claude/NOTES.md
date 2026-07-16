# Claude resume note - ROGII Wellbore Geology Prediction

Status as of 2026-07-16. Read this first when picking the competition back up, then
read `../COLLABORATION.md` (shared with Codex) and its handoff log.

## Where we are
- **Task:** predict `TVT` (ft) for the masked toe of each horizontal well.
  Metric: plain RMSE. Crux: **group-CV by well** (`../shared/folds.csv`); the
  public LB is leaked/meaningless (3 test wells are also train wells). Full spec:
  `../shared/DATA_SPEC.md`.
- **My best model: `src/03_gr_correlation_model.py`, OOF RMSE 15.249 ft.**
  GR->TVT inversion (nearest match vs type-log and the lateral's own pre-PS
  GR-vs-TVT) + LightGBM on `dTVT = TVT - TVT_PS`. `oof.csv`/`test_pred.csv`
  currently hold this model's output.
- **Codex** (independent, `../codex/`): **14.14 ft** standalone (geometry anchor +
  ridge + offset priors + residual correction). Stronger solo than me and still
  improving.
- **Blend: `../shared/blend.py` -> 13.873 ft** (weight ~0.31 me / 0.69 Codex),
  beats both. Submission: `../outputs/submissions/blend_claude_codex_20260716.csv`.
  Rerun `blend.py` after either agent refreshes an `oof.csv`.
- `claude/oof.csv` = model 03, backed up to `claude/oof_03_best.csv`.

## What I learned (don't re-derive)
- "Flat" (toe TVT = TVT at PS) = 15.91 ft. That's the bar.
- Geometry is useless: `corr(dTVT, dZ) = -0.13`, extrapolating slope = 117 ft.
  Wells are geosteered, so Z swings ~88 ft while TVT moves only ~11 ft. **GR is the
  only signal.** dTVT is small (mean|dTVT| 11 ft, p95 32 ft).
- **Four FAILED standalone attempts (all kept in `src/` for the record; none beat
  03=15.249):** `05` GR-context+heavy-smoothing (15.76); `06` DP path correlation
  (21.7 - GR too ambiguous for a path to trust); `07` GBM stack of the two OOFs
  (14.54, worse than the weighted blend - so stacking is OUT, weighted blend wins);
  `08` offset-well prior (15.41 - helped fold 0, misled folds 1/3/4, and converges
  toward Codex's line so it blends worse). **My GR standalone is plateaued at 15.25.**

## Next levers (honest assessment)
My GR-inversion approach has plateaued (4 failed attempts). Realistic options:
1. **Accept the diverse-model role.** My 03 contributes ~0.31 weight to the blend;
   the blend improves as Codex improves. That may be my best contribution.
2. **A genuinely different model family** I have not tried (e.g. a 1-D CNN over the
   GR sequence, if a DL stack is available) - only worth it if it's truly diverse
   from both 03 and Codex.
3. **Codex's line is the stronger horse** (14.14, improving). Higher joint EV may
   be helping review/improve Codex than squeezing my standalone.
4. Fold 3 stays hardest for both; Codex's diagnostic: error tracks true |dTVT|
   (large excursions), and neighbor priors did NOT fix it (my 08 made fold 3
   worse). Genuinely hard wells.

## How to run
```
python claude/src/03_gr_correlation_model.py   # my best model -> oof.csv, test_pred.csv, submission
python shared/blend.py                         # blend with Codex -> blended submission
```
Data lives in `data/raw/` (git-ignored; re-download: `kaggle competitions download
-c rogii-wellbore-geology-prediction -p data/raw`, extract CSVs, skip PNGs).

## Git etiquette (two agents, one repo)
Commit only your own subdir (`git add rogii-wellbore/claude ... COLLABORATION.md`),
never `git add -A` (it sweeps Codex's live work). `oof.csv` is git-ignored (150 MB,
over GitHub's limit). Commit, then `git pull --rebase`, then push.
