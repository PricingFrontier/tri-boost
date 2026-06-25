# Goal-optimization loop — experiment log

Approach (per direction 2026-06-25): **NOT per-dataset hyperparameter tuning** — that overfits the
benchmark and doesn't generalize. Instead, **diagnose why a rival wins, identify the algorithmic
technique behind it, and implement that technique as a genuine library feature** that helps broadly.
Every change keeps G0 (exact ≤3rd-order purified decomposition, all 5 I2 gates, `mode=Exact`) — so
leaf-wise/asymmetric growth and order>3 are off the table; value-level leaf methods, histogram/engine
engineering, sampling, bagging, categorical encoding, split-finding are fair game.

Metrics measured vs the **frozen rival baseline** (`.fair_cache.json`), across the whole suite — never
tuned to a single dataset. RMSE-log↓ (regression), ROC-AUC↑ (classification).

## Rival wins → technique gaps (the map the loop works through)
- **G1 (EBM)** beats tri at order-1/2 (main-effect shape quality). EBM technique: cyclic/round-robin
  per-feature boosting + heavy outer bagging + careful binning.
- **G2 (XGBoost/LightGBM @ d3)** beat tri on diamonds (fine-continuous; partly structural — leaf-wise is
  more expressive, OFF-TABLE) and amazon (all-categorical). Technique: native optimal categorical split
  (sort categories by gradient, split into 2 groups), quantile split candidates.
- **G3 (CatBoost d3-ctr1)** beats tri only on amazon. Technique: ordered target statistics / ordered boosting.
- **G5 (speed)** LightGBM 7–27× faster training. Technique: histogram subtraction, quantized hist, GOSS/EFB.

Note: **amazon (all-categorical) loses BOTH G2 and G3** → better categorical handling is the highest-leverage
single technique (closes two goals at once). A `rival-technique-roadmap` workflow is producing the prioritized,
G0-verified plan; entries below are filled as each technique is implemented + measured + committed.

## Aborted approach (recorded so it isn't retried)
- Per-dataset order-2 knob sweeps (lr/leaf_refine/l1/path_smooth/n_bags): confirmed single knobs *narrow*
  but don't *flip* diamonds/kick G1@2, and any "win" would be dataset-specific overfitting. Abandoned in
  favor of genuine technique adoption. (One real fact surfaced: miami G1@2 is a WIN under the committed
  gentle early stopping — a measurement correction, not a tuning gain.)

## Roadmap — rival techniques to adopt (from the rival-technique-roadmap workflow, all G0-verified)
ROI-ordered work queue; each is a genuine generalizing technique a rival uses that tri lacks.

| # | technique | source | goals | effort | G0 | status |
|---|---|---|---|---|---|---|
| 1 | Per-split categorical gradient re-sort (Fisher optimal split) | LightGBM | G2+G3(+G1) | L | conditional | **next** |
| 2 | Histogram subtraction (parent−child), QuantizedI32-only | LightGBM | G5 | M | safe | queued |
| 3 | Integer-space quantized hist scan (unlocks #2) + QHIST default | LightGBM | G5 | S | safe | queued |
| 4 | Cyclic/round-robin per-feature boosting | EBM | G1 | L | safe | queued |
| 5 | Automatic categorical CTR combination axes | CatBoost | G3+G2 | L | conditional | queued |
| 6 | FAST pairwise interaction detection (populate InteractionPolicy.groups) | EBM | G1 | L | safe | queued |
| 7 | Hessian-weighted quantile bin borders | XGBoost | G2 | M | safe | queued |

Rejected (already present / G0-incompatible): EFB (g0=no, low impact); missing default direction (tri HAS it);
ordered TS (tri HAS it, KFold OOF beats Ordered{1}); ordered boosting (weak, leakage already closed); GOSS
(subsumed by MVS + #2); colsample_bylevel, heavy-bagging-default, mains-first, low-card one-hot (composable add-ons).

Sequencing: #1 first (multi-goal: amazon on G2+G3 at once). Then #3+#2 together (G5, cheap, mutually dependent).
Then #4 (suite-wide G1 order-1). #5/#6 compose with #4 (EBM's mains-first recipe falls out). #7 last. Re-measure
G5 after #2/#3 before bothering with GOSS. Every step: live G0 `tables()` check (FAIR_G0=o3) green before commit.

## Implemented techniques (filled by the loop, with measured deltas vs frozen rivals)
_(none committed yet — starting #1)_
