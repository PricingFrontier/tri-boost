# Score Rating Tables in SQL

This recipe shows how to deploy an exact `RatingExport` as ordinary SQL lookup
tables. It is for filing, reporting, and simple warehouse scoring. The serialized
`Model` remains the canonical deployment artifact when Rust or Python scoring is
available.

## Export

Export rating tables from the fitted model and the same audited serve matrix used
for explanations:

```python
from tri_boost.sklearn import TriBoostRegressor

est = TriBoostRegressor(n_trees=200, learning_rate=0.05).fit(X_train, y_train)
rating_json = est.tables(X_audit, ref_measure="uniform")
```

Rust callers can use the same surface through
`TableBank::to_rating_export(model.link, &model.mode, &model.schema, basis)`.
The export is allowed only when `mode == "Exact"`. The exactness firewall rejects
approximate models because a rating table must reconstruct the model score without
coarsening, truncation, or sampled attribution.

## Pure vs Rating View

The default export is the PURE fANOVA form:

```text
raw_score = f0 + sum(table_value)
```

Each table is zero-mean under the export reference measure. A pure table is the
right artifact for audit and variance/Sobol accounting, but its reference cells do
not generally read as neutral.

If a downstream rating sheet needs selected reference cells to read as `0.0` in
score space, or `1.000` for a log-link relativity, pass a `RatingBasis`. The basis
subtracts the selected cell value from that table and folds the shift into `f0`;
`raw_score` is unchanged for every row.

## Cell Conventions

Each `RatingTable.axis` carries:

- `raw`: the raw feature id.
- `name`: the schema feature name.
- `borders`: finite upper borders from the merged grid.
- `cells`: total cell count, including the explicit missing cell.

Cell `0` is always missing. Finite cells start at `1`. To reproduce tri-boost
numeric binning in SQL, use `NULL` as missing and count the number of borders
strictly below the value:

```sql
case
  when age is null then 0
  when age <= 24.5 then 1
  when age <= 39.5 then 2
  when age <= 64.5 then 3
  else 4
end as age_cell
```

This is equivalent to the core rule `bin(v) = 1 + count(borders < v)`. Values
outside the fitted range clamp to the first or last finite cell. If a warehouse can
store IEEE NaN, normalize NaN to `NULL` before scoring so it routes to cell `0`.

Categorical target-statistic axes are already numeric model axes. For production
scoring, first apply the persisted categorical encoder from the serialized `Model`,
then apply the same border rule to the encoded value. The JSON rating export is
not a substitute for the encoder store.

## Table Layout

Load each exported table into a narrow lookup table. One-dimensional tables need
one cell key; two- and three-dimensional tables need one key per axis:

```sql
create table tri_boost_effect (
  table_id integer not null,
  axis0_raw integer not null,
  axis1_raw integer,
  axis2_raw integer,
  cell0 integer not null,
  cell1 integer,
  cell2 integer,
  value double precision not null,
  relativity double precision
);
```

For each dense `RatingTable`, flatten `values` in row-major order with the last
axis varying fastest. For a shape `[a, b, c]`, the row-major offset is:

```text
offset = ((cell0 * b) + cell1) * c + cell2
```

For a one- or two-dimensional table, omit the unused cell columns. Keep
`table_id` stable in Sobol-descending export order, or use the table's feature-set
metadata as the key.

## Scoring Query

A scoring query bins each feature once, joins the relevant effect rows, and sums
the score-space values:

```sql
with binned as (
  select
    id,
    case when age is null then 0
         when age <= 24.5 then 1
         when age <= 39.5 then 2
         when age <= 64.5 then 3
         else 4
    end as age_cell,
    case when vehicle_value is null then 0
         when vehicle_value <= 5000 then 1
         when vehicle_value <= 15000 then 2
         else 3
    end as vehicle_value_cell
  from scoring_rows
),
effects as (
  select b.id, e.value
  from binned b
  join tri_boost_effect e
    on e.table_id = 17
   and e.cell0 = b.age_cell
   and e.cell1 = b.vehicle_value_cell
  union all
  select b.id, e.value
  from binned b
  join tri_boost_effect e
    on e.table_id = 3
   and e.cell0 = b.age_cell
)
select
  b.id,
  :f0 + coalesce(sum(e.value), 0.0) as raw_score
from binned b
left join effects e on e.id = b.id
group by b.id;
```

For an identity-link model, `raw_score` is the prediction.

For a log-link model, the response-space prediction is:

```sql
exp(raw_score)
```

The exported `relativities` are `exp(value)` for display and rating-sheet
multiplication, but score-space summation is the numerically safest SQL contract:
sum `value`, then exponentiate once.

For a logit-link model, the probability is:

```sql
1.0 / (1.0 + exp(-raw_score))
```

If a model was trained with an exposure offset or another serving offset, add the
same offset to `raw_score` before applying the inverse link.

## Validation Checklist

Before using the SQL path in production:

- Compare SQL raw scores against Python `predict_raw` or Rust `Model::score_trees`
  on a fixed audit batch.
- Confirm the export says `mode == "Exact"`.
- Confirm missing values route to cell `0` and finite values route to cells
  `1..cells-1`.
- Use score-space `value` for reconstruction; treat `support`, `se_band`,
  `variance`, `sobol`, and `relativities` as display metadata.
- Keep the serialized `ModelDoc` with the SQL artifact so schema version, feature
  names, categorical encoders, link, and objective remain auditable.
