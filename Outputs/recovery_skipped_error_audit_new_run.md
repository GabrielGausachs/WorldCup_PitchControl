# Recovery Dataset Audit (New Run): Skipped Rows and Error Frames

## Source
- CSV: `C:\Users\g4a4b\OneDrive - University of Twente\SportsAnalyticsProject\pff_data\dataset_passes_recovery\recovery_space_metrics_5s_r20_late_knockout.csv`

## Headline Stats
- Total rows: `125607`
- Total recoveries: `834`
- Status counts: `ok=92110`, `partial=33497`, `skipped=0`

## Skipped
- Skipped rows: `0`
- Recoveries with any skipped row: `0`
- Skipped rows per recovery distribution:
  - `0` skipped rows: `834` recoveries

## Errors
- Rows with non-null `error`: `33497`
- Recoveries with any `error`: `224`
- Rows with non-null `frame_error`: `15478`
- Recoveries with any `frame_error`: `224`

- `error` causes:
  - `one_or_more_frame_errors`: `33497`

- `frame_error` causes:
  - `invalid_ball_coordinates`: `15477`
  - `pc_frame_failed`: `1`

## Consecutive Error Runs
- Recoveries with >=1 second consecutive frame errors (>=30 frames): `174`
- Recoveries with >=50 frame-error rows: `141`

## 50+ Error Frame Clustering
- `single_block`: `140` recoveries
- `few_blocks_mostly_clustered`: `1` recoveries

## Window Size Sanity
- Recoveries with `>151` rows: `0`
- Recoveries with `=151` rows: `831`
- Recoveries with `<151` rows: `3`
- `window_n_frames` non-constant within a recovery: `0`
- `window_n_frames` mismatch with actual row count: `0`

## Game Concentration (recoveries with frame errors)
- Game `10517`: recoveries `162`, with frame errors `50`, frame-error rows `3346`
- Game `10511`: recoveries `106`, with frame errors `36`, frame-error rows `1967`
- Game `10514`: recoveries `105`, with frame errors `34`, frame-error rows `2449`
- Game `10516`: recoveries `92`, with frame errors `30`, frame-error rows `2642`
- Game `10513`: recoveries `94`, with frame errors `21`, frame-error rows `1426`
- Game `10512`: recoveries `102`, with frame errors `18`, frame-error rows `1122`
- Game `10515`: recoveries `93`, with frame errors `18`, frame-error rows `1425`
- Game `10510`: recoveries `80`, with frame errors `17`, frame-error rows `1101`

## Metric Ranges
- `pc_mean`: non-null `110129`, min `0.4193769554464089`, max `0.5793105959892273`
- `pv_mean`: non-null `110129`, min `0.3708059629600946`, max `0.5105015995270024`
- `sq_mean`: non-null `110129`, min `0.1411014528884162`, max `0.2911906592718745`
- `pc_mean_r20`: non-null `110129`, min `0.2253627156382364`, max `0.7746372843617635`
- `pv_mean_r20`: non-null `110129`, min `0.1944873365567013`, max `0.9225925144390204`
- `sq_mean_r20`: non-null `110129`, min `0.0442070640793804`, max `0.5622421929562608`
