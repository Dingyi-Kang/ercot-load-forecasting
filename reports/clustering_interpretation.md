# Interpretation (descriptive, not causal)

- **Rule-selected k=2** for both representations: silhouette favors small k; k=3–4 are also stable.
- **Raw MW clusters are not stationary**: the "high-load" raw cluster grows from 36% of fit-period
  days to 78% of 2024–2025 days. ERCOT annual mean load grew ~37% from 2017 (40.8 GW) to 2025 (55.7 GW), so raw-magnitude clusters
  mostly track growth and season, not day type. A clustering used over years needs a level-adjusted
  or shape representation (or periodic refits).
- **Shape clusters are stable** across periods (k=2: 37% → 40%) and largely independent of raw
  clusters (ARI 0.12): magnitude and shape are different information.
- **Exploratory shape k=4**: (0) winter evening-peak days, (1) winter **morning-peak** days (peak
  HE08, minimum HE17; cold mornings), (2) summer afternoon-peak days (largest peak-to-mean 1.24),
  (3) shoulder-season evening-peak days.
- **Where the forecast struggles (retrospective only)**: on 2024–2025 validation, morning-peak
  winter days have the highest error (MAPE 3.6%, bias +306 MW), summer afternoon-peak days the
  lowest (1.9%). Plausible but unverified explanation: cold-morning heating load is sensitive to
  small temperature-forecast errors and solar/heat-pump behavior.
- **Testable next step (not done)**: predict the regime from issue-time information (forecast
  morning minimum temperature, calendar) and test a regime-aware model against the global model
  under the same protocol. Realized labels must never be used as inputs.
