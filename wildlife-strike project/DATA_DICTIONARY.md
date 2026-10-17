# Modeling data dictionary

| Field | Definition |
|---|---|
| strikes_prior_1m | Reported strike count in prior calendar month |
| strikes_prior_3m | Reported strike sum in prior 3 months |
| strikes_prior_12m | Reported strike sum in prior 12 months |
| damages_prior_1m | Reports indicating damage in prior calendar month |
| damages_prior_3m | Reports indicating damage in prior 3 months |
| damages_prior_12m | Reports indicating damage in prior 12 months |
| sin_month and cos_month | Seasonal encoding of target calendar month |
| AIRPORT_ID | Airport identity known before target month |
| y | At least one report indicating damage during target month |
| date | Target month start date |

Predictors are lagged by incident date, assuming report availability. No current-month species, damage, cost, or outcome fields enter the model. Counts of damage refer to reports, not necessarily independently verified incidents.
