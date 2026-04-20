# ERA5 Data Selection for Soil Moisture Fine-Tuning

## Task

Fine-tune Aurora to predict **volumetric soil water layer 1** (0–7 cm depth), a variable the pretrained model was never trained on. This extends Aurora from purely atmospheric prediction into land surface processes.

## Why Soil Moisture?

Soil moisture is a stronger fine-tuning use case than precipitation:

- **Genuinely new state variable**: Aurora's pretraining includes no land surface variables beyond static fields (orography, land-sea mask, soil type). Soil moisture has its own dynamics (infiltration, drainage, freeze/thaw) not captured by atmospheric variables alone.
- **Feeds back into atmosphere**: Evapotranspiration from soil affects humidity and temperature, so teaching Aurora about soil moisture could improve its atmospheric predictions too.
- **Instantaneous analysis variable**: Unlike precipitation (which is accumulated over forecast windows), soil moisture is an instantaneous state in ERA5, fitting cleanly into Aurora's state-to-state prediction framework.
- **Precipitation was considered and deprioritized**: Precipitation is largely a *diagnostic* of the atmospheric state Aurora already predicts (temperature, humidity, wind profiles). It's also spatially spiky, stochastic at 25 km resolution, and accumulated rather than instantaneous. It's less compelling as a "new capability" for the model.

## Why Layer 1 Only?

ERA5 provides soil moisture at 4 depth layers:

| Layer | Depth | Characteristic timescale |
|-------|-------|--------------------------|
| Layer 1 | 0–7 cm | Hours–days (responds directly to weather) |
| Layer 2 | 7–28 cm | Days–weeks |
| Layer 3 | 28–100 cm | Weeks–months |
| Layer 4 | 100–289 cm | Months–seasons |

Layer 1 is the most responsive to weather on Aurora's 6-hour prediction timescale. Deeper layers barely change in a 6-hour window — training on near-zero deltas just teaches the model to predict persistence.

## Variables

### Design Principle

We only include variables that represent **independent state** — quantities with their own dynamics that cannot be derived from a single atmospheric snapshot. Variables that are diagnostic of the atmospheric state (derivable from temperature, humidity, wind, and pressure profiles) are excluded to prevent the model from learning shortcuts instead of the underlying physics.

### Aurora Pretraining Variables (baseline)

These are the variables Aurora was pretrained on (Table C3 in the paper). We continue to include them.

**Surface-level** (from `reanalysis-era5-single-levels`):

| CDS Variable | Short Name | Reason for inclusion |
|---|---|---|
| 2m temperature | t2m | Core atmospheric state variable from Aurora pretraining. Near-surface temperature drives evaporation from soil. |
| 10m u-component of wind | u10 | Core atmospheric state variable from Aurora pretraining. Wind speed increases evaporation by removing moist air from the surface. |
| 10m v-component of wind | v10 | Core atmospheric state variable from Aurora pretraining. Together with u10, determines wind speed and direction. |
| Mean sea level pressure | msl | Core atmospheric state variable from Aurora pretraining. Encodes synoptic-scale weather patterns (highs/lows) that drive precipitation and frontal activity. |

**Static** (from `reanalysis-era5-single-levels`, downloaded once):

| CDS Variable | Short Name | Reason for inclusion |
|---|---|---|
| Geopotential | z | Orography (terrain height). Determines where orographic precipitation occurs and how water flows downhill. |
| Land-sea mask | lsm | Distinguishes land from ocean. Soil moisture is only meaningful over land. |
| Soil type | slt | Determines soil hydraulic properties (permeability, water-holding capacity). Sandy soils drain fast; clay soils retain water. |

**Atmospheric** (from `reanalysis-era5-pressure-levels`, 13 levels: 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000 hPa):

| CDS Variable | Short Name | Reason for inclusion |
|---|---|---|
| Temperature | t | Core Aurora variable. Vertical thermal structure determines atmospheric stability, condensation levels, and whether precipitation falls as rain or snow. |
| U-component of wind | u | Core Aurora variable. Atmospheric dynamics and moisture transport. |
| V-component of wind | v | Core Aurora variable. Together with u, determines large-scale moisture advection and frontal movement. |
| Specific humidity | q | Core Aurora variable. Moisture at each pressure level. Vertical integral gives total atmospheric moisture; saturation at any level triggers condensation. |
| Geopotential | z | Core Aurora variable. Heights of pressure surfaces; encodes the large-scale flow patterns. |

### New Variables for Soil Moisture

These three variables are genuine independent state — each has its own dynamics and memory that cannot be inferred from a single atmospheric snapshot.

| CDS Variable | Short Name | Reason for inclusion |
|---|---|---|
| Volumetric soil water layer 1 | swvl1 | **Primary prediction target.** Water content in the top 0–7 cm of soil. Has its own dynamics (infiltration, drainage, capillary rise) that persist independently of the atmospheric state. Not derivable from atmospheric variables. |
| Soil temperature level 1 | stl1 | **Freeze/thaw gate.** Frozen soil blocks water infiltration entirely, determining whether precipitation enters the soil or runs off. Soil has thermal inertia independent of air temperature — soil temperature lags and differs from 2m air temperature due to ground heat capacity. |
| Snow depth | sd | **Snowmelt water source.** Water stored as snowpack accumulates over days to weeks and is not derivable from a single atmospheric snapshot. A warm spell can release the accumulated snowpack into the soil, causing the largest soil moisture changes of the year. Without snow state, the model cannot anticipate these events. |

### Variables Considered and Excluded

#### Excluded: derivable from atmospheric profiles (redundant)

| CDS Variable | Short Name | Reason for exclusion |
|---|---|---|
| 2m dewpoint temperature | d2m | Humidity at 2m; controls evaporation rate via moisture deficit (T − Td). Excluded because it is derivable from temperature and specific humidity (q) at the lowest pressure levels, which Aurora already has. |
| Total column water vapour | tcwv | Total atmospheric moisture integrated vertically. Excluded because it is literally the vertical integral of specific humidity (q) across the 13 pressure levels Aurora already has. |
| Surface pressure | sp | Actual pressure at the surface (more accurate than MSL over mountains). Excluded because the MSL-to-surface correction is a function of orography, which Aurora has as a static variable. |
| Skin temperature | skt | Temperature of the Earth's surface itself. Excluded because it is closely correlated with 2m temperature (already included) and soil temperature level 1 (newly included) captures the relevant thermal state for freeze/thaw. |
| Total cloud cover | tcc | Fraction of sky covered by clouds; modulates solar radiation and thus evaporation. Excluded because cloud cover is diagnostic of the atmospheric humidity and temperature profiles Aurora already has. |

#### Excluded: flux/rate variables (shortcut risk)

These variables directly encode the water and energy fluxes that determine soil moisture change. Including them as inputs would let the model learn a trivial mapping (Δsoil_moisture ≈ precipitation − evaporation − runoff + snowmelt) instead of learning the atmospheric-to-land dynamics from the atmospheric state. During autoregressive rollout, any error in predicted rates would cascade directly into soil moisture because the model never learned the robust underlying physics.

| CDS Variable | Short Name | Reason for exclusion |
|---|---|---|
| Mean total precipitation rate | mtpr | Diagnostic of atmospheric temperature, humidity, and wind profiles. |
| Mean evaporation rate | mer | Derivable from temperature, wind, humidity deficit, and radiation (Penman-Monteith). |
| Mean snowmelt rate | msmr | Derivable from temperature, radiation, and snow state. |
| Mean surface runoff rate | msror | Derivable from precipitation rate, soil moisture, soil type, and terrain. |
| Mean surface downward short-wave radiation flux | msdwswrf | Derivable from solar geometry (latitude + time of year), cloud cover, and atmospheric composition. |

## Data Source

All surface/static variables come from `reanalysis-era5-single-levels`.
All atmospheric variables come from `reanalysis-era5-pressure-levels`.
Product type: **reanalysis** (deterministic high-resolution, 0.25° / ~31 km).

These are the same datasets used in the Aurora paper's pretraining (Table C3).

- Surface/static: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels
- Pressure levels: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels

## Download Strategy

**Monthly bulk requests** instead of per-day (reduces API calls from 730 to 24+1):
- 12 months × 2 requests each (surface + atmospheric) = 24 API calls
- 1 static variable request
- All 24 hourly time steps per day (not just 6-hourly)
- ~8,500 valid 6-hour-apart training pairs per year vs ~1,095 with 6-hourly only

**Why all 24 hours**: The Aurora paper used all hourly ERA5 data for pretraining (368k time steps = hourly over 1979–2020). Downloading all hours gives 6× more starting points for 6-hour prediction pairs and matches the paper's training data diversity.

**File layout**: Monthly NetCDF files (`2025-01-surface.nc`, `2025-01-atmospheric.nc`, `static.nc`).

## Summary

| Category | Variables | Count |
|---|---|---|
| Surface (existing) | t2m, u10, v10, msl | 4 |
| Static (existing) | z, lsm, slt | 3 |
| Atmospheric (existing) | t, u, v, q, z × 13 levels | 5 (65 fields) |
| New state variables | swvl1, stl1, sd | 3 |
| **Total surface variables** | | **7** |
| **Total atmospheric variables** | | **5 × 13 levels** |
