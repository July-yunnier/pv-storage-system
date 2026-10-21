# Original load model GEFCom adapter

This is an additional offline pre-experiment, not a replacement of existing
GEFCom results or online factory weights. The main experiment has not started.
PV models and dispatch are neither trained nor changed.

`load_gefcom_adapter.py` inherits `load_advanced.EliteEnergyTaskNet` modules:
two-layer LSTM (hidden=128, dropout=0.3), Bahdanau attention, 8-dimensional MES
embedding, target decoder (GELU, dropout=0.2), and total/workshop heads.
It overrides the fixed 96-slot forward contract to accept hourly 1h/4h targets.
All MES IDs are unknown (0), with a frozen zero embedding. The workshop head is
frozen because no real workshop labels exist; auxiliary/consistency losses are
disabled. Only total-load final-horizon MSE is trained. No simulated labels are
created. Future decoder features contain known calendar features only.

The history window is 48h to match the existing comparison, not the original
factory training's 30 days. GEFCom weather channels replace the original
temperature/humidity schema. This is a documented architecture adaptation,
NOT an exactly unchanged model with a pure dataset swap. Its weights cannot
be loaded through the original 15-minute/96-slot online factory API.

Training uses the previous runner's source reconstruction and time-window
builder, verifies identical train/validation/test timestamps and split labels,
normalizes on train only, and uses seed=42 with chronological batches.
Default budget=120 epochs, patience=10, learning rate=0.001, batch size=512.
Early stopping uses validation MSE only. Both horizon selections freeze before
test inference. Budget exhaustion is reported as not converged.

```bash
python scripts/run_original_load_gefcom.py \
  --data /path/to/gefcom2014_benchmark \
  --previous /path/to/gefcom_preexperiment_convergence_v3 \
  --output /path/to/new_independent_output
python scripts/plot_original_load_comparison.py \
  --previous /path/to/gefcom_preexperiment_convergence_v3 \
  --output /path/to/new_independent_output
```

The point model's P10/P90 use the second validation half's residual quantiles.
This is post-calibration, not a native Pinball head; the point estimate is not
guaranteed to be a conditional median. Validation was also used for checkpoint
selection, so calibration is not independent and no coverage guarantee exists.
Native LSTM/TCN quantile heads must be labelled separately in probability
comparisons. Historical test exposure means all these results remain exploratory.

Load Cap is the historical training maximum, a statistical reporting proxy,
not physical capacity. Keep original source units; do not present regional load
as a real factory's kW profile. No native 15-minute labels, MES effect, workshop
accuracy, factory generalization, or closed-loop savings are established here.

Sources: original in-repository EliteEnergyTaskNet; Bahdanau et al. (2015),
"Neural Machine Translation by Jointly Learning to Align and Translate";
GEFCom2014 public load track supplied by the user, Hong et al. (2016),
"Probabilistic energy forecasting: Global Energy Forecasting Competition 2014
and beyond". PyTorch (BSD-3-Clause), NumPy (BSD-3-Clause), pandas (BSD-3-Clause),
Matplotlib (PSF-based license). GEFCom data redistribution rights are not inferred
from public availability; raw data is not uploaded by this experiment.
