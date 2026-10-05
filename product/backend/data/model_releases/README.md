# Released model weights

These are the model artifacts produced or preserved during the 2026-10-05 experiment pass.

- `ampdnet_state_grid_1_epoch2.pt`: official AMPDNet architecture, state-grid site 1, seed 42, two completed CPU epochs. The test set was not evaluated after the run was stopped; do not report this checkpoint as passing the 96% gate.
- `fusion_model30-15.pth`: existing MobileNetV3 + ConvLSTM + PV attention + gate-fusion checkpoint. Its original training protocol is recorded in `EXPERIMENT_REPORT.md`; it has not been certified against the fixed protocol.
- `fusion_model.pth`: existing fusion checkpoint with the same compatibility limitation.

Verify SHA256 values in `MANIFEST.json` before loading a checkpoint.
