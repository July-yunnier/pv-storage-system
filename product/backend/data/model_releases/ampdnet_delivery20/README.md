# AMPDNet 20-epoch delivery release

Fixed selection: Solar1/2/6/7/8 × 15min/1h/4h, full, all-season, chronological 60/20/20, seed42. Every run completed20 epochs; its saved checkpoint is selected by minimum validation MAE, not necessarily epoch20. No ranking or selection across test scores.

Original selected weights are in `evidence/*.pth`; deployable graphs are `*.torchscript.pt`; paired JSON files contain exact architecture/scalers, output MW-to-kW conversion, source hashes and selected epoch. Runtime needs only PyTorch and NumPy, not the upstream source repository or timm/einops. The original training library is cited in every manifest; its local pinned checkout has no LICENSE. No new upstream redistribution license is asserted.

`verification.json` compares every saved test-window output from all15 weights against delivered arrays on CPU, and every serialized graph against CPU eager output. CPU/GPU results may differ numerically; maximum errors and tolerances are recorded without rounding them to zero. Original arrays/configs with machine paths remain local and are excluded from upload kit.

Original data compresses nighttime into daylight sample steps. Historical test metrics are not a new continuous-clock 60/10/30 experiment. Input uses48/96/192 daylight TSI/DNI/GHI/Temp samples, not historical power. Historical multistep boundary overlap, Solar2 threshold filtering and Solar8 interpolated labels are disclosed in each manifest. Runtime preserves input sampling and refuses output targets crossing18:45; do not market it as overnight, day-ahead or universally generalizing performance. It does not replace the system's auto default.

This release supersedes the interrupted epoch2 *integration draft*, not the historical epoch2 experiment records. See `product/docs/14-ampdnet-delivery20.md`.
