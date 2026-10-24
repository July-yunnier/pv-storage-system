# LLM strategic advisor (experimental)

`LLMStrategicAdvisor.advise(daily_forecast, historical_summary)` accepts JSON objects
or JSON strings and returns a JSON-serializable policy:

```json
{"soc_min_safe": 0.15, "peak_discharge_offset": 0}
```

- `soc_min_safe`: bounded to `[0.10, 0.95]`, forecast-side discharge reserve.
- `peak_discharge_offset`: integer signed 15-minute slots, bounded to `[-4, 4]`.
  Positive delays the peak discharge window; slots never wrap into another day.
- `last_diagnostics`: mode/status/failure type for policy audit; no API keys or raw
  error strings. Transport/parse/invalid-field failures fall back to 10% reserve,
  zero offset. Extra response fields do not reach execution.
- The LLM never controls hard SOC/power/efficiency limits. `execute` is unchanged.
  Forecast error can cause actual SOC to cross the advisory reserve, but physical
  SOC clipping stays at 10%..95%.

From `backend/`, install dependencies (`requirements.txt`, plus numpy/scipy for
the experiment runner). Offline inference does not need torch or a GPU.

```bash
python -m pip install -r requirements.txt numpy scipy
python scripts/run_official_experiment.py --agent llm --mock-offline
```

Mock mode is the default and never constructs an API client. It returns a fixed
15% reserve and zero offset; this is a deterministic plumbing experiment, not an
LLM economic-performance benchmark. Outputs default to
`reports/official_experiment/llm_mock/` to preserve the earlier experiment.

For a real API, copy `backend/.env.example` to `backend/.env`, then configure the
three environment variables. Process environment takes precedence over `.env`.
The template's endpoint/model/placeholder defaults are local Ollama development
settings only. Hosted providers require their own endpoint/model/key. Python
source contains no endpoint/model/key literals. Never commit `backend/.env`.

```bash
python scripts/run_official_experiment.py --agent llm --no-mock-offline
# Optional explicit destination:
python scripts/run_official_experiment.py --agent llm --mock-offline --output-dir /path/to/run
```

API runs default to `reports/official_experiment/llm_api/`. The experiment makes
485 daily advisor calls across P0/P2/P4; P1/P3 reuse P0 execution for settlement
and same-window LP comparison. The OpenAI-compatible client uses a bounded 20s
request timeout and no automatic retries. API nondeterminism remains possible
at temperature zero; issued forecasts, history, policies and fallback metadata
are archived **before actual settlement** in `agent_policy_archive.jsonl`.

P0 input forecasts use measurements strictly before D. P2 uses archived PJM
price forecasts and actual prices only for settlement; PV/load is a disclosed
synthetic source-day alignment. Existing TOU thresholds remain in force, so P2
can be a degenerate equal-cost comparison at PJM price levels. P4 has a separate
365-day synthetic cohort with causal 30-day warmup and continuous SOC; do not
compare its E_LLM number to the legacy P4 cohort's total. P0-A is untouched.

`p0_replay/summary.json` contains E_LLM totals, paired daily bootstrap comparisons,
zero-violation count and P0–P4 result links. Each stage's summary also includes
E_LLM, and `run_meta.json` includes input/implementation/archive hashes and mode.

Tests:

```bash
python -m pytest -q
```
