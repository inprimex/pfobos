# Changelog

All notable changes to `pfobos` will be documented in this file.

The format is loosely based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/) — minor
version bumps signal breaking-API changes pre-1.0.

## [0.4.0] — 2026-06-18

### Changed (BREAKING — behavior)

- `FobosSDR.read_rx_sync` now returns the FULL number of complex samples
  that libfobos delivered, not half. Pre-0.4.0 the wrapper extracted
  `actual_len * 4` bytes from libfobos's float buffer when `actual_len`
  is the IQ pair count delivered (per the libfobos source contract:
  `*actual_buf_length = actual / 4` from `fobos.c`). Since the float
  buffer holds 2 floats per IQ pair (= 8 bytes per pair), `actual_len * 4`
  extracted only the first half of every chunk.

  Concrete impact: sustained captures at 8 MSPS were delivering ~3.8 MSPS
  effective into the returned ndarray (47% of nominal). Post-0.4.0 they
  deliver the full nominal rate modulo libfobos's small overhead.

  **Existing captured `.iq` files remain valid** — they contain the first
  half of each chunk libfobos produced, in correct I/Q-interleaved order.
  Downstream analysis on them is statistically valid; it just had fewer
  samples available than the requested rate × duration would suggest.
  Re-capturing the corpus with the fix doubles the sample count per
  unit wall-clock.

### Migration

- Consumers of `FobosSDR.read_rx_sync` get ~2× the samples per call from
  the same `start_rx_sync(buf_length)` configuration. No API change;
  buffer sizes and the complex64 dtype are unchanged.
- Unit tests that assert specific sample counts per `read_rx_sync` call
  need expected-count values doubled.
- `watchtower-edge` `FobosIQSource.read_chunk` inherits the doubled
  sample count and should bump its `pfobos` pin from `^0.3.0` to
  `^0.4.0`.

### Why 0.4.0 (not 0.3.1)

API surface is unchanged, but the wrapper's observable contract changes
from "delivers ~half the IQ samples libfobos produced" to "delivers all
IQ samples". Treated as breaking-behavior so consumers opt in via the
minor-version bump rather than silently inherit via a patch upgrade.

### Reference

- libfobos source: rigexpert/libfobos master post-2.4.0,
  `fobos/fobos.c::fobos_rx_read_sync` and `::fobos_rx_start_sync`
- pfobos diagnostic that surfaced the bug: bus message
  `20260618T085536` (47% effective rate, independent of edge container)
- Contract-change notice to embedded-agent: bus message
  `20260618T090654` (pre-PR heads-up per CLAUDE.md spec change rules)

## [0.3.0] — 2026-06-17

### Changed (BREAKING)

- `FobosSDR.read_rx_sync` no longer calls `stop_rx_sync()` in its error
  path. On `FobosException`, sync mode remains active and the caller
  decides whether to retry, restart (`stop_rx_sync` + `start_rx_sync`
  explicitly), close, or escalate. This restores the documented
  start/stop lifecycle contract — `start_rx_sync` and `stop_rx_sync` are
  paired calls owned by the caller, not implicit side effects of error
  handling. See OpenSpec change
  `pfobos-read-rx-sync-no-auto-stop-on-error` in `watchtower-specs` for
  the full rationale + the cross-repo migration plan.

- `FobosSDR.read_rx_sync` now re-raises the original `FobosException`
  with its libfobos error code intact, rather than wrapping it in a new
  `FobosException(-1, ...)`. Consumers that inspect `.code` (e.g. to
  branch on `FobosError.LIBUSB == -9` for transient USB hiccups) now
  see the real code from the C call.

### Migration

If you catch `FobosException` from `read_rx_sync` in a sustained read
loop, your existing code is now CORRECT and worked-around bugs latent
since 0.2.0 silently break on the second iteration. No change required.

If you depend on the pre-0.3.0 auto-stop side effect (e.g. you skip
calling `stop_rx_sync` explicitly because you assume the wrapper did it
for you on error), update your error path to call `stop_rx_sync`
explicitly OR use the recommended pattern in the class docstring:

```python
sdr.start_rx_sync(buf_length)
try:
    while keep_going:
        try:
            iq = sdr.read_rx_sync()
        except FobosException as e:
            if e.code == FobosError.LIBUSB:
                continue  # transient; retry
            raise
        process(iq)
finally:
    sdr.stop_rx_sync()
```

### Internal

- In-repo consumers (`rtanalyzer/`, `fmreceiver/`, `webui/sdr_worker.py`)
  audited; all four were latently broken under the 0.2.0 auto-stop
  semantics and are now correct under the 0.3.0 contract without code
  changes.

- `bench/observation_mode_research/retune_classic.py` keeps the
  `_ensure_sync_started` helper as defensive code; its docstring is
  updated to document that it's no longer required (pfobos 0.3.0 makes
  the auto-stop trap go away) but remains for belt-and-suspenders.

### Cross-repo

- `watchtower-edge` `FobosIQSource(IQSource)` migration is owned by
  embedded-agent under §3 of the same OpenSpec change. No code lands in
  `watchtower-edge` against the new contract until 0.3.0 is published.

## [0.2.0] — 2026-06-10

### Changed (BREAKING)

- Importable package renamed from `shared.fwrapper` to `pfobos`.
  `from pfobos import FobosSDR, FobosException, FobosError`.

- Core dependencies slimmed from `numpy + scipy + matplotlib + cffi +
  pandas + tabulate` to `numpy + cffi`. Optional extras `[apps]`,
  `[audio]`, `[webui]` for the analyzer, fm receiver, and webui
  respectively.

- Wheel ships only `pfobos/`; `rtanalyzer/`, `fmreceiver/`, `webui/`
  stay in the repo as dev tooling, not installable packages.

See OpenSpec change `pfobos-as-primary-sdr-backend` in
`watchtower-specs` (archived 2026-06-16) for the rationale.

## [0.1.0]

Initial release.
