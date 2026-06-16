# Changelog

All notable changes to `pfobos` will be documented in this file.

The format is loosely based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/) — minor
version bumps signal breaking-API changes pre-1.0.

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
