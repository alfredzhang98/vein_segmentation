# Clarius Cast SDK

The vendor's native API for the Clarius probe: shared libraries, C headers, the upstream
`cast` submodule, and Clarius's own example programs.

> 中文版：[README_CN.md](README_CN.md)

**You only need this for native-API development and data acquisition.** Training,
evaluation and inference never touch it — if you are only training a model, ignore this
directory entirely.

## Layout

| Path | What it is |
|---|---|
| `lib/libcast.so`, `lib/pyclariuscast.so` | Linux shared libraries |
| `lib/cast.dll`, `lib/cast.lib`, `lib/pyclariuscast.pyd` | Windows equivalents |
| `lib/cast/*.h` | C headers — `cast.h`, `cast_def.h`, `cast_cb.h`, `cast_export.h` |
| `cast/` | Upstream [clariusdev/cast](https://github.com/clariusdev/cast) as a git submodule |
| `examples/` | Clarius's own sample programs (see below) |

The submodule is not cloned by default:

```bash
git submodule update --init data/clarius_sdk/cast
```

## Who loads what

[`data/collection/collect_clarius.py`](../collection/collect_clarius.py) is the only file
in this project that touches the SDK. It loads `lib/libcast.so` and
`lib/pyclariuscast.so` by explicit path before importing `pyclariuscast`, so the
libraries do not need to be on `LD_LIBRARY_PATH`. If you move `lib/`, fix `_lib_dir`
there.

## Examples

These are Clarius's, kept verbatim as reference. They are **not** wired into this
project's pipeline.

| File | What it does |
|---|---|
| `examples/pycaster.py` | Minimal Cast API client. Run with `python pycaster.py --ip <probe-ip>` |
| `examples/pysidecaster.py` | PySide6 GUI over the Cast API |
| `examples/pysidecaster_fps.py` | Same, with a frame-rate readout |
| `examples/pyimu.py` | GUI showing the probe's IMU stream. **Known bug: yaw is inaccurate** |
| `examples/DICOMserver.py` | Runs a DICOM server locally. Reachable only when the probe is on a real Wi-Fi network, not a hotspot. Writes studies to `dicom/` when the scan is ended in the Clarius app |
| `examples/3Dmodel/` | `scanner.obj` / `scanner.mtl` — probe mesh used by `pyimu.py` |

## Requirements

A Clarius probe, its research/Cast API credentials, and a network path from this machine
to the probe. None of that is needed to train or run the model.
