# Clarius Cast SDK

Clarius 探头的厂商原生 API：动态库、C 头文件、上游 `cast` submodule，以及 Clarius 官方的示例程序。

> English: [README.md](README.md)

**只有做原生 API 开发和数据采集时才需要它。** 训练、评估、推理完全不碰这个目录 —— 如果你只是要训模型，可以彻底忽略它。

## 目录结构

| 路径 | 内容 |
|---|---|
| `lib/libcast.so`、`lib/pyclariuscast.so` | Linux 动态库 |
| `lib/cast.dll`、`lib/cast.lib`、`lib/pyclariuscast.pyd` | Windows 对应文件 |
| `lib/cast/*.h` | C 头文件 —— `cast.h`、`cast_def.h`、`cast_cb.h`、`cast_export.h` |
| `cast/` | 上游 [clariusdev/cast](https://github.com/clariusdev/cast)，以 git submodule 形式引入 |
| `examples/` | Clarius 官方示例程序（见下） |

submodule 默认不会被克隆：

```bash
git submodule update --init data/clarius_sdk/cast
```

## 谁在加载这些库

[`data/collection/collect_clarius.py`](../collection/collect_clarius.py) 是本项目里唯一接触 SDK 的文件。它在 `import pyclariuscast` 之前按显式路径加载 `lib/libcast.so` 和 `lib/pyclariuscast.so`，所以这些库不需要放进 `LD_LIBRARY_PATH`。如果你移动了 `lib/`，记得同步改那里的 `_lib_dir`。

## 示例程序

这些是 Clarius 官方的，原样保留作参考，**没有**接入本项目的流水线。

| 文件 | 作用 |
|---|---|
| `examples/pycaster.py` | 最小的 Cast API 客户端。用 `python pycaster.py --ip <探头IP>` 运行 |
| `examples/pysidecaster.py` | 基于 PySide6 的 Cast API 图形界面 |
| `examples/pysidecaster_fps.py` | 同上，多一个帧率显示 |
| `examples/pyimu.py` | 显示探头 IMU 数据流的界面。**已知 bug：yaw 不准** |
| `examples/DICOMserver.py` | 在本机起一个 DICOM 服务器。只有探头连在真实 Wi-Fi 网络（而非热点）时才可达。在 Clarius App 里结束扫查后，会把 DICOM 文件写到 `dicom/` |
| `examples/3Dmodel/` | `scanner.obj` / `scanner.mtl` —— `pyimu.py` 用的探头三维模型 |

## 前置条件

一台 Clarius 探头、它的研究 / Cast API 凭证，以及本机到探头的网络连通。训练和运行模型都不需要这些。
