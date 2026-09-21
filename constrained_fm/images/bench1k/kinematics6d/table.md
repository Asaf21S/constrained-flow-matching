### kinematics6d -- 1000 constraints

| method | AR median | AR p5 | SWD (x floor) | MMD (x floor) | JSD (x floor) | KLD median | in support (%) |
|:---|:---|:---|:---|:---|:---|:---|:---|
| Ground Truth | 100.000 | 100.000 | 0.04565 (1.0x) | 0.0002106 (1.0x) | 0.002489 (1.0x) | -- | 100.0000 |
| Explicit (ours) | 94.505 | 52.000 | 0.04607 (1.0x) | 0.0002403 (1.1x) | 0.03149 (13.6x) | 0.1179 | 96.6700 |
| ECI | 100.000 | 100.000 | 0.222 (4.4x) | 0.01095 (46.1x) | 0.289 (118.5x) | -- | 98.6200 |
| HardFlow | 89.300 | 18.816 | 0.4965 (10.8x) | 0.1027 (425.3x) | 0.07953 (33.7x) | -- | 82.5850 |

### Excess MMD by constraint mass

| mass band | Explicit (ours) | ECI | HardFlow |
|:---|:---|:---|:---|
| [0.0, 0.1) | 1.1x | 48.4x | 674.5x |
| [0.1, 0.2) | 1.1x | 52.9x | 282.5x |
| [0.2, 0.4) | 1.2x | 37.6x | 91.4x |
| [0.4, 0.6) | 1.4x | 35.4x | 36.9x |

### Excess SWD by constraint mass

| mass band | Explicit (ours) | ECI | HardFlow |
|:---|:---|:---|:---|
| [0.0, 0.1) | 1.0x | 4.5x | 13.3x |
| [0.1, 0.2) | 1.0x | 4.5x | 8.9x |
| [0.2, 0.4) | 1.0x | 4.3x | 5.4x |
| [0.4, 0.6) | 1.1x | 3.9x | 4.6x |
