### kinematics6d -- 1000 constraints

| method | SR median | SR 5th percentile | SWD (x floor) | MMD (x floor) | JSD (x floor) | KLD median | in support (%) |
|:---|:---|:---|:---|:---|:---|:---|:---|
| Ground Truth | 100.000 | 100.000 | 0.04565 (1.0x) | 0.0002106 (1.0x) | 0.002489 (1.0x) | -- | 100.0000 |
| Explicit (ours) | 93.910 | 35.401 | 0.04634 (1.0x) | 0.0002425 (1.1x) | 0.05284 (20.9x) | 0.1179 | 97.4400 |
| ECI | 100.000 | 100.000 | 0.2092 (4.3x) | 0.01058 (45.7x) | 0.2722 (112.0x) | -- | 98.7000 |
| HardFlow | 94.230 | 42.191 | 0.1018 (2.1x) | 0.001598 (6.7x) | 0.1008 (41.8x) | -- | 99.3000 |

### Excess MMD by constraint mass

| mass band | Explicit (ours) | ECI | HardFlow |
|:---|:---|:---|:---|
| [0.0, 0.1) | 1.1x | 45.9x | 4.1x |
| [0.1, 0.2) | 1.1x | 58.0x | 7.8x |
| [0.2, 0.4) | 1.2x | 35.0x | 12.8x |
| [0.4, 0.6) | 1.4x | 40.1x | 17.3x |

### Excess SWD by constraint mass

| mass band | Explicit (ours) | ECI | HardFlow |
|:---|:---|:---|:---|
| [0.0, 0.1) | 1.0x | 4.4x | 1.9x |
| [0.1, 0.2) | 1.0x | 4.7x | 2.3x |
| [0.2, 0.4) | 1.0x | 4.2x | 2.6x |
| [0.4, 0.6) | 1.1x | 4.1x | 2.7x |
