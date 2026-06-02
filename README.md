
# m5stick-fluid

Real-time FLIP fluid simulation running on the M5Stack StickS3 (ESP32-based).
Tilt the device and gravity follows the IMU. Connect it to a PC and CPU load drives the sloshing intensity.

> Demo video
https://github.com/user-attachments/assets/f462f6c0-06ec-47af-815c-d81d297b62bf


https://github.com/user-attachments/assets/9e4bc0a9-a659-4337-b666-420a397b5fbe


https://github.com/user-attachments/assets/4a39b8e1-1f76-482c-ba0a-b75d337a7988







## Features

- FLIP / PIC hybrid solver — stable yet detail-preserving fluid dynamics
- IMU-driven gravity — tilt in any direction, fluid responds in real time
- CPU-reactive sloshing — feed CPU usage over serial; higher load produces more violent waves
- Particle density rendering with gamma correction and a pink gradient color map
- Floating heart easter egg — a buoyancy-simulated object that rides the fluid surface; toggle with BtnB

---

## Controls

| Button | Action |
|--------|--------|
| BtnA | Toggle CPU monitor mode (enables / disables serial input) |
| BtnB | Show or hide the floating heart |

---

## CPU Monitor Mode

Send a float value between `0.0` and `100.0` over serial at 115200 baud to control sloshing intensity. If no data arrives for 3 seconds, the device returns to IMU-only mode.

```python
# Example: stream CPU usage from a host machine
import serial, psutil, time

s = serial.Serial('/dev/ttyUSB0', 115200)
while True:
    s.write(f'{psutil.cpu_percent()}\n'.encode())
    time.sleep(0.5)
```

---

## Hardware

| Item | Detail |
|------|--------|
| Device | M5Stack StickS3 |
| Display | 135 x 240 px, ST7789 |
| IMU | Built-in accelerometer |
| Connection | USB-C serial at 115200 baud |

---

## Build and Flash

### PlatformIO (recommended)

```ini
; platformio.ini
; Note: PlatformIO has no official StickS3 board definition.
; Use the generic ESP32-S3 devkit and configure flash/PSRAM manually.
[env:m5sticks3]
platform = espressif32
board = esp32-s3-devkitc-1
framework = arduino
board_build.arduino.memory_type = qio_opi
board_upload.flash_size = 8MB
lib_deps =
    m5stack/M5Unified @ ^0.2.0
```

```bash
pio run --target upload
```

### Arduino IDE

1. Install board support: M5Stack (via Board Manager)
2. Install library: M5Unified
3. Open `src/main.cpp`, select board M5Stack StickS3, and upload

---

## How It Works

Each frame runs the full FLIP pipeline in this order:

```
1. integrate_particles     Apply gravity and advance particle positions (symplectic Euler)
2. push_particles_apart    Resolve overlaps using a spatial hash neighbor search
3. handle_collisions       Clamp particles to tank bounds, reflect and damp velocity
4. transfer_vel (P -> G)   Splat particle velocities onto the MAC grid (bilinear weights)
5. update_density          Accumulate particle density onto the grid
6. solve_incompressibility Gauss-Seidel pressure projection to enforce divergence-free flow
7. transfer_vel (G -> P)   Gather updated velocities back to particles using FLIP/PIC blend
8. render                  Density field -> gamma LUT -> color map -> display
```

Particles are initialized in a hexagonal close-packed layout. The floating heart samples
the velocity field via bilinear interpolation and responds to buoyancy, fluid drag, and
local vorticity.

---

## Tuning Reference

All primary parameters are defined as macros at the top of `main.cpp`.
The table below describes what each one controls and the effect of changing it.

### Simulation behavior

| Parameter | Default | Effect |
|-----------|---------|--------|
| `FILL_RATIO` | `0.55` | Fraction of tank height filled with fluid at startup. Increase for more fluid, decrease for less. |
| `GRAVITY_SCALE` | `7.0` | Multiplier applied to the normalized IMU gravity vector. Increase for snappier response to tilting; decrease for sluggish, heavy-feeling fluid. |
| `FLIP_RATIO` | `0.9` | Blend between PIC (0.0) and FLIP (1.0). Higher values preserve more kinetic energy and produce splashier motion but can introduce noise. Lower values are more damped and stable. |
| `DT` | `1/60` | Fixed simulation timestep in seconds. Smaller values improve stability but increase CPU load per frame. Do not exceed `1/SIM_FPS` or the simulation will fall behind. |

### Pressure solver

| Parameter | Default | Effect |
|-----------|---------|--------|
| `PRES_ITERS` | `8` | Number of Gauss-Seidel iterations per frame. More iterations produce a more accurately incompressible result but cost more CPU time. 4 is viable on this hardware; above 12 provides diminishing returns. |
| `OVER_RELAX` | `1.3` | Over-relaxation factor for the pressure solve. Values in [1.0, 1.9] accelerate convergence. Values at or above 2.0 will cause divergence. |

### Particle behavior

| Parameter | Default | Effect |
|-----------|---------|--------|
| `PUSH_ITERS` | `1` | Iterations of the particle separation pass per frame. Increasing to 2 or 3 reduces clumping at the cost of extra compute. |
| `WALL_DAMP` (in `handle_collisions`) | `0.20` | Velocity retention on wall bounce. 0.0 = no bounce; 1.0 = perfectly elastic. |
| `damp` (in `integrate_particles`) | `0.995` | Per-step velocity damping. Values closer to 1.0 let fluid stay in motion longer; values closer to 0.98 make it settle faster. |

### Rendering

| Parameter | Default | Effect |
|-----------|---------|--------|
| `DENSITY_CLAMP` | `1.2` | Density values above this multiple of rest density are rendered at full brightness. Lower values make fluid look more uniformly bright; higher values increase contrast. |
| `GAMMA_F` | `0.6` | Gamma exponent applied before color mapping. Below 1.0 brightens midtones; above 1.0 darkens them. |
| `CELL_SIZE` | `5` | Pixel size of each rendered grid cell. Reducing this requires updating GRID_W, GRID_H, SIM_W, and SIM_H consistently and will increase particle count and memory usage. |

### CPU sloshing

| Parameter | Default | Effect |
|-----------|---------|--------|
| `GRAVITY_SCALE` | `7.0` | Also sets the amplitude envelope for CPU-driven perturbation. At 100% CPU the vertical amplitude is 2.0 * GRAVITY_SCALE, enough to temporarily reverse downward gravity. |
| `freq` formula | `1 + intensity * 3` | Oscillation frequency in Hz as a function of CPU load (0-1). Adjust the multiplier to change rocking speed at high load. |

### Heart physics

| Parameter | Location | Effect |
|-----------|----------|--------|
| `buoy` | `heart_update` | Buoyancy acceleration (px/s^2) pushing the heart against gravity. Increase to make it float higher and more aggressively. |
| `drag` | `heart_update` | How strongly the heart tracks local fluid velocity. Higher values cause it to follow the flow more tightly. |
| `gravity` | `heart_update` | Gravitational acceleration applied when the heart is airborne, as a multiple of GRAVITY_SCALE. |

---

## Project Structure

```
m5stick-fluid/
├── src/
│   └── main.cpp
├── platformio.ini
├── README.md
└── assets/
    └── demo.gif
```

---

## License

MIT
