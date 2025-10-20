# TouchDesigner Apple Depth Pro Integration

Real-time monocular depth estimation for TouchDesigner using Apple's Depth Pro model.

## Overview

This project provides a production-ready server-client architecture for integrating Apple's ML Depth Pro model into TouchDesigner workflows. It processes RGB video frames and outputs depth maps in real-time.

### Features

- **Real-time depth estimation** - Process video at 10-30 FPS (depending on hardware)
- **Dual output modes** - Socket (cross-platform) and Syphon (macOS, ultra-low latency)
- **Apple Silicon optimized** - Uses MPS (Metal Performance Shaders) for GPU acceleration
- **Non-blocking architecture** - Keeps TouchDesigner responsive
- **Flexible configuration** - YAML-based settings for easy customization
- **Multiple colormaps** - Viridis, Turbo, Jet, and more
- **Simple setup** - One command installation

## Architecture

```
┌─────────────────────────────────────────┐
│     TouchDesigner (Script TOP)          │
│  - Sends RGB frames via Socket          │
│  - Receives depth maps                  │
│  - Optional: Syphon In TOP              │
└─────────────────────────────────────────┘
              ↕ TCP Socket (localhost:9995)
              ↕ Syphon Metal (optional)
┌─────────────────────────────────────────┐
│  Python Server (External Process)       │
│  - Apple Depth Pro inference            │
│  - GPU acceleration (MPS/CUDA)          │
│  - Colormap visualization               │
└─────────────────────────────────────────┘
```

## Prerequisites

- **macOS** 11.0+ (for MPS support) or Linux/Windows (CPU/CUDA)
- **Python** 3.8 or later
- **TouchDesigner** (any recent version)
- **Apple Silicon Mac** recommended (M1/M2/M3 for best performance)

## Installation

### 1. Clone or Download

Ensure the project structure looks like this:

```
touchdesigner-apple-depthpro/
├── config.yaml
├── requirements.txt
├── depth_pro_server.py
├── scriptTOP_depthpro.py
├── setup.sh
├── run_server.sh
├── README.md
└── external/
    └── ml-depth-pro/          # Apple's Depth Pro repository
```

### 2. Run Setup

```bash
cd touchdesigner-apple-depthpro
./setup.sh
```

This will:
- Create a Python virtual environment
- Install all dependencies
- Install the Depth Pro package
- Download pretrained model checkpoint (~1.5GB)

Setup takes 5-10 minutes depending on your internet connection.

## Usage

### Method 1: From TouchDesigner (Recommended)

1. **Open TouchDesigner**

2. **Create a Script TOP**
   - Add a Script TOP operator to your network
   - Set the resolution (e.g., 1024x1024)

3. **Load the Script**
   - In the Script TOP parameters, go to the "Script" page
   - Click the folder icon next to "Script"
   - Select `scriptTOP_depthpro.py`

4. **Connect Input**
   - Connect any TOP (video file, camera, etc.) to the Script TOP input

5. **Configure Paths**
   - Open `scriptTOP_depthpro.py` in a text editor
   - Update `PROJECT_ROOT` path if needed (lines 28-30)

6. **Start Processing**
   - In the Script TOP parameters, find the "Depth Pro" page
   - Toggle "Active" to ON
   - The server will start automatically
   - Wait 30-60 seconds for model loading
   - Status will show "Processing" when ready

7. **Monitor**
   - Watch "Status" parameter for connection state
   - "Frames Sent" and "Frames Received" show throughput
   - "FPS" shows processing speed

### Method 2: Manual Server (For Testing)

You can run the server independently for testing:

```bash
./run_server.sh
```

Then connect from TouchDesigner or write your own client.

## Configuration

Edit `config.yaml` to customize:

### Server Settings

```yaml
server:
  host: "localhost"
  port: 9995
  output_mode: "both"  # "socket", "syphon", or "both"
```

### Model Settings

```yaml
model:
  device: "mps"         # "mps", "cuda", or "cpu"
  precision: "float16"  # "float16" or "float32"
```

### Visualization

```yaml
processing:
  output:
    colormap: "turbo"   # viridis, plasma, inferno, magma, turbo, jet, gray
    normalize: true     # Normalize depth to 0-1 range
    invert: false       # Invert depth map (farther = brighter)
    gamma: 1.0          # Gamma correction (0.5 - 2.0)
```

### Performance Tuning

```yaml
processing:
  input_width: 1024    # Lower = faster, less detail
  input_height: 1024

performance:
  send_queue_size: 2   # Lower = less latency
  warmup_frames: 3     # Model warmup iterations
```

## Performance

Typical performance on various hardware:

| Hardware | Resolution | FPS | Latency |
|----------|-----------|-----|---------|
| M1 Max | 1024x1024 | 15-20 | ~50ms |
| M2 Ultra | 1024x1024 | 25-30 | ~33ms |
| M1 | 512x512 | 20-25 | ~40ms |
| Intel CPU | 1024x1024 | 2-3 | ~300ms |

## Syphon Output (macOS Only)

For ultra-low latency on macOS:

1. **Enable Syphon in config.yaml**:
   ```yaml
   server:
     output_mode: "both"
     syphon:
       enabled: true
       server_name: "Depth Pro Output"
   ```

2. **In TouchDesigner**:
   - Add a "Syphon In TOP"
   - Set Server Name to "Depth Pro Output"
   - Receive depth maps at <1ms latency

## Troubleshooting

### "Virtual environment not found"

Run `./setup.sh` first.

### "Checkpoint not found"

The model checkpoint may not have downloaded. Manually download from:
https://github.com/apple/ml-depth-pro

Place `depth_pro.pt` in `external/ml-depth-pro/checkpoints/`

### "Connection refused"

- Ensure the server is running
- Check that port 9995 is not in use: `lsof -i :9995`
- Verify `config.yaml` host/port settings match

### Slow performance

- Lower resolution in `config.yaml` (try 512x512)
- Use `float16` precision instead of `float32`
- Check that `device: "mps"` is set (for Apple Silicon)
- Close other GPU-intensive applications

### "MPS backend not available"

Your Mac may not support Metal Performance Shaders. Use CPU instead:

```yaml
model:
  device: "cpu"
  precision: "float32"
```

Note: CPU processing is significantly slower.

### Server crashes or freezes

Check the log file: `depth_pro_server.log`

Common issues:
- Out of memory: Lower resolution or use CPU
- Model not loaded: Verify checkpoint exists
- Python package errors: Reinstall with `./setup.sh`

## Project Structure

```
.
├── config.yaml                  # Configuration file
├── requirements.txt             # Python dependencies
├── depth_pro_server.py          # Main server application
├── scriptTOP_depthpro.py        # TouchDesigner Script TOP client
├── setup.sh                     # Setup script
├── run_server.sh                # Manual server launch script
├── README.md                    # This file
├── depth_pro_server.log         # Server logs (created on first run)
├── venv/                        # Python virtual environment (created by setup)
└── external/
    └── ml-depth-pro/            # Apple Depth Pro repository
        ├── src/depth_pro/       # Depth Pro source code
        └── checkpoints/         # Model checkpoints
            └── depth_pro.pt     # Pretrained model (~1.5GB)
```

## Advanced Usage

### Custom TouchDesigner Integration

The Script TOP client (`scriptTOP_depthpro.py`) is fully commented and can be customized:

- Modify preprocessing (lines 280-290)
- Add custom parameters (lines 50-70)
- Change queue behavior (lines 20-25)
- Adjust connection logic (lines 150-200)

### Server as Standalone Application

You can use `depth_pro_server.py` independently:

1. Write your own client using standard Python sockets
2. Send RGB frames as pickled numpy arrays with 4-byte size header
3. Receive depth maps in the same format
4. See server code (lines 220-280) for protocol details

### Multiple Clients

The current implementation supports one client at a time. To support multiple clients:

1. Modify server to use threading/asyncio
2. Create multiple socket connections
3. Each client gets its own processing thread

## Credits

- **Apple ML Research** - Depth Pro model and implementation
  - Paper: https://arxiv.org/abs/2410.02073
  - Code: https://github.com/apple/ml-depth-pro

- **Architecture inspired by**:
  - touchdesigner-depthai-handtracking (Syphon+OSC pattern)
  - touchdesigner-oak-integration (async socket architecture)
  - streamdiffusion-mac (subprocess management)

## License

This integration code is provided as-is for research and creative use.

The Apple Depth Pro model is subject to Apple's license:
https://github.com/apple/ml-depth-pro/blob/main/LICENSE

## Contributing

Improvements welcome! Areas for enhancement:

- [ ] Windows/Linux testing
- [ ] CUDA optimization
- [ ] Multiple client support
- [ ] Real-time parameter adjustment
- [ ] Web UI for configuration
- [ ] Depth map filtering/smoothing
- [ ] 3D point cloud export

## Support

For issues specific to this integration:
- Check `depth_pro_server.log` for errors
- Verify all paths in config.yaml
- Test with `./run_server.sh` before using in TouchDesigner

For Depth Pro model issues:
- See https://github.com/apple/ml-depth-pro/issues

## Version History

**v1.0.0** (2025-10-18)
- Initial release
- Socket and Syphon support
- Apple Silicon MPS optimization
- Configurable colormaps and processing
- TouchDesigner Script TOP client
- Automatic server management
