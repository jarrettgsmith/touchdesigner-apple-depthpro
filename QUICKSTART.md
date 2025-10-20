# Quick Start Guide

Get up and running with Depth Pro in 5 minutes.

## Step 1: Setup (One-time, ~10 minutes)

```bash
cd /Users/jarrett/Projects/UCSB/Vision/touchdesigner-apple-depthpro
./setup.sh
```

This installs everything and downloads the model checkpoint (~1.5GB).

**Wait for**: "Setup Complete!" message

## Step 2: Test the Server (Optional)

```bash
./run_server.sh
```

You should see:
```
Loading Depth Pro model...
Model loaded successfully
Starting socket server on localhost:9995
Waiting for TouchDesigner connection...
```

Press Ctrl+C to stop (or leave it running for Step 3).

## Step 3: Use in TouchDesigner

### A. Create the Network

1. Open TouchDesigner
2. Add these operators:
   ```
   [Movie File In TOP] → [Script TOP] → [Out TOP]
   ```

### B. Configure Script TOP

1. Select the Script TOP
2. Parameters → Script page
3. Click folder icon next to "Script"
4. Browse to: `scriptTOP_depthpro.py`
5. Click "Reload Script"

### C. Start Processing

1. Find the "Depth Pro" parameter page
2. Toggle "Active" to ON
3. Wait 30-60 seconds (model loading)
4. Status will change to "Processing"
5. You should see depth maps!

## Step 4: Adjust Settings

### In config.yaml:

**Change colormap**:
```yaml
processing:
  output:
    colormap: "viridis"  # try: turbo, plasma, jet, gray
```

**Improve performance**:
```yaml
processing:
  input_width: 512   # smaller = faster
  input_height: 512
```

**Invert depth**:
```yaml
processing:
  output:
    invert: true  # white = far, black = near
```

Restart the server after changing config.yaml.

## Troubleshooting

**"Virtual environment not found"**
→ Run `./setup.sh`

**"Connection refused"**
→ Check server is running, look for "Waiting for TouchDesigner connection..."

**Slow FPS**
→ Lower resolution in config.yaml (try 512x512)

**Script TOP shows black**
→ Check Status parameter - should say "Processing", not "Connecting..."

**Server won't start**
→ Check `depth_pro_server.log` for errors

## Example TouchDesigner Network

```
┌─────────────┐
│ Movie File  │ (Load a video file)
│  In TOP     │
└──────┬──────┘
       │
       ↓
┌─────────────┐
│  Script TOP │ (Load scriptTOP_depthpro.py)
│             │ (Toggle Active = ON)
└──────┬──────┘
       │
       ↓
┌─────────────┐
│  Level TOP  │ (Optional: adjust brightness/contrast)
└──────┬──────┘
       │
       ↓
┌─────────────┐
│   Out TOP   │
└─────────────┘
```

## Next Steps

- Read the full README.md for advanced features
- Try different colormaps in config.yaml
- Experiment with gamma correction
- Use Syphon output for ultra-low latency (macOS)
- Combine depth with original video using Composite TOP

## Performance Tips

**M1/M2 Mac**:
- 1024x1024: 15-20 FPS
- 512x512: 20-30 FPS

**Intel Mac**:
- Set `device: "cpu"` in config.yaml
- Use 512x512 or smaller
- Expect 2-5 FPS

**Tips**:
- Close other GPU apps (Chrome, etc.)
- Lower resolution if needed
- Use float16 precision (default)
- Enable Syphon for lowest latency

Enjoy real-time depth estimation!
