# Performance Guide

## How to Control Speed

Edit `config.yaml` and change the resolution:

```yaml
processing:
  input_width: 512   # ← Change these values
  input_height: 512
```

Then **restart the server** (or restart TouchDesigner if using Script TOP).

## Recommended Presets

### 🚀 Fast (30+ FPS on M1)
```yaml
input_width: 384
input_height: 384
```
- Best for: Real-time interaction, live performance
- Quality: Good for general use
- Speed: Very fast

### ⚖️ Balanced (20-25 FPS on M1) **← DEFAULT**
```yaml
input_width: 512
input_height: 512
```
- Best for: Most use cases
- Quality: Excellent
- Speed: Fast

### 🎨 Quality (12-15 FPS on M1)
```yaml
input_width: 768
input_height: 768
```
- Best for: Recording, final output
- Quality: Very good
- Speed: Moderate

### 💎 Best (8-12 FPS on M1)
```yaml
input_width: 1024
input_height: 1024
```
- Best for: Offline rendering, stills
- Quality: Best
- Speed: Slower

### ⚡ Ultra Fast (40+ FPS on M1)
```yaml
input_width: 256
input_height: 256
```
- Best for: Sketching, prototyping
- Quality: Reduced detail
- Speed: Very fast

## How It Works

The server now:
1. **Receives** your TouchDesigner frame (any size)
2. **Resizes** it to the configured resolution
3. **Processes** depth at that size (faster!)
4. **Upscales** back to your original size
5. **Sends** to TouchDesigner

**Example:**
- TD sends 1920x1080 frame
- Server resizes to 512x512
- Processes depth (fast!)
- Upscales to 1920x1080
- Returns to TD

## Other Speed Tips

### 1. Lower Warmup Frames
```yaml
performance:
  warmup_frames: 1  # Default is 3
```

### 2. Use Float16 (default, faster)
```yaml
model:
  precision: "float16"  # Faster than float32
```

### 3. Reduce Queue Size (lower latency)
```yaml
performance:
  send_queue_size: 1
  receive_queue_size: 1
```

### 4. Try Different Colormaps
Some are slightly faster than others:
```yaml
processing:
  output:
    colormap: "gray"  # Fastest
    # vs
    colormap: "turbo"  # Slightly slower (default)
```

## Measuring Performance

Check the server log for FPS:
```
2025-10-18 20:30:45 [INFO] Processed 30 frames | FPS: 22.5
```

Or in TouchDesigner:
- Look at the "FPS" parameter on the Script TOP

## Quick Changes

**Want it faster right now?**
1. Open `config.yaml`
2. Change both numbers to `384`
3. Save
4. Restart (in TD: toggle Active off/on)

**Want best quality?**
1. Change to `1024`
2. Accept slower FPS
3. Great for recordings!

## Platform Differences

| Platform | 384x384 | 512x512 | 1024x1024 |
|----------|---------|---------|-----------|
| M1/M2 Mac | 30+ FPS | 20-25 FPS | 8-12 FPS |
| M3 Mac | 35+ FPS | 25-30 FPS | 12-15 FPS |
| Intel Mac (CPU) | 8-10 FPS | 5-8 FPS | 2-3 FPS |

**Note:** These are estimates. Your mileage may vary!

## Finding Your Sweet Spot

1. Start with 512x512 (default)
2. Too slow? Try 384x384
3. Still slow? Try 256x256
4. Want better quality? Try 768x768
5. Need maximum quality? Try 1024x1024

Remember: The server returns your original resolution regardless of processing size!
