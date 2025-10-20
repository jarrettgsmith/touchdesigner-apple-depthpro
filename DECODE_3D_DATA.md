# Decoding 3D Data from Depth Pro Server

## What You Receive (4-channel RGBA)

The server now sends a **4-channel RGBA image** with everything you need for 3D:

```
R channel = Original RED from your video
G channel = Original GREEN from your video
B channel = Original BLUE from your video
A channel = Normalized depth (0-255)
```

**Plus metadata in top-left corner (3x3 pixels):**
```
Top-left R = Focal length (encoded)
Top-left G = Depth minimum (encoded)
Top-left B = Depth maximum (encoded)
```

## In TouchDesigner: Extract Metadata

```python
# In a Script DAT
def decode_metadata(depth_top):
    """Decode metadata from top-left corner"""

    # Sample top-left pixel
    r, g, b, a = depth_top.sample(0, 0)

    # Decode focal length (normalized from 500-2000 range)
    focal_length = r * 1500 + 500

    # Decode depth range (normalized from 0-100 range)
    depth_min = g * 100
    depth_max = b * 100

    return focal_length, depth_min, depth_max

# Usage
top = op('scriptTOP_depthpro')
focal_length, depth_min, depth_max = decode_metadata(top)
print(f"Focal: {focal_length}, Range: {depth_min}-{depth_max}")
```

## Convert Alpha to Real Depth

```python
def get_real_depth(alpha_value, depth_min, depth_max):
    """Convert normalized alpha (0-1) to real depth"""
    return alpha_value * (depth_max - depth_min) + depth_min
```

## Create Point Cloud (Python SOP)

```python
# In a Script SOP
def onCook(scriptOp):
    scriptOp.clear()

    # Get the depth TOP
    depth_top = op('scriptTOP_depthpro')

    # Decode metadata
    focal_length, depth_min, depth_max = decode_metadata(depth_top)

    # Image dimensions
    width = depth_top.width
    height = depth_top.height
    cx = width / 2.0
    cy = height / 2.0

    # Sampling step (every Nth pixel for performance)
    step = 4

    # Create points
    for y in range(0, height, step):
        for x in range(0, width, step):
            # Sample pixel
            r, g, b, a = depth_top.sample(x, y)

            # Skip top-left metadata corner
            if x < 3 and y < 3:
                continue

            # Convert alpha to real depth
            depth = a * (depth_max - depth_min) + depth_min

            # Skip background (very far)
            if depth < 0.1 or depth > 50:
                continue

            # Convert to 3D position
            X = (x - cx) * depth / focal_length
            Y = -(y - cy) * depth / focal_length  # Flip Y
            Z = -depth  # Negative Z goes into screen

            # Create point
            pt = scriptOp.appendPoint()
            pt.point.P = (X, Y, Z)
            pt.point.Cd = (r, g, b)  # Original RGB color

def decode_metadata(depth_top):
    r, g, b, a = depth_top.sample(0, 0)
    focal_length = r * 1500 + 500
    depth_min = g * 100
    depth_max = b * 100
    return focal_length, depth_min, depth_max
```

## Using GLSL for Speed (Better!)

**Step 1: Extract depth to separate TOP**

GLSL TOP - "Extract Depth":
```glsl
// Input: RGBA from server
// Output: Just the depth

out vec4 fragColor;
void main() {
    vec2 uv = vUV.st;
    vec4 data = texture(sTD2DInputs[0], uv);

    // Get metadata from top-left
    vec4 meta = texture(sTD2DInputs[0], vec2(0.001, 0.001));
    float depth_min = meta.g * 100.0;
    float depth_max = meta.b * 100.0;

    // Convert alpha to real depth
    float depth = data.a * (depth_max - depth_min) + depth_min;

    // Output as grayscale
    fragColor = vec4(depth, depth, depth, 1.0);
}
```

**Step 2: Create position map**

GLSL TOP - "Depth to Position":
```glsl
// Convert depth to XYZ position
uniform vec2 uResolution;

out vec4 fragColor;
void main() {
    vec2 uv = vUV.st;

    // Get metadata
    vec4 meta = texture(sTD2DInputs[0], vec2(0.001, 0.001));
    float focal_length = meta.r * 1500.0 + 500.0;
    float depth_min = meta.g * 100.0;
    float depth_max = meta.b * 100.0;

    // Get depth from alpha
    float depth_norm = texture(sTD2DInputs[0], uv).a;
    float depth = depth_norm * (depth_max - depth_min) + depth_min;

    // Image center
    vec2 center = uResolution * 0.5;

    // Pixel coordinates
    vec2 pixel = uv * uResolution;

    // Convert to 3D
    float X = (pixel.x - center.x) * depth / focal_length;
    float Y = -(pixel.y - center.y) * depth / focal_length;  // Flip Y
    float Z = -depth;  // Negative Z

    // Normalize for output (will need to scale in SOP)
    // Or output directly if using in shader
    fragColor = vec4(X, Y, Z, 1.0);
}
```

**Step 3: Convert to SOP**

Use **TOP to CHOP to SOP** chain or direct render to geometry.

## Simple Example: Just RGB + Depth Visualization

```python
# In a Script DAT connected to scriptTOP_depthpro

def onFrameStart(frame):
    top = op('scriptTOP_depthpro')

    # Get metadata
    r, g, b, a = top.sample(0, 0)
    focal = r * 1500 + 500

    # Store in project parameters
    parent().par.Focallength = focal

    print(f"Frame {frame}: Focal length = {focal:.1f}px")
```

Then in your point cloud SOP, you can reference:
```python
focal_length = parent().par.Focallength.eval()
```

## Quick Test

Add this to a Text DAT to see what you're getting:

```python
top = op('scriptTOP_depthpro')

# Sample different locations
tl = top.sample(0, 0)  # Top-left metadata
center = top.sample(top.width/2, top.height/2)  # Center pixel

print("Top-left (metadata):")
print(f"  R={tl[0]:.3f} G={tl[1]:.3f} B={tl[2]:.3f} A={tl[3]:.3f}")

print("\nCenter pixel:")
print(f"  RGB=({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
print(f"  Depth={center[3]:.3f}")

# Decode
focal = tl[0] * 1500 + 500
depth_min = tl[1] * 100
depth_max = tl[2] * 100

print(f"\nDecoded metadata:")
print(f"  Focal length: {focal:.1f}px")
print(f"  Depth range: {depth_min:.2f} - {depth_max:.2f}")
```

## Full Workflow

```
[Video In]
    ↓
[Script TOP - Depth Pro] → RGBA output
    ↓
    ├─→ [Display as-is] (shows original RGB)
    │
    ├─→ [Select TOP - Alpha only] → Depth visualization
    │
    └─→ [Python SOP] → 3D Point Cloud!
```

## Performance Tips

- Use `step = 4` or higher (sample every 4th pixel)
- Add Level TOP after to darken RGB if needed
- Use Limit SOP to cull far points
- Add Point SOP to adjust scale

You now have everything for full 3D reconstruction! 🎉
