# Converting Depth Maps to 3D Positions

## Overview

You have:
- RGB image (color)
- Depth map (distance from camera)

You want:
- 3D positions (X, Y, Z coordinates)

This creates a **point cloud** you can use for 3D effects in TouchDesigner!

## The Formula

For each pixel at position `(x, y)`:

```python
# Get depth value at this pixel
Z = depth_map[y, x]

# Image dimensions and center
width, height = image.shape
cx = width / 2
cy = height / 2

# Focal length (from Depth Pro)
fx = focal_length_px
fy = focal_length_px  # Usually same as fx

# Convert to 3D position
X = (x - cx) * Z / fx
Y = (y - cy) * Z / fy
# Z is already the depth
```

**Result:** 3D point at `(X, Y, Z)` with color from `RGB[y, x]`

## In TouchDesigner

### Method 1: Using CHOPs (For Sparse Points)

```python
# In a CHOP Execute DAT
def create_point_cloud(depth_chop, rgb_top, focal_length):
    width = rgb_top.width
    height = rgb_top.height

    cx = width / 2.0
    cy = height / 2.0

    points = []
    colors = []

    # Sample every Nth pixel (for performance)
    step = 4

    for y in range(0, height, step):
        for x in range(0, width, step):
            # Get depth (assuming depth is in a CHOP channel)
            z = depth_chop[f'pixel_{y}_{x}'][0]

            if z > 0:  # Valid depth
                # Convert to 3D
                X = (x - cx) * z / focal_length
                Y = (y - cy) * z / focal_length
                Z = z

                # Get color
                r, g, b, a = rgb_top.sample(x, y)

                points.append([X, Y, Z])
                colors.append([r, g, b])

    return points, colors
```

### Method 2: Using TOPs + SOPs (Better)

Use a **GLSL TOP** to compute positions, then convert to geometry:

**GLSL Pixel Shader:**
```glsl
// Input: depth map in red channel
// Output: XYZ position in RGB

uniform float focal_length;
uniform vec2 image_size;

void main() {
    vec2 uv = gl_FragCoord.xy / image_size;
    float depth = texture(sTD2DInputs[0], uv).r;

    // Image center
    vec2 center = image_size * 0.5;

    // Pixel coordinates
    vec2 pixel = gl_FragCoord.xy;

    // Convert to 3D
    float X = (pixel.x - center.x) * depth / focal_length;
    float Y = (pixel.y - center.y) * depth / focal_length;
    float Z = depth;

    // Output position as color (will be read by SOP)
    fragColor = vec4(X, Y, Z, 1.0);
}
```

Then use **TOP to CHOP to SOP** to create geometry.

### Method 3: Point File Export (Simplest)

Export to standard point cloud format (.ply):

```python
# Python script to export PLY file
def export_point_cloud(depth_array, rgb_array, focal_length, filename):
    import numpy as np

    height, width = depth_array.shape
    cx = width / 2.0
    cy = height / 2.0

    # Create meshgrid of pixel coordinates
    x_coords, y_coords = np.meshgrid(range(width), range(height))

    # Convert to 3D
    Z = depth_array
    X = (x_coords - cx) * Z / focal_length
    Y = (y_coords - cy) * Z / focal_length

    # Flatten arrays
    points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
    colors = rgb_array.reshape(-1, 3)

    # Filter out invalid depths
    valid = Z.flatten() > 0
    points = points[valid]
    colors = colors[valid]

    # Write PLY file
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for i in range(len(points)):
            x, y, z = points[i]
            r, g, b = (colors[i] * 255).astype(int)
            f.write(f"{x} {y} {z} {r} {g} {b}\n")
```

## What We Need to Add to the Server

Currently, the server only returns the **colorized depth map**.

To do 3D reconstruction, you need:
1. ✅ Depth values (currently sent as colors)
2. ❌ **Raw depth values** (not sent yet)
3. ❌ **Focal length** (not sent yet)

### Option A: Send Metadata Separately

Modify server to send:
```python
{
    'depth_colored': rgb_array,  # What we send now
    'depth_raw': depth_array,     # Raw depth values
    'focal_length': focal_length_px,
    'width': width,
    'height': height
}
```

### Option B: Encode in Image

Pack focal length into alpha channel or unused space:
```python
# Encode focal length in bottom-right corner
depth_colored[0, 0] = focal_length_px / 10000.0  # Normalized
```

### Option C: Separate Channel (Recommended)

Send **two images**:
1. RGB colorized depth (what you see)
2. Grayscale raw depth (what you use for 3D)

## TouchDesigner Workflow

```
[Video In TOP]
      ↓
[Script TOP - Depth Pro Client]
      ↓
[Depth Map TOP] ──→ [GLSL TOP - Depth to Position]
      ↓                        ↓
[Composite]            [TOP to CHOP]
                              ↓
                       [CHOP to SOP]
                              ↓
                       [3D Point Cloud!]
```

## Example: Simple Point Cloud in TD

```python
# In a SOP Execute DAT
def create_points(scriptOp):
    # Get depth TOP and RGB TOP
    depth_top = op('depth_map')
    rgb_top = op('rgb_input')

    # Get focal length (from somewhere - we need to add this!)
    focal_length = 1000.0  # Placeholder

    # Sample rate (every Nth pixel)
    step = 2

    # Create points
    geo = scriptOp.createGeo()

    for y in range(0, depth_top.height, step):
        for x in range(0, depth_top.width, step):
            # Get depth (from red channel of colored depth)
            depth = depth_top.sample(x, y)[0]

            if depth > 0.1:  # Skip background
                # Convert to 3D
                X = (x - depth_top.width/2) * depth / focal_length
                Y = (y - depth_top.height/2) * depth / focal_length
                Z = depth * 100  # Scale depth

                # Create point
                pt = geo.addPoint()
                pt.P = (X, Y, Z)

                # Set color
                r, g, b, a = rgb_top.sample(x, y)
                pt.Cd = (r, g, b)
```

## Want Me to Add This?

I can modify the server to:
1. Send raw depth values (not just colored)
2. Send focal length back to TouchDesigner
3. Add example TD project that creates 3D points

This would give you everything you need for proper 3D reconstruction!

Should I add it?
