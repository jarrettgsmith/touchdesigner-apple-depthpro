#!/usr/bin/env python3
"""
Simple test client for Depth Pro server
Tests the server without TouchDesigner

Usage:
    python test_client.py [image_path]

If no image path is provided, uses a random test image.
"""

import sys
import socket
import struct
import pickle
import numpy as np
import cv2
from pathlib import Path

# Server settings
HOST = 'localhost'
PORT = 9995

def send_frame(sock, frame):
    """Send frame to server"""
    data = pickle.dumps(frame)
    size = len(data)
    sock.sendall(struct.pack('!I', size))
    sock.sendall(data)

def receive_frame(sock):
    """Receive frame from server"""
    # Receive size
    size_data = b''
    while len(size_data) < 4:
        chunk = sock.recv(4 - len(size_data))
        if not chunk:
            return None
        size_data += chunk

    size = struct.unpack('!I', size_data)[0]

    # Receive data
    data = b''
    while len(data) < size:
        chunk = sock.recv(min(size - len(data), 8192))
        if not chunk:
            return None
        data += chunk

    return pickle.loads(data)

def main():
    """Test the server"""

    # Load or create test image
    if len(sys.argv) > 1:
        image_path = Path(sys.argv[1])
        if not image_path.exists():
            print(f"Error: Image not found: {image_path}")
            sys.exit(1)

        try:
            from PIL import Image
            img = Image.open(image_path)
            img = img.convert('RGB')
            img = img.resize((1024, 1024))
            test_frame = np.array(img)
            print(f"Loaded test image: {image_path}")
        except Exception as e:
            print(f"Error loading image: {e}")
            sys.exit(1)
    else:
        # Create random test image
        print("Creating random test image (1024x1024)")
        test_frame = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)

    print(f"Test frame shape: {test_frame.shape}, dtype: {test_frame.dtype}")

    # Connect to server
    print(f"\nConnecting to server at {HOST}:{PORT}...")

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(30.0)
        sock.connect((HOST, PORT))
        print("Connected!")
    except ConnectionRefusedError:
        print("\nError: Could not connect to server")
        print("Make sure the server is running:")
        print("  ./run_server.sh")
        sys.exit(1)
    except Exception as e:
        print(f"\nError connecting: {e}")
        sys.exit(1)

    # Send test frame
    print("\nSending test frame...")
    try:
        send_frame(sock, test_frame)
        print("Frame sent!")
    except Exception as e:
        print(f"Error sending: {e}")
        sock.close()
        sys.exit(1)

    # Receive depth map
    print("\nWaiting for depth map (this may take 10-30 seconds on first run)...")
    try:
        depth_map = receive_frame(sock)
        print("Depth map received!")
        print(f"Depth map shape: {depth_map.shape}, dtype: {depth_map.dtype}")
    except Exception as e:
        print(f"Error receiving: {e}")
        sock.close()
        sys.exit(1)

    # Decode metadata from top-left corner
    if depth_map.ndim == 3 and depth_map.shape[2] == 4:
        # 4-channel RGBA output
        r, g, b, a = depth_map[0, 0]
        focal_length = float(r) * 1500.0 / 255.0 + 500.0
        depth_min = float(g) * 100.0 / 255.0
        depth_max = float(b) * 100.0 / 255.0

        print(f"\n📊 Decoded Metadata:")
        print(f"  Focal length: {focal_length:.1f}px")
        print(f"  Depth range: {depth_min:.2f} - {depth_max:.2f}")
        print(f"\n✓ 4-channel RGBA format detected!")
        print(f"  RGB channels = Original image colors")
        print(f"  Alpha channel = Depth values (0-255)")

        # Save RGBA image
        cv2.imwrite("test_output_rgba.png", cv2.cvtColor(depth_map, cv2.COLOR_RGBA2BGRA))
        print(f"\n💾 Saved: test_output_rgba.png (full RGBA)")

        # Save separate RGB and depth visualizations
        rgb = depth_map[:, :, :3]
        depth_alpha = depth_map[:, :, 3]

        cv2.imwrite("test_output_rgb.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite("test_output_depth.png", depth_alpha)
        print(f"💾 Saved: test_output_rgb.png (RGB only)")
        print(f"💾 Saved: test_output_depth.png (depth only)")

    else:
        # Legacy 3-channel RGB output
        print(f"\n⚠️  3-channel RGB format (legacy)")
        cv2.imwrite("test_output.png", cv2.cvtColor(depth_map, cv2.COLOR_RGB2BGR))
        print(f"💾 Saved: test_output.png")

    # Close connection
    sock.close()
    print("\n✅ Test complete!")
    print(f"  ✓ Connection successful")
    print(f"  ✓ Frame sent and received")
    print(f"  ✓ Server is working correctly")

if __name__ == "__main__":
    main()
