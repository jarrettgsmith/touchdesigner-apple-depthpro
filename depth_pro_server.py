#!/usr/bin/env python3
"""
Depth Pro Server for TouchDesigner
Provides depth estimation using Apple's Depth Pro model
Supports Socket and Syphon output modes
"""

import os
import sys
import socket
import struct
import pickle
import signal
import time
import logging
from pathlib import Path
from typing import Optional, Tuple

import yaml
import numpy as np
import torch
import cv2
from PIL import Image

# Add Depth Pro to Python path
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR / "external" / "ml-depth-pro" / "src"))

from depth_pro import create_model_and_transforms, load_rgb

# Optional Syphon support (macOS only)
try:
    import syphon
    from syphon.utils.numpy import copy_image_to_mtl_texture
    from syphon.utils.raw import create_mtl_texture
    SYPHON_AVAILABLE = True
except ImportError:
    SYPHON_AVAILABLE = False
    print("Syphon not available - Socket-only mode")


class Config:
    """Configuration loader"""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def get(self, *keys, default=None):
        """Get nested config value"""
        value = self.config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return default
        return value if value is not None else default


class DepthProServer:
    """Depth Pro inference server with Socket and Syphon output"""

    def __init__(self, config: Config):
        self.config = config
        self.running = True
        self.frame_count = 0
        self.last_time = time.time()

        # Setup logging
        self._setup_logging()

        # Load model
        self.device = self._get_device()
        self.model, self.transform = self._load_model()

        # Setup outputs
        self.socket_server = None
        self.syphon_server = None
        self.syphon_texture = None

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _setup_logging(self):
        """Configure logging"""
        level = getattr(logging, self.config.get('logging', 'level', default='INFO'))
        log_file = self.config.get('logging', 'log_file', default='depth_pro_server.log')

        handlers = []
        if self.config.get('logging', 'console_output', default=True):
            handlers.append(logging.StreamHandler())
        if log_file:
            handlers.append(logging.FileHandler(log_file))

        logging.basicConfig(
            level=level,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=handlers
        )
        self.logger = logging.getLogger(__name__)

    def _get_device(self) -> torch.device:
        """Determine compute device"""
        device_name = self.config.get('model', 'device', default='mps')

        if device_name == 'mps' and torch.backends.mps.is_available():
            self.logger.info("Using Apple Silicon GPU (MPS)")
            return torch.device('mps')
        elif device_name == 'cuda' and torch.cuda.is_available():
            self.logger.info("Using NVIDIA GPU (CUDA)")
            return torch.device('cuda')
        else:
            self.logger.warning("GPU not available, using CPU")
            return torch.device('cpu')

    def _load_model(self):
        """Load Depth Pro model"""
        self.logger.info("Loading Depth Pro model...")

        # Create model and transforms
        # The checkpoint path in DEFAULT_MONODEPTH_CONFIG_DICT is relative to where the script runs
        # It will automatically load ./checkpoints/depth_pro.pt if it exists
        model, transform = create_model_and_transforms(
            device=self.device,
            precision=torch.float16 if self.config.get('model', 'precision') == 'float16' else torch.float32
        )

        model.eval()
        self.logger.info("Model loaded successfully")

        return model, transform

    def _setup_socket_server(self) -> socket.socket:
        """Create socket server"""
        host = self.config.get('server', 'host', default='localhost')
        port = self.config.get('server', 'port', default=9995)

        self.logger.info(f"Starting socket server on {host}:{port}")

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        sock.listen(1)

        self.logger.info("Waiting for TouchDesigner connection...")
        conn, addr = sock.accept()
        self.logger.info(f"Connected to {addr}")

        return conn

    def _setup_syphon_server(self):
        """Create Syphon server (macOS only)"""
        if not SYPHON_AVAILABLE:
            return None, None

        server_name = self.config.get('server', 'syphon', 'server_name', default='Depth Pro Output')
        width = self.config.get('processing', 'input_width', default=1024)
        height = self.config.get('processing', 'input_height', default=1024)

        self.logger.info(f"Creating Syphon server: {server_name}")

        server = syphon.SyphonMetalServer(server_name)
        texture = create_mtl_texture(server.device, width, height)

        return server, texture

    def _receive_frame(self, conn: socket.socket) -> Optional[np.ndarray]:
        """Receive frame from TouchDesigner"""
        try:
            # Receive size header (4 bytes)
            size_data = b''
            while len(size_data) < 4:
                chunk = conn.recv(4 - len(size_data))
                if not chunk:
                    return None
                size_data += chunk

            frame_size = struct.unpack('!I', size_data)[0]

            # Receive frame data
            frame_data = b''
            while len(frame_data) < frame_size:
                chunk = conn.recv(min(frame_size - len(frame_data), 8192))
                if not chunk:
                    return None
                frame_data += chunk

            # Deserialize
            frame = pickle.loads(frame_data)
            return frame

        except Exception as e:
            self.logger.error(f"Error receiving frame: {e}")
            return None

    def _send_frame(self, conn: socket.socket, frame: np.ndarray) -> bool:
        """Send processed frame to TouchDesigner"""
        try:
            # Serialize
            frame_data = pickle.dumps(frame)
            size = len(frame_data)

            # Send size header + data
            conn.sendall(struct.pack('!I', size))
            conn.sendall(frame_data)

            return True

        except Exception as e:
            self.logger.error(f"Error sending frame: {e}")
            return False

    def _process_depth(self, rgb_frame: np.ndarray) -> Tuple[np.ndarray, Optional[float]]:
        """Process RGB frame to depth map"""

        # Convert to PIL Image
        if rgb_frame.dtype == np.float32 or rgb_frame.dtype == np.float64:
            rgb_frame = (rgb_frame * 255).astype(np.uint8)

        # Handle RGBA -> RGB
        if rgb_frame.shape[2] == 4:
            rgb_frame = rgb_frame[:, :, :3]

        # Convert BGR to RGB if needed (OpenCV convention)
        rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)

        # Resize to configured resolution for faster processing
        target_width = self.config.get('processing', 'input_width', default=1024)
        target_height = self.config.get('processing', 'input_height', default=1024)

        # Store original size for later upscaling
        original_height, original_width = rgb_frame.shape[:2]

        # Resize if different from target
        if original_width != target_width or original_height != target_height:
            rgb_frame = cv2.resize(rgb_frame, (target_width, target_height), interpolation=cv2.INTER_AREA)

        image = Image.fromarray(rgb_frame)

        # Transform and run inference
        with torch.no_grad():
            image_tensor = self.transform(image)

            # Run model (infer expects transformed image, no batch dimension)
            prediction = self.model.infer(image_tensor, f_px=None)

            # Extract depth and focal length
            depth = prediction['depth'].squeeze().cpu().numpy()
            focallength_px = prediction.get('focallength_px')
            if focallength_px is not None:
                focallength_px = focallength_px.item()

        return depth, focallength_px, (original_width, original_height)

    def _colorize_depth(self, depth: np.ndarray, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Convert depth map to colored visualization"""

        # Normalize
        if self.config.get('processing', 'output', 'normalize', default=True):
            depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        else:
            depth_normalized = depth

        # Invert if requested
        if self.config.get('processing', 'output', 'invert', default=False):
            depth_normalized = 1.0 - depth_normalized

        # Apply gamma correction
        gamma = self.config.get('processing', 'output', 'gamma', default=1.0)
        if gamma != 1.0:
            depth_normalized = np.power(depth_normalized, gamma)

        # Convert to uint8
        depth_uint8 = (depth_normalized * 255).astype(np.uint8)

        # Apply colormap
        colormap_name = self.config.get('processing', 'output', 'colormap', default='turbo')
        colormap_dict = {
            'viridis': cv2.COLORMAP_VIRIDIS,
            'plasma': cv2.COLORMAP_PLASMA,
            'inferno': cv2.COLORMAP_INFERNO,
            'magma': cv2.COLORMAP_MAGMA,
            'turbo': cv2.COLORMAP_TURBO,
            'jet': cv2.COLORMAP_JET,
            'gray': None
        }

        colormap = colormap_dict.get(colormap_name, cv2.COLORMAP_TURBO)

        if colormap is None:
            # Grayscale
            colored = cv2.cvtColor(depth_uint8, cv2.COLOR_GRAY2RGB)
        else:
            colored = cv2.applyColorMap(depth_uint8, colormap)

        # Resize back to original size if needed
        if target_size is not None:
            colored = cv2.resize(colored, target_size, interpolation=cv2.INTER_LINEAR)

        return colored

    def _create_output_frame(self, rgb_frame: np.ndarray, depth: np.ndarray,
                            focal_length: Optional[float], original_size: Tuple[int, int]) -> np.ndarray:
        """Create 4-channel output: RGB + depth in alpha, with focal length encoded"""

        # Resize depth to original size if needed
        if depth.shape[0] != original_size[1] or depth.shape[1] != original_size[0]:
            depth_resized = cv2.resize(depth, original_size, interpolation=cv2.INTER_LINEAR)
        else:
            depth_resized = depth

        # Normalize depth to 0-255 for alpha channel
        depth_min, depth_max = depth_resized.min(), depth_resized.max()
        if depth_max > depth_min:
            depth_normalized = (depth_resized - depth_min) / (depth_max - depth_min)
        else:
            depth_normalized = np.zeros_like(depth_resized)

        depth_uint8 = (depth_normalized * 255).astype(np.uint8)

        # Ensure RGB frame is the right size and format
        if rgb_frame.shape[0] != original_size[1] or rgb_frame.shape[1] != original_size[0]:
            rgb_resized = cv2.resize(rgb_frame, original_size, interpolation=cv2.INTER_LINEAR)
        else:
            rgb_resized = rgb_frame.copy()

        # Handle RGB conversion if needed
        if rgb_resized.shape[2] == 4:
            rgb_resized = rgb_resized[:, :, :3]

        # Ensure RGB is uint8
        if rgb_resized.dtype != np.uint8:
            rgb_resized = (rgb_resized * 255).astype(np.uint8)

        # Create RGBA output
        h, w = depth_uint8.shape
        output = np.zeros((h, w, 4), dtype=np.uint8)
        output[:, :, :3] = rgb_resized  # RGB channels = original image
        output[:, :, 3] = depth_uint8   # Alpha channel = depth

        # Encode metadata in top-left corner (3x3 pixel block for robustness)
        # Store focal length, depth_min, depth_max
        if focal_length is not None:
            # Normalize focal length to 0-255 range (typical range 500-2000)
            focal_norm = int(np.clip((focal_length - 500) / 1500 * 255, 0, 255))
        else:
            focal_norm = 128  # Default middle value

        # Encode in top-left 3x3 block
        # R = focal length, G = depth_min scale, B = depth_max scale
        depth_min_norm = int(np.clip(depth_min / 100 * 255, 0, 255))
        depth_max_norm = int(np.clip(depth_max / 100 * 255, 0, 255))

        output[0:3, 0:3, 0] = focal_norm      # R = focal length
        output[0:3, 0:3, 1] = depth_min_norm  # G = depth min
        output[0:3, 0:3, 2] = depth_max_norm  # B = depth max

        return output

    def _warmup(self):
        """Warmup the model with dummy data"""
        warmup_frames = self.config.get('performance', 'warmup_frames', default=3)
        width = self.config.get('processing', 'input_width', default=1024)
        height = self.config.get('processing', 'input_height', default=1024)

        self.logger.info(f"Warming up model with {warmup_frames} frames...")

        for i in range(warmup_frames):
            dummy_frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            _, _, _ = self._process_depth(dummy_frame)

        self.logger.info("Warmup complete")

    def _log_fps(self):
        """Log FPS periodically"""
        log_interval = self.config.get('performance', 'log_interval', default=30)

        if self.frame_count % log_interval == 0:
            current_time = time.time()
            elapsed = current_time - self.last_time
            fps = log_interval / elapsed if elapsed > 0 else 0

            self.logger.info(f"Processed {self.frame_count} frames | FPS: {fps:.2f}")

            self.last_time = current_time

    def _signal_handler(self, sig, frame):
        """Handle shutdown signals"""
        self.logger.info("Shutdown signal received")
        self.running = False

    def run(self):
        """Main server loop"""
        output_mode = self.config.get('server', 'output_mode', default='socket')

        # Setup outputs
        if output_mode in ['socket', 'both']:
            self.socket_server = self._setup_socket_server()

        if output_mode in ['syphon', 'both'] and SYPHON_AVAILABLE:
            syphon_enabled = self.config.get('server', 'syphon', 'enabled', default=True)
            if syphon_enabled:
                self.syphon_server, self.syphon_texture = self._setup_syphon_server()

        # Warmup
        self._warmup()

        self.logger.info("Server ready - Processing frames...")
        self.last_time = time.time()

        try:
            while self.running:
                # Receive frame from TouchDesigner
                if self.socket_server:
                    rgb_frame = self._receive_frame(self.socket_server)
                    if rgb_frame is None:
                        self.logger.warning("Connection closed")
                        break
                else:
                    # Syphon-only mode would need different input mechanism
                    self.logger.error("Syphon-only mode not implemented - need Socket for input")
                    break

                # Process depth
                depth, focal_length, original_size = self._process_depth(rgb_frame)

                # Create output: RGB from original image + depth in alpha
                output_frame = self._create_output_frame(rgb_frame, depth, focal_length, original_size)

                # Send via Socket
                if self.socket_server:
                    if not self._send_frame(self.socket_server, output_frame):
                        self.logger.warning("Failed to send frame")
                        break

                # Send via Syphon
                if self.syphon_server and self.syphon_texture:
                    # Convert to BGRA for Syphon (flip Y for Syphon coordinate system)
                    syphon_frame = cv2.flip(output_frame, 0)
                    syphon_bgra = cv2.cvtColor(syphon_frame, cv2.COLOR_RGBA2BGRA)

                    copy_image_to_mtl_texture(syphon_bgra, self.syphon_texture)
                    self.syphon_server.publish_frame_texture(self.syphon_texture)

                self.frame_count += 1
                self._log_fps()

        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")

        except Exception as e:
            self.logger.error(f"Server error: {e}", exc_info=True)

        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup resources"""
        self.logger.info(f"Shutting down - Processed {self.frame_count} total frames")

        if self.socket_server:
            try:
                self.socket_server.close()
            except:
                pass

        if self.syphon_server:
            # Syphon cleanup happens automatically
            pass

        # Clear GPU cache
        if self.device.type == 'mps':
            torch.mps.empty_cache()
        elif self.device.type == 'cuda':
            torch.cuda.empty_cache()

        self.logger.info("Cleanup complete")


def main():
    """Entry point"""
    print("=" * 60)
    print("Depth Pro Server for TouchDesigner")
    print("=" * 60)

    # Load config
    config_path = Path(__file__).parent / "config.yaml"
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)

    config = Config(str(config_path))

    # Create and run server
    server = DepthProServer(config)
    server.run()


if __name__ == "__main__":
    main()
