"""
TouchDesigner Script TOP - Depth Pro Client
Connects to depth_pro_server.py and receives depth maps

Instructions:
1. Add this script to a Script TOP in TouchDesigner
2. Set the Script TOP format to match your input resolution
3. Connect an input TOP (video, camera, etc.)
4. Toggle 'Active' parameter to start/stop the server
"""

import socket
import struct
import pickle
import subprocess
import time
import threading
import queue
import os
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# Server settings (should match config.yaml)
SERVER_HOST = "localhost"
SERVER_PORT = 9995

# Project paths
PROJECT_ROOT = Path("/Users/jarrett/Projects/UCSB/Vision/touchdesigner-apple-depthpro")
SERVER_SCRIPT = PROJECT_ROOT / "depth_pro_server.py"
VENV_PYTHON = PROJECT_ROOT / "venv" / "bin" / "python"

# Connection settings
CONNECTION_RETRIES = 60
CONNECTION_RETRY_DELAY = 2.0
SOCKET_TIMEOUT = 30.0

# Queue settings
SEND_QUEUE_SIZE = 2
RECEIVE_QUEUE_SIZE = 2

# ============================================================================
# GLOBAL STATE
# ============================================================================

# Server process
server_process = None
server_log_file = None

# Communication
sock = None
send_queue = queue.Queue(maxsize=SEND_QUEUE_SIZE)
receive_queue = queue.Queue(maxsize=RECEIVE_QUEUE_SIZE)

# Threading
send_thread = None
receive_thread = None
threads_running = False

# Status tracking
send_count = 0
receive_count = 0
last_receive_time = 0
_latest_output = None

# ============================================================================
# PARAMETER SETUP
# ============================================================================

def onSetupParameters(scriptOp):
	"""Setup custom parameters"""
	page = scriptOp.appendCustomPage('Depth Pro')

	# Active toggle
	page.appendToggle('Active', label='Active')

	# Status displays
	page.appendStr('Status', label='Status')
	scriptOp.par.Status.readOnly = True

	page.appendStr('Serverstatus', label='Server Status')
	scriptOp.par.Serverstatus.readOnly = True

	# Counters
	page.appendInt('Sendcount', label='Frames Sent')
	scriptOp.par.Sendcount.readOnly = True

	page.appendInt('Receivecount', label='Frames Received')
	scriptOp.par.Receivecount.readOnly = True

	# FPS display
	page.appendFloat('Fps', label='FPS')
	scriptOp.par.Fps.readOnly = True

# ============================================================================
# SERVER MANAGEMENT
# ============================================================================

def start_server():
	"""Start the external server process"""
	global server_process, server_log_file

	if server_process is not None:
		return

	# Check if venv Python exists
	if not VENV_PYTHON.exists():
		print(f"ERROR: Virtual environment not found at {VENV_PYTHON}")
		print("Please run setup.sh first")
		return

	# Open log file
	log_path = PROJECT_ROOT / "depth_pro_server.log"
	server_log_file = open(log_path, 'w', buffering=1)

	# Start server process
	cmd = [str(VENV_PYTHON), '-u', str(SERVER_SCRIPT)]

	print(f"Starting server: {' '.join(cmd)}")

	server_process = subprocess.Popen(
		cmd,
		stdout=server_log_file,
		stderr=server_log_file,
		bufsize=1
	)

	print(f"Server started (PID: {server_process.pid})")
	print(f"Logs: {log_path}")

def stop_server():
	"""Stop the external server process"""
	global server_process, server_log_file

	if server_process is None:
		return

	print("Stopping server...")

	try:
		server_process.terminate()
		server_process.wait(timeout=5)
	except subprocess.TimeoutExpired:
		print("Server did not terminate gracefully, forcing...")
		server_process.kill()

	if server_log_file:
		server_log_file.close()
		server_log_file = None

	server_process = None
	print("Server stopped")

	# Wait for port to be released
	time.sleep(1)

# ============================================================================
# COMMUNICATION THREADS
# ============================================================================

def send_worker():
	"""Background thread for sending frames to server"""
	global sock, threads_running, send_count

	# Connect to server with retries
	connected = False
	for attempt in range(CONNECTION_RETRIES):
		if not threads_running:
			return

		try:
			print(f"Connection attempt {attempt + 1}/{CONNECTION_RETRIES}...")

			# Create fresh socket
			sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
			sock.settimeout(SOCKET_TIMEOUT)

			# Try to connect
			sock.connect((SERVER_HOST, SERVER_PORT))

			print(f"Connected to server at {SERVER_HOST}:{SERVER_PORT}")
			connected = True
			break

		except (ConnectionRefusedError, OSError) as e:
			if sock:
				sock.close()
				sock = None

			if attempt < CONNECTION_RETRIES - 1:
				time.sleep(CONNECTION_RETRY_DELAY)
			else:
				print(f"Failed to connect after {CONNECTION_RETRIES} attempts")
				return

	if not connected:
		return

	# Send frames
	try:
		while threads_running:
			try:
				# Get frame from queue (blocking with timeout)
				frame_data = send_queue.get(timeout=0.1)

				if frame_data is None:  # Stop signal
					break

				# Serialize
				serialized = pickle.dumps(frame_data)
				size = len(serialized)

				# Send size header + data
				sock.sendall(struct.pack('!I', size))
				sock.sendall(serialized)

				send_count += 1

			except queue.Empty:
				continue

			except Exception as e:
				print(f"Send error: {e}")
				break

	finally:
		if sock:
			try:
				sock.close()
			except:
				pass
		sock = None

def receive_worker():
	"""Background thread for receiving depth maps from server"""
	global sock, threads_running, receive_count, last_receive_time, _latest_output

	# Wait for socket to be ready
	while threads_running and sock is None:
		time.sleep(0.1)

	if not threads_running or sock is None:
		return

	try:
		while threads_running:
			try:
				# Receive size header (4 bytes)
				size_data = b''
				while len(size_data) < 4:
					chunk = sock.recv(4 - len(size_data))
					if not chunk:
						print("Server disconnected")
						return
					size_data += chunk

				frame_size = struct.unpack('!I', size_data)[0]

				# Receive frame data
				frame_data = b''
				while len(frame_data) < frame_size:
					chunk = sock.recv(min(frame_size - len(frame_data), 8192))
					if not chunk:
						print("Server disconnected")
						return
					frame_data += chunk

				# Deserialize
				depth_frame = pickle.loads(frame_data)

				# Put in queue (drop old if full)
				try:
					receive_queue.put_nowait(depth_frame)
				except queue.Full:
					# Drop oldest frame
					try:
						receive_queue.get_nowait()
					except queue.Empty:
						pass
					receive_queue.put_nowait(depth_frame)

				receive_count += 1
				last_receive_time = time.time()

			except OSError as e:
				if threads_running:
					print(f"Receive error: {e}")
				break

			except Exception as e:
				print(f"Receive error: {e}")
				break

	finally:
		pass

# ============================================================================
# THREAD MANAGEMENT
# ============================================================================

def start_threads():
	"""Start communication threads"""
	global send_thread, receive_thread, threads_running

	if threads_running:
		return

	threads_running = True

	# Start send thread
	send_thread = threading.Thread(target=send_worker, daemon=True)
	send_thread.start()

	# Small delay before starting receive thread
	time.sleep(0.3)

	# Start receive thread
	receive_thread = threading.Thread(target=receive_worker, daemon=True)
	receive_thread.start()

	print("Communication threads started")

def stop_threads():
	"""Stop communication threads"""
	global threads_running, send_queue, sock

	if not threads_running:
		return

	print("Stopping communication threads...")
	threads_running = False

	# Send stop signal
	try:
		send_queue.put_nowait(None)
	except queue.Full:
		pass

	# Close socket
	if sock:
		try:
			sock.close()
		except:
			pass

	# Reset counters
	global send_count, receive_count
	send_count = 0
	receive_count = 0

	print("Communication threads stopped")

# ============================================================================
# MAIN COOK FUNCTION
# ============================================================================

def onCook(scriptOp):
	"""Main processing function called every frame"""
	global _latest_output

	# Check if active
	active = scriptOp.par.Active.eval()

	# Update status
	if active:
		if server_process is None:
			scriptOp.par.Status = "Starting server..."
			scriptOp.par.Serverstatus = "Starting"
			start_server()
			time.sleep(2)  # Give server time to start
			start_threads()
		elif not threads_running:
			scriptOp.par.Status = "Starting threads..."
			scriptOp.par.Serverstatus = "Running"
			start_threads()
		elif sock is None:
			scriptOp.par.Status = "Connecting..."
			scriptOp.par.Serverstatus = "Running"
		else:
			scriptOp.par.Status = "Processing"
			scriptOp.par.Serverstatus = "Connected"
	else:
		if threads_running or server_process is not None:
			scriptOp.par.Status = "Stopping..."
			scriptOp.par.Serverstatus = "Stopping"
			stop_threads()
			stop_server()
			_latest_output = None
		else:
			scriptOp.Status = "Inactive"
			scriptOp.Serverstatus = "Stopped"

	# Update counters
	scriptOp.par.Sendcount = send_count
	scriptOp.par.Receivecount = receive_count

	# Calculate FPS
	if receive_count > 0 and last_receive_time > 0:
		fps = 1.0 / max(time.time() - last_receive_time, 0.001)
		scriptOp.par.Fps = min(fps, 60.0)  # Cap at 60
	else:
		scriptOp.par.Fps = 0.0

	# Process if active and connected
	if active and threads_running and sock is not None:
		# Get input frame
		inputTop = scriptOp.inputs[0] if len(scriptOp.inputs) > 0 else None

		if inputTop:
			# Get frame as numpy array
			input_array = inputTop.numpyArray()

			# Convert RGBA float32 to RGB uint8
			if input_array.dtype != 'uint8':
				input_array = (input_array * 255).astype('uint8')

			# Send to queue (non-blocking, drop if full)
			try:
				send_queue.put_nowait(input_array)
			except queue.Full:
				# Drop oldest frame
				try:
					send_queue.get_nowait()
				except queue.Empty:
					pass
				send_queue.put_nowait(input_array)

		# Drain receive queue (get latest)
		while not receive_queue.empty():
			try:
				_latest_output = receive_queue.get_nowait()
			except queue.Empty:
				break

		# Display output
		if _latest_output is not None:
			# Convert RGB uint8 to RGBA float32
			if _latest_output.shape[2] == 3:
				# Add alpha channel
				import numpy as np
				alpha = np.ones((_latest_output.shape[0], _latest_output.shape[1], 1), dtype=_latest_output.dtype) * 255
				output_rgba = np.concatenate([_latest_output, alpha], axis=2)
			else:
				output_rgba = _latest_output

			# Convert to float32
			if output_rgba.dtype != 'float32':
				output_rgba = output_rgba.astype('float32') / 255.0

			# Copy to Script TOP output
			scriptOp.copyNumpyArray(output_rgba)
			return

	# Passthrough input if not active or no output yet
	if not active or _latest_output is None:
		inputTop = scriptOp.inputs[0] if len(scriptOp.inputs) > 0 else None
		if inputTop:
			scriptOp.copy(inputTop)

# ============================================================================
# CLEANUP
# ============================================================================

def onDestroy(scriptOp):
	"""Cleanup when Script TOP is destroyed"""
	stop_threads()
	stop_server()
