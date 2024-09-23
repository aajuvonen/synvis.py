import pyaudio
import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from collections import deque
import math

# Constants
WIDTH, HEIGHT = 1280, 720
GRID_SIZE = 12
AUDIO_RATE = 22050
CHUNK = 512  # Buffer size
SMOOTHING_FACTOR = 4  # Smoothing over n frames

# Initialize PyAudio
p = pyaudio.PyAudio()

# Function to list available audio devices
def list_audio_devices():
    info = p.get_host_api_info_by_index(0)
    num_devices = info.get('deviceCount')
    devices = []
    for i in range(0, num_devices):
        device_info = p.get_device_info_by_host_api_device_index(0, i)
        devices.append(device_info)
        print(f"{i}: {device_info.get('name')}")
    return devices

# Select audio input device
def select_audio_device():
    devices = list_audio_devices()
    device_index = int(input("Select audio device by index: "))
    return device_index

# Initialize PyGame and OpenGL
def init_pygame():
    pygame.init()
    pygame.display.set_mode((WIDTH, HEIGHT), DOUBLEBUF | OPENGL | RESIZABLE)
    resize_viewport(WIDTH, HEIGHT)

def resize_viewport(width, height):
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(75, (width / height), 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

# Create 3D mesh grid
def create_grid(size):
    grid = []
    for x in range(-size, size):
        row = []
        for y in range(-size, size):
            row.append([x / (size / 2), y / (size / 2), 0])
        grid.append(row)
    return grid

# Draw 3D mesh grid as wireframe
def draw_grid(grid, frame_count, intensity):
    # Adjust color based on intensity
    color_shift = (math.sin(frame_count * 0.02) + 1) / 2 * intensity  # Cycling between 0 and intensity
    glColor3f(0.3, color_shift, 1.0 - color_shift)  # Adjust color dynamically

    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)  # Set to wireframe mode
    glBegin(GL_QUADS)
    for i in range(len(grid) - 1):
        for j in range(len(grid[i]) - 1):
            for di, dj in [(0, 0), (1, 0), (1, 1), (0, 1)]:
                glVertex3fv(grid[i + di][j + dj])
    glEnd()
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)  # Reset to fill mode

# Deform grid based on smoothed audio spectrum
def deform_grid(grid, smoothed_spectrum, frame_count, intensity):
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            # Base deformation on audio spectrum
            z = smoothed_spectrum[(i + j) % len(smoothed_spectrum)] * 0.005 * intensity
            
            # Add a secondary wave motion, also dependent on intensity
            z += intensity * 0.05 * math.sin((i + frame_count) * 0.3) * math.cos((j + frame_count) * 0.3)
            
            grid[i][j][2] = z  # Update the z-coordinate

# Smooth the spectrum over the last few frames
def smooth_spectrum(spectrum_history):
    smoothed_spectrum = np.mean(spectrum_history, axis=0)
    return smoothed_spectrum

# Calculate audio intensity
def calculate_intensity(spectrum):
    return np.clip(np.mean(spectrum) / 4, 0, 1)  # Normalize intensity to [0, 1]

# Process audio and return spectrum
def get_spectrum(stream):
    try:
        data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.float32)
        spectrum = np.fft.fft(data)
        return np.abs(spectrum[:GRID_SIZE])
    except IOError:
        print("Buffer overflowed, skipping this chunk.")
        return np.zeros(GRID_SIZE)

# Generate lightning bolts
def generate_lightning(start, end, branch_factor=0.3, depth=4, intensity=1):
    if depth == 0:
        return [(start, end)]
    
    mid = (np.array(start) + np.array(end)) / 2
    
    # Introduce some random displacement to create the jagged lightning effect
    displacement = np.random.randn(3) * branch_factor * intensity
    mid += displacement
    
    # Recursively generate branches
    left_branch = generate_lightning(start, mid, branch_factor, depth-1, intensity)
    right_branch = generate_lightning(mid, end, branch_factor, depth-1, intensity)
    
    return left_branch + right_branch

# Draw lightning bolts with aggressive tapering
def draw_lightning(lightning_segments):
    total_segments = len(lightning_segments)
    max_width = 4.0  # Start with a thicker line

    for i, segment in enumerate(lightning_segments):
        # Aggressively reduce the width based on position in the bolt
        width = max(1.0, max_width * (1.0 - (i / total_segments) ** 3))  # Sharper tapering
        glLineWidth(width)
        
        glColor3f(0.5, 1.0, 0.5)
        glBegin(GL_LINES)
        glVertex3fv(segment[0])
        glVertex3fv(segment[1])
        glEnd()

    # Reset line width to default
    glLineWidth(1.0)

# Draw waveform in the background
def draw_waveform(data):
    glPushMatrix()
    glLoadIdentity()
    glTranslatef(0.0, 0.4, -2.5)  # Move waveform back
    glScalef(3.5, 0.5, 1.0)  # Scale waveform to fit the screen width
    
    glColor3f(0.0, 1.0, 0.4)  # Set waveform color
    glLineWidth(0.5)
    
    glBegin(GL_LINE_STRIP)
    for i in range(len(data)):
        x = (i / len(data)) * 2.0 - 1.0  # X coordinates from -1.0 to 1.0
        y = data[i] * 0.5  # Scale Y coordinates
        glVertex2f(x, y)
    glEnd()

    glPopMatrix()

# Main loop
def main():
    # Select audio input device
    device_index = select_audio_device()
    
    # Open audio stream
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=AUDIO_RATE,
                    input=True,
                    input_device_index=device_index,
                    frames_per_buffer=CHUNK)

    init_pygame()
    grid = create_grid(GRID_SIZE)
    
    # Initialize deque to store recent spectrums for smoothing
    spectrum_history = deque(maxlen=SMOOTHING_FACTOR)

    frame_count = 0  # Frame counter for effects

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                stream.stop_stream()
                stream.close()
                p.terminate()
                return
            elif event.type == VIDEORESIZE:
                resize_viewport(event.w, event.h)

        spectrum = get_spectrum(stream)
        spectrum_history.append(spectrum)
        
        if len(spectrum_history) == SMOOTHING_FACTOR:
            smoothed_spectrum = smooth_spectrum(spectrum_history)
            intensity = calculate_intensity(smoothed_spectrum)
            deform_grid(grid, smoothed_spectrum, frame_count, intensity)
        else:
            intensity = 0

        # Process raw audio for waveform display
        raw_audio = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.float32)
        normalized_audio = raw_audio / np.max(np.abs(raw_audio))  # Normalize audio data for waveform display
        
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Draw waveform in the background
        draw_waveform(normalized_audio)
        
        # Rotate the grid slowly, affected by intensity
        glLoadIdentity()
        glTranslatef(0.0, 0.05, -1.5)
        glRotatef(-80, 1, 0, 0)
        glRotatef(90 * math.sin(frame_count / 50) * 0.1 * intensity, 0, 0, 1)
        
        draw_grid(grid, frame_count, intensity)
        
        # Trigger lightning effect based on intensity
        if intensity > 0.97 and np.random.random() > 0.95:  # Adjust threshold for lightning
            lightning_start = [0, np.random.uniform(-1, 1), 2]
            lightning_end = [np.random.uniform(-1, 1), np.random.uniform(-1, 1), -1.5]
            lightning_segments = generate_lightning(lightning_start, lightning_end, intensity=intensity)
            draw_lightning(lightning_segments)
        
        pygame.display.flip()
        pygame.time.wait(10)
        frame_count += 1

if __name__ == "__main__":
    main()
