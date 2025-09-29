

import cv2
import numpy as np
import os
from datetime import datetime
import sys

class WebcamCapture:
    def __init__(self):
        self.current_camera = 0
        self.available_cameras = self.find_cameras()
        self.recording = False
        self.video_writer = None
        self.current_filter = 'normal'
        self.resolution_presets = {
            '480p': (640, 480),
            '720p': (1280, 720),
            '1080p': (1920, 1080)
        }
        self.current_resolution = '720p'
        
    def find_cameras(self):
        """Find all available cameras"""
        available_cameras = []
        for i in range(5):  # Check first 5 indexes
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append(i)
                cap.release()
        return available_cameras
    
    def apply_filter(self, frame):
        """Apply various filters to the frame"""
        if self.current_filter == 'normal':
            return frame
        elif self.current_filter == 'grayscale':
            return cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        elif self.current_filter == 'blur':
            return cv2.GaussianBlur(frame, (15, 15), 0)
        elif self.current_filter == 'edge':
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        elif self.current_filter == 'sepia':
            kernel = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]])
            return cv2.transform(frame, kernel)
        return frame
    
    def capture_image(self, frame):
        """Save the current frame as an image"""
        if not os.path.exists('captures'):
            os.makedirs('captures')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'captures/image_{timestamp}.jpg'
        cv2.imwrite(filename, frame)
        print(f"Image saved as {filename}")
    
    def toggle_recording(self, frame):
        """Toggle video recording"""
        if not self.recording:
            if not os.path.exists('recordings'):
                os.makedirs('recordings')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'recordings/video_{timestamp}.avi'
            frame_height, frame_width = frame.shape[:2]
            self.video_writer = cv2.VideoWriter(
                filename,
                cv2.VideoWriter_fourcc(*'XVID'),
                20.0,
                (frame_width, frame_height)
            )
            self.recording = True
            print("Started recording...")
        else:
            self.video_writer.release()
            self.recording = False
            print("Recording saved!")
    
    def set_resolution(self, cap):
        """Set the resolution of the webcam"""
        width, height = self.resolution_presets[self.current_resolution]
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    def run(self):
        """Main loop for capturing and displaying webcam feed"""
        if not self.available_cameras:
            print("No cameras found!")
            return
        
        cap = cv2.VideoCapture(self.current_camera)
        self.set_resolution(cap)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("\nWebcam Controls:")
        print("ESC - Quit")
        print("SPACE - Capture image")
        print("R - Start/Stop recording")
        print("F - Change filter")
        print("C - Switch camera")
        print("Q - Change resolution")
        print(f"Available cameras: {self.available_cameras}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Can't receive frame. Exiting...")
                break
            
            # Apply current filter
            filtered_frame = self.apply_filter(frame)
            
            # Add status indicators
            status_text = f"Camera: {self.current_camera} | Filter: {self.current_filter}"
            status_text += f" | Resolution: {self.current_resolution}"
            if self.recording:
                status_text += " | RECORDING"
                self.video_writer.write(filtered_frame)
            
            cv2.putText(filtered_frame, status_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display the frame
            cv2.imshow('Webcam Feed', filtered_frame)
            
            # Handle keypresses
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                break
            elif key == 32:  # SPACE
                self.capture_image(filtered_frame)
            elif key == ord('r'):
                self.toggle_recording(filtered_frame)
            elif key == ord('f'):
                filters = ['normal', 'grayscale', 'blur', 'edge', 'sepia']
                current_idx = filters.index(self.current_filter)
                self.current_filter = filters[(current_idx + 1) % len(filters)]
            elif key == ord('c') and len(self.available_cameras) > 1:
                current_idx = self.available_cameras.index(self.current_camera)
                self.current_camera = self.available_cameras[(current_idx + 1) % len(self.available_cameras)]
                cap.release()
                cap = cv2.VideoCapture(self.current_camera)
                self.set_resolution(cap)
            elif key == ord('q'):
                resolutions = list(self.resolution_presets.keys())
                current_idx = resolutions.index(self.current_resolution)
                self.current_resolution = resolutions[(current_idx + 1) % len(resolutions)]
                self.set_resolution(cap)
        
        # Cleanup
        if self.recording:
            self.video_writer.release()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    webcam = WebcamCapture()
    webcam.run()