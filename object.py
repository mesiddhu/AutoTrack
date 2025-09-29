import cv2
import numpy as np
import time
import math
import json
import os
from datetime import datetime
import logging

class RailwayTrackDefectDetector:
    def __init__(self):
        # Detection parameters
        self.crack_threshold = 50
        self.bend_threshold = 0.1
        self.flaw_area_threshold = 100
        self.weak_spot_variance_threshold = 800
        
        # AUTO-SAVE parameters
        self.confidence_threshold = 0.4  # Lower threshold for more sensitive detection
        self.save_interval = 1.0  # Minimum seconds between saves to avoid spam
        self.last_save_time = 0
        
        # Track detection history
        self.defect_history = []
        self.frame_count = 0
        
        # Color codes for different defect types
        self.defect_colors = {
            'crack': (0, 0, 255),      # Red
            'bend': (0, 255, 255),     # Yellow
            'flaw': (255, 0, 255),     # Magenta
            'weak_spot': (0, 165, 255) # Orange
        }
        
        # Defect counters
        self.defect_counts = {
            'crack': 0,
            'bend': 0,
            'flaw': 0,
            'weak_spot': 0
        }
        
        # Performance metrics
        self.processing_times = []
        self.confidence_scores = []
        self.saved_photos_count = 0
        
        # Setup logging
        self.setup_logging()
        
        # Camera session tracking
        self.session_start_time = None
        self.total_session_time = 0
        
    def setup_logging(self):
        """Setup logging for the detection system"""
        if not os.path.exists('logs'):
            os.makedirs('logs')
            
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'logs/track_detection_{datetime.now().strftime("%Y%m%d")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def preprocess_frame(self, frame):
        """Enhanced preprocessing with multiple filtering techniques"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to preserve edges while reducing noise
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(bilateral, (5, 5), 0)
        
        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(blurred)
        
        # Apply morphological operations to clean up
        kernel = np.ones((3,3), np.uint8)
        cleaned = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
        
        # Additional sharpening filter
        sharpening_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(cleaned, -1, sharpening_kernel)
        
        return gray, enhanced, cleaned, sharpened
    
    def detect_cracks(self, processed_frame):
        """Enhanced crack detection with multiple edge detection methods"""
        cracks = []
        
        # Method 1: Canny edge detection
        edges_canny = cv2.Canny(processed_frame, 50, 150, apertureSize=3)
        
        # Method 2: Sobel edge detection
        sobelx = cv2.Sobel(processed_frame, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(processed_frame, cv2.CV_64F, 0, 1, ksize=3)
        edges_sobel = np.sqrt(sobelx**2 + sobely**2).astype(np.uint8)
        edges_sobel = cv2.threshold(edges_sobel, 100, 255, cv2.THRESH_BINARY)[1]
        
        # Combine edge detection methods
        combined_edges = cv2.bitwise_or(edges_canny, edges_sobel)
        
        # Use HoughLinesP to detect line segments
        lines = cv2.HoughLinesP(combined_edges, 1, np.pi/180, threshold=30, 
                               minLineLength=20, maxLineGap=15)
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                
                # Check if line could be a crack
                if length > self.crack_threshold:
                    angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
                    
                    # Calculate confidence based on length and angle
                    confidence = min(0.9, length / 200.0)
                    
                    # Cracks often have irregular angles
                    if not (abs(angle) < 10 or abs(angle) > 80):
                        severity = 'critical' if length > 150 else ('high' if length > 100 else 'medium')
                        
                        cracks.append({
                            'type': 'crack',
                            'points': [(x1, y1), (x2, y2)],
                            'length': length,
                            'angle': angle,
                            'severity': severity,
                            'confidence': confidence
                        })
        
        return cracks
    
    def detect_bends(self, processed_frame):
        """Enhanced bend detection with improved curvature analysis"""
        # Apply adaptive threshold
        binary = cv2.adaptiveThreshold(processed_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bends = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 300:
                # Approximate contour
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Calculate curvature using arc length and chord length
                arc_length = cv2.arcLength(contour, True)
                if len(approx) >= 4:
                    # Calculate bounding rect
                    rect = cv2.minAreaRect(contour)
                    width, height = rect[1]
                    chord_length = max(width, height)
                    
                    if chord_length > 0:
                        curvature_ratio = arc_length / chord_length
                        
                        if curvature_ratio > 1.2:  # Indicates significant curvature
                            # Get extreme points
                            leftmost = tuple(contour[contour[:,:,0].argmin()][0])
                            rightmost = tuple(contour[contour[:,:,0].argmax()][0])
                            topmost = tuple(contour[contour[:,:,1].argmin()][0])
                            bottommost = tuple(contour[contour[:,:,1].argmax()][0])
                            
                            confidence = min(0.9, (curvature_ratio - 1.0) / 2.0)
                            severity = 'critical' if curvature_ratio > 2.0 else 'moderate'
                            
                            bends.append({
                                'type': 'bend',
                                'points': [leftmost, rightmost, topmost, bottommost],
                                'curvature_ratio': curvature_ratio,
                                'severity': severity,
                                'confidence': confidence,
                                'area': area
                            })
        
        return bends
    
    def detect_flaws(self, original_frame, processed_frame):
        """Enhanced flaw detection with texture and color analysis"""
        # Method 1: Texture analysis using local binary patterns
        def local_binary_pattern(img, radius=3, n_points=8):
            """Simple LBP implementation"""
            rows, cols = img.shape
            lbp = np.zeros((rows, cols), dtype=np.uint8)
            
            for i in range(radius, rows - radius):
                for j in range(radius, cols - radius):
                    center = img[i, j]
                    code = 0
                    for k in range(n_points):
                        angle = 2 * np.pi * k / n_points
                        x = int(i + radius * np.cos(angle))
                        y = int(j + radius * np.sin(angle))
                        if 0 <= x < rows and 0 <= y < cols:
                            if img[x, y] >= center:
                                code |= (1 << k)
                    lbp[i, j] = code
            return lbp
        
        # Calculate LBP
        lbp = local_binary_pattern(processed_frame)
        lbp_var = cv2.Laplacian(lbp, cv2.CV_64F).var()
        
        # Method 2: Color variance analysis
        gray_var = cv2.Laplacian(processed_frame, cv2.CV_64F).var()
        
        # Method 3: Local variance
        kernel = np.ones((7, 7), np.float32) / 49
        mean = cv2.filter2D(processed_frame.astype(np.float32), -1, kernel)
        variance = cv2.filter2D((processed_frame.astype(np.float32) - mean)**2, -1, kernel)
        
        # Adaptive threshold for flaw detection
        threshold = np.percentile(variance, 85)
        _, flaw_mask = cv2.threshold(variance, threshold, 255, cv2.THRESH_BINARY)
        flaw_mask = flaw_mask.astype(np.uint8)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        flaw_mask = cv2.morphologyEx(flaw_mask, cv2.MORPH_CLOSE, kernel)
        flaw_mask = cv2.morphologyEx(flaw_mask, cv2.MORPH_OPEN, kernel)
        
        # Find flaw contours
        contours, _ = cv2.findContours(flaw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        flaws = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.flaw_area_threshold:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h
                
                # Calculate confidence based on area and texture variance
                confidence = min(0.9, area / 1000.0 * (gray_var / 100.0))
                severity = 'critical' if area > 800 else ('high' if area > 400 else 'medium')
                
                flaws.append({
                    'type': 'flaw',
                    'bbox': (x, y, w, h),
                    'area': area,
                    'aspect_ratio': aspect_ratio,
                    'severity': severity,
                    'confidence': confidence,
                    'texture_variance': gray_var
                })
        
        return flaws
    
    def detect_weak_spots(self, processed_frame):
        """Enhanced weak spot detection with multiple analysis methods"""
        # Method 1: Gradient analysis
        grad_x = cv2.Sobel(processed_frame, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(processed_frame, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Method 2: Structure tensor analysis
        Ixx = grad_x * grad_x
        Iyy = grad_y * grad_y
        Ixy = grad_x * grad_y
        
        # Apply Gaussian smoothing
        kernel_size = 5
        Ixx = cv2.GaussianBlur(Ixx, (kernel_size, kernel_size), 1.0)
        Iyy = cv2.GaussianBlur(Iyy, (kernel_size, kernel_size), 1.0)
        Ixy = cv2.GaussianBlur(Ixy, (kernel_size, kernel_size), 1.0)
        
        # Calculate eigenvalues
        trace = Ixx + Iyy
        det = Ixx * Iyy - Ixy * Ixy
        
        # Harris corner response
        k = 0.04
        harris_response = det - k * (trace ** 2)
        
        # Find regions with low gradient and low Harris response
        low_gradient_threshold = np.percentile(gradient_magnitude, 25)
        low_harris_threshold = np.percentile(harris_response, 25)
        
        weak_mask = (gradient_magnitude < low_gradient_threshold) & \
                   (harris_response < low_harris_threshold)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        weak_mask = cv2.morphologyEx(weak_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        
        # Find weak spot contours
        contours, _ = cv2.findContours(weak_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        weak_spots = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 150:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    
                    # Calculate confidence
                    confidence = min(0.9, area / 500.0)
                    severity = 'critical' if area > 1200 else ('high' if area > 600 else 'moderate')
                    
                    weak_spots.append({
                        'type': 'weak_spot',
                        'center': (cx, cy),
                        'radius': int(radius),
                        'area': area,
                        'severity': severity,
                        'confidence': confidence
                    })
        
        return weak_spots
    
    def auto_save_defect_photo(self, frame, defects):
        """🚨 ENHANCED AUTO-SAVE with BRIGHT RED markings when defects detected"""
        current_time = time.time()
        
        # Check if enough time has passed since last save
        if current_time - self.last_save_time < self.save_interval:
            return
        
        # Filter defects by confidence threshold
        high_confidence_defects = [d for d in defects if d.get('confidence', 0) >= self.confidence_threshold]
        
        if not high_confidence_defects:
            return
        
        # Create a copy for marking
        marked_frame = frame.copy()
        
        # Mark ALL high-confidence defects with BRIGHT RED spots/markings
        red_color = (0, 0, 255)  # Bright Red in BGR
        
        for defect in high_confidence_defects:
            defect_type = defect['type']
            confidence = defect.get('confidence', 0.5)
            severity = defect.get('severity', 'unknown')
            
            if defect_type == 'crack':
                # Mark crack with THICK RED LINES and LARGE RED CIRCLES
                pt1, pt2 = defect['points']
                cv2.line(marked_frame, pt1, pt2, red_color, 8)  # Very thick red line
                cv2.circle(marked_frame, pt1, 15, red_color, -1)  # Large red filled circles
                cv2.circle(marked_frame, pt2, 15, red_color, -1)
                
                # Add red text label with larger font
                mid_x = (pt1[0] + pt2[0]) // 2
                mid_y = (pt1[1] + pt2[1]) // 2
                text = f"CRACK! {severity.upper()}"
                cv2.putText(marked_frame, text, (mid_x-80, mid_y-25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, red_color, 4)
                cv2.putText(marked_frame, f"Conf: {confidence:.2f}", (mid_x-50, mid_y+15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, red_color, 3)
            
            elif defect_type == 'bend':
                # Mark bend with LARGE RED CIRCLES and thick lines
                points = defect['points']
                for point in points:
                    cv2.circle(marked_frame, point, 20, red_color, -1)  # Very large red circles
                
                # Connect with thick red lines
                if len(points) >= 2:
                    for i in range(len(points) - 1):
                        cv2.line(marked_frame, points[i], points[i+1], red_color, 6)
                
                # Add red text
                if points:
                    text = f"BEND! {severity.upper()}"
                    cv2.putText(marked_frame, text, (points[0][0]-80, points[0][1]-30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, red_color, 4)
                    cv2.putText(marked_frame, f"Conf: {confidence:.2f}", (points[0][0]-50, points[0][1]+20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, red_color, 3)
            
            elif defect_type == 'flaw':
                # Mark flaw with THICK RED RECTANGLE and large cross
                x, y, w, h = defect['bbox']
                cv2.rectangle(marked_frame, (x, y), (x + w, y + h), red_color, 6)
                
                # Add large red cross inside rectangle
                cv2.line(marked_frame, (x, y), (x + w, y + h), red_color, 5)
                cv2.line(marked_frame, (x + w, y), (x, y + h), red_color, 5)
                
                # Add center circle
                center_x = x + w // 2
                center_y = y + h // 2
                cv2.circle(marked_frame, (center_x, center_y), 12, red_color, -1)
                
                # Add red text
                text = f"FLAW! {severity.upper()}"
                cv2.putText(marked_frame, text, (x, y-15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, red_color, 4)
                cv2.putText(marked_frame, f"Conf: {confidence:.2f}", (x, y-45), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, red_color, 3)
            
            elif defect_type == 'weak_spot':
                # Mark weak spot with THICK RED CIRCLE and large cross
                center = defect['center']
                radius = defect['radius']
                cv2.circle(marked_frame, center, radius, red_color, 6)
                cv2.circle(marked_frame, center, 12, red_color, -1)  # Large center dot
                
                # Add large red cross
                cv2.line(marked_frame, (center[0]-radius, center[1]), 
                        (center[0]+radius, center[1]), red_color, 5)
                cv2.line(marked_frame, (center[0], center[1]-radius), 
                        (center[0], center[1]+radius), red_color, 5)
                
                # Add red text
                text = f"WEAK SPOT! {severity.upper()}"
                cv2.putText(marked_frame, text, 
                           (center[0]-100, center[1]-radius-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, red_color, 4)
                cv2.putText(marked_frame, f"Conf: {confidence:.2f}", 
                           (center[0]-60, center[1]+radius+25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, red_color, 3)
        
        # Add header warning
        cv2.rectangle(marked_frame, (0, 0), (marked_frame.shape[1], 80), (0, 0, 0), -1)
        cv2.putText(marked_frame, "🚨 RAILWAY TRACK DEFECTS DETECTED! 🚨", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, red_color, 4)
        cv2.putText(marked_frame, f"Frame: {self.frame_count} | Defects: {len(high_confidence_defects)} | Time: {datetime.now().strftime('%H:%M:%S')}", 
                   (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Save the marked photo with enhanced filename
        timestamp = datetime.now()
        defect_types = list(set([d['type'] for d in high_confidence_defects]))
        defect_summary = "_".join(defect_types)
        filename = f"DEFECT_{defect_summary}_F{self.frame_count}_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
        save_path = os.path.join("detections", filename)
        
        # Make sure directory exists
        os.makedirs("detections", exist_ok=True)
        
        # Save the image with high quality
        success = cv2.imwrite(save_path, marked_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        if success:
            self.saved_photos_count += 1
            self.last_save_time = current_time
            
            # Enhanced console output
            print(f"\n🚨🚨🚨 DEFECT ALERT! 🚨🚨🚨")
            print(f"📸 AUTO-SAVED: {filename}")
            print(f"🔍 Defects found: {defect_summary.upper().replace('_', ', ')}")
            print(f"📊 Total defects: {len(high_confidence_defects)}")
            print(f"📈 Confidence levels: {[ '{:.2f}'.format(d.get('confidence', 0)) for d in high_confidence_defects ]}")
            print(f"⚠️  Severities: {[d.get('severity', 'unknown').upper() for d in high_confidence_defects]}")
            print(f"💾 Total photos saved: {self.saved_photos_count}")
            print("-" * 60)
            
            # Log the detection
            self.logger.warning(f"AUTO-SAVE: {len(high_confidence_defects)} defects detected in frame {self.frame_count}. "
                              f"Photo saved: {filename}. Types: {defect_summary}")
        else:
            print(f"❌ Failed to save detection photo: {filename}")
            self.logger.error(f"Failed to save detection photo: {filename}")
    
    def analyze_frame(self, frame):
        """Enhanced analysis with timing and confidence tracking"""
        start_time = time.time()
        self.frame_count += 1
        
        # Preprocess frame
        gray, enhanced, cleaned, sharpened = self.preprocess_frame(frame)
        
        # Detect different types of defects
        cracks = self.detect_cracks(sharpened)
        bends = self.detect_bends(enhanced)
        flaws = self.detect_flaws(frame, cleaned)
        weak_spots = self.detect_weak_spots(enhanced)
        
        # Combine all detections
        all_defects = cracks + bends + flaws + weak_spots
        
        # Update counters and confidence tracking
        frame_confidences = []
        for defect in all_defects:
            self.defect_counts[defect['type']] += 1
            if 'confidence' in defect:
                frame_confidences.append(defect['confidence'])
        
        if frame_confidences:
            avg_confidence = np.mean(frame_confidences)
            self.confidence_scores.append(avg_confidence)
        
        # Track processing time
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        # Store in history with enhanced metadata
        if all_defects:
            self.defect_history.append({
                'frame': self.frame_count,
                'timestamp': time.time(),
                'defects': all_defects,
                'processing_time': processing_time,
                'avg_confidence': np.mean(frame_confidences) if frame_confidences else 0
            })
        
        # 🚨 AUTO-SAVE DETECTION: Save photo with RED markings when defects found
        if all_defects:
            self.auto_save_defect_photo(frame.copy(), all_defects)
        
        return all_defects, enhanced
    
    def draw_detections(self, frame, defects):
        """Enhanced visualization with confidence indicators"""
        for defect in defects:
            defect_type = defect['type']
            color = self.defect_colors[defect_type]
            confidence = defect.get('confidence', 0.5)
            
            # Adjust color intensity based on confidence
            intensity = int(255 * confidence)
            adjusted_color = tuple(int(c * confidence + 50 * (1 - confidence)) for c in color)
            
            if defect_type == 'crack':
                pt1, pt2 = defect['points']
                thickness = int(2 + confidence * 3)
                cv2.line(frame, pt1, pt2, adjusted_color, thickness)
                cv2.circle(frame, pt1, 5, adjusted_color, -1)
                cv2.circle(frame, pt2, 5, adjusted_color, -1)
                
                # Enhanced label with confidence
                mid_x = (pt1[0] + pt2[0]) // 2
                mid_y = (pt1[1] + pt2[1]) // 2
                label = f"CRACK ({defect['severity']}) {confidence:.2f}"
                cv2.putText(frame, label, (mid_x, mid_y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, adjusted_color, 1)
            
            elif defect_type == 'bend':
                points = defect['points']
                for i, point in enumerate(points):
                    radius = int(6 + confidence * 4)
                    cv2.circle(frame, point, radius, adjusted_color, -1)
                
                if len(points) >= 2:
                    thickness = int(1 + confidence * 2)
                    for i in range(len(points) - 1):
                        cv2.line(frame, points[i], points[i+1], adjusted_color, thickness)
                
                if points:
                    label = f"BEND ({defect['severity']}) {confidence:.2f}"
                    cv2.putText(frame, label, (points[0][0], points[0][1] - 15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, adjusted_color, 1)
            
            elif defect_type == 'flaw':
                x, y, w, h = defect['bbox']
                thickness = int(1 + confidence * 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), adjusted_color, thickness)
                
                label = f"FLAW ({defect['severity']}) {confidence:.2f}"
                cv2.putText(frame, label, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, adjusted_color, 1)
            
            elif defect_type == 'weak_spot':
                center = defect['center']
                radius = defect['radius']
                thickness = int(1 + confidence * 2)
                cv2.circle(frame, center, radius, adjusted_color, thickness)
                cv2.circle(frame, center, 3, adjusted_color, -1)
                
                label = f"WEAK SPOT ({defect['severity']}) {confidence:.2f}"
                cv2.putText(frame, label, (center[0] - 50, center[1] - radius - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, adjusted_color, 1)
    
    def draw_statistics(self, frame):
        """Enhanced statistics with AUTO-SAVE info"""
        # Background
        cv2.rectangle(frame, (10, 10), (500, 300), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (500, 300), (255, 255, 255), 2)
        
        # Title
        cv2.putText(frame, "🚨 AUTO-SAVE DEFECT DETECTION 🚨", (15, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Statistics
        y_pos = 60
        total_defects = sum(self.defect_counts.values())
        cv2.putText(frame, f"Total Defects Found: {total_defects}", (15, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Auto-save info
        y_pos += 25
        cv2.putText(frame, f"📸 Photos Auto-Saved: {self.saved_photos_count}", (15, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        y_pos += 25
        for defect_type, count in self.defect_counts.items():
            color = self.defect_colors[defect_type]
            text = f"{defect_type.replace('_', ' ').title()}: {count}"
            cv2.putText(frame, text, (15, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            y_pos += 20
        
        # Performance metrics
        y_pos += 10
        if self.processing_times:
            avg_processing_time = np.mean(self.processing_times[-30:]) * 1000  # Last 30 frames in ms
            fps = 1.0 / np.mean(self.processing_times[-30:]) if self.processing_times else 0
            cv2.putText(frame, f"Avg Processing: {avg_processing_time:.1f}ms", (15, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            y_pos += 15
            cv2.putText(frame, f"FPS: {fps:.1f}", (15, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            y_pos += 15
        
        if self.confidence_scores:
            avg_confidence = np.mean(self.confidence_scores[-30:])  # Last 30 frames
            cv2.putText(frame, f"Avg Confidence: {avg_confidence:.3f}", (15, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            y_pos += 15
        
        # Session info
        if self.session_start_time:
            session_duration = time.time() - self.session_start_time
            minutes = int(session_duration // 60)
            seconds = int(session_duration % 60)
            cv2.putText(frame, f"Session: {minutes:02d}:{seconds:02d}", (15, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            y_pos += 15
        
        cv2.putText(frame, f"Frames: {self.frame_count}", (15, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        y_pos += 15
        
        cv2.putText(frame, f"Confidence Threshold: {self.confidence_threshold:.2f}", (15, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Controls
        cv2.putText(frame, "Controls: 'q' quit, 'r' reset, 's' save, 'p' pause", (15, 285), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    def start_session(self):
        """Start a new detection session"""
        self.session_start_time = time.time()
        self.logger.info("AUTO-SAVE Detection session started")
        print("🚀 AUTO-SAVE Detection session started!")
        print(f"📍 Saving photos to: detections/ folder")
        print(f"🎯 Confidence threshold: {self.confidence_threshold}")
    
    def end_session(self):
        """End the current detection session"""
        if self.session_start_time:
            self.total_session_time = time.time() - self.session_start_time
            self.logger.info(f"AUTO-SAVE session ended. Duration: {self.total_session_time:.2f}s, Photos saved: {self.saved_photos_count}")
            print(f"📸 Total photos auto-saved this session: {self.saved_photos_count}")
    
    def reset_detection(self):
        """Reset all detection data but keep saved photos count"""
        self.defect_history.clear()
        self.defect_counts = {'crack': 0, 'bend': 0, 'flaw': 0, 'weak_spot': 0}
        self.frame_count = 0
        self.processing_times.clear()
        self.confidence_scores.clear()
        # Keep saved_photos_count to track session total
        self.logger.info("Detection data reset! (Keeping saved photos count)")
        print("🔄 Detection data reset! Auto-save continues...")
    
    def adjust_sensitivity(self, increase=True):
        """Adjust detection sensitivity"""
        if increase:
            self.confidence_threshold = max(0.1, self.confidence_threshold - 0.1)
            print(f"🔍 Increased sensitivity! New threshold: {self.confidence_threshold:.2f}")
        else:
            self.confidence_threshold = min(0.9, self.confidence_threshold + 0.1)
            print(f"🎯 Decreased sensitivity! New threshold: {self.confidence_threshold:.2f}")
        
        self.logger.info(f"Confidence threshold adjusted to: {self.confidence_threshold:.2f}")
    
    def save_report(self):
        """Enhanced report with AUTO-SAVE statistics"""
        timestamp = datetime.now()
        report_dir = f"reports/{timestamp.strftime('%Y%m%d')}"
        os.makedirs(report_dir, exist_ok=True)
        
        # Text report
        report_file = f"{report_dir}/AUTO_SAVE_report_{timestamp.strftime('%H%M%S')}.txt"
        json_file = f"{report_dir}/AUTO_SAVE_data_{timestamp.strftime('%H%M%S')}.json"
        
        # Prepare data for JSON
        report_data = {
            'session_info': {
                'timestamp': timestamp.isoformat(),
                'frames_analyzed': self.frame_count,
                'session_duration': self.total_session_time,
                'photos_auto_saved': self.saved_photos_count,
                'confidence_threshold': self.confidence_threshold,
                'avg_processing_time': np.mean(self.processing_times) if self.processing_times else 0,
                'avg_confidence': np.mean(self.confidence_scores) if self.confidence_scores else 0
            },
            'defect_summary': self.defect_counts,
            'total_defects': sum(self.defect_counts.values()),
            'detections': self.defect_history[-100:]  # Last 100 detections
        }
        
        # Save JSON report
        with open(json_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        # Save text report
        with open(report_file, 'w') as f:
            f.write("🚨 RAILWAY TRACK AUTO-SAVE DEFECT DETECTION REPORT 🚨\n")
            f.write("="*70 + "\n")
            f.write(f"Generated: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Session Duration: {self.total_session_time:.2f} seconds\n")
            f.write(f"Frames Analyzed: {self.frame_count}\n")
            f.write(f"📸 Photos Auto-Saved: {self.saved_photos_count}\n")
            f.write(f"🎯 Confidence Threshold Used: {self.confidence_threshold:.2f}\n")
            
            if self.processing_times:
                f.write(f"Average Processing Time: {np.mean(self.processing_times)*1000:.2f}ms\n")
                f.write(f"Average FPS: {1.0/np.mean(self.processing_times):.1f}\n")
            
            if self.confidence_scores:
                f.write(f"Average Confidence: {np.mean(self.confidence_scores):.3f}\n")
            
            f.write("\nDEFECT SUMMARY:\n")
            f.write("-" * 40 + "\n")
            total_defects = sum(self.defect_counts.values())
            for defect_type, count in self.defect_counts.items():
                percentage = (count / total_defects) * 100 if total_defects > 0 else 0
                f.write(f"🔸 {defect_type.replace('_', ' ').title()}: {count} ({percentage:.1f}%)\n")
            
            f.write(f"\n📊 Total Defects Detected: {total_defects}\n")
            
            if self.saved_photos_count > 0:
                save_rate = (self.saved_photos_count / self.frame_count) * 100 if self.frame_count > 0 else 0
                f.write(f"📸 Photo Save Rate: {save_rate:.2f}% of frames\n")
            
            if self.defect_history:
                f.write(f"\n🔍 AUTO-SAVE DETECTION EVENTS:\n")
                f.write("-" * 50 + "\n")
                recent_saves = [d for d in self.defect_history[-20:] if d.get('avg_confidence', 0) >= self.confidence_threshold]
                for detection in recent_saves:
                    f.write(f"Frame {detection['frame']} - Confidence: {detection.get('avg_confidence', 0):.3f}\n")
                    for defect in detection['defects']:
                        if defect.get('confidence', 0) >= self.confidence_threshold:
                            f.write(f"  🚨 {defect['type'].upper()}: {defect.get('severity', 'unknown')} "
                                   f"(conf: {defect.get('confidence', 0):.3f})\n")
                    f.write("\n")
        
        self.logger.info(f"AUTO-SAVE Reports saved: {report_file}, {json_file}")
        print(f"📋 Report saved: {report_file}")
        print(f"💾 Data saved: {json_file}")

def initialize_camera():
    """Enhanced camera initialization with better error handling"""
    backends = [
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_V4L2, "Video4Linux"),
        (cv2.CAP_ANY, "Default")
    ]
    
    for backend_id, backend_name in backends:
        try:
            cap = cv2.VideoCapture(0, backend_id)
            if cap.isOpened():
                # Set camera properties for better quality
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                cap.set(cv2.CAP_PROP_FPS, 30)
                
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"✅ Camera initialized with {backend_name}")
                    print(f"Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
                    return cap
                cap.release()
        except Exception as e:
            print(f"Failed to initialize camera with {backend_name}: {e}")
    
    print("❌ Could not initialize any camera!")
    return None

def main():
    print("🚂🚨 ENHANCED AUTO-SAVE RAILWAY TRACK DEFECT DETECTION 🚨🚂")
    print("="*75)
    print("🔍 Detects: Cracks, Bends, Flaws, Weak Spots")
    print("📸 AUTO-SAVES photos with BRIGHT RED markings when defects found!")
    print("🎯 Adjustable sensitivity with + and - keys")
    print("📁 All photos saved to 'detections/' folder")
    print("="*75)
    
    # Initialize camera
    cap = initialize_camera()
    if cap is None:
        return
    
    # Initialize detector
    detector = RailwayTrackDefectDetector()
    detector.start_session()
    
    # Control variables
    paused = False
    show_help = False
    
    try:
        print("🔥 AUTO-SAVE MODE ACTIVE!")
        print("📹 Point camera at railway tracks!")
        print("🚨 Photos with RED markings AUTO-SAVED when defects detected!")
        print("📊 Use +/- keys to adjust sensitivity")
        print("📁 Check 'detections/' folder for saved photos")
        print("-" * 60)
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Failed to read frame from camera")
                    break
                
                # Analyze frame for defects (AUTO-SAVE happens inside this function)
                defects, processed = detector.analyze_frame(frame.copy())
                
                # Draw detections and statistics
                detector.draw_detections(frame, defects)
                detector.draw_statistics(frame)
                
                # Add real-time AUTO-SAVE status
                high_conf_defects = [d for d in defects if d.get('confidence', 0) >= detector.confidence_threshold]
                if high_conf_defects:
                    cv2.putText(frame, "🚨 DEFECTS FOUND - PHOTO AUTO-SAVED! 🚨", (10, 320), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                
                # Show results
                cv2.imshow('🚨 AUTO-SAVE Railway Defect Detection 🚨', frame)
                cv2.imshow('Processed View', processed)
                
                # Real-time console alerts for high confidence detections
                if high_conf_defects:
                    defect_types = list(set([d['type'].upper() for d in high_conf_defects]))
                    severities = list(set([d.get('severity', 'unknown').upper() for d in high_conf_defects]))
                    print(f"🔥 LIVE ALERT Frame {detector.frame_count}: {', '.join(defect_types)} - {', '.join(severities)}")
            
            # Handle controls
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("🛑 Stopping AUTO-SAVE detection...")
                break
            elif key == ord('r'):
                detector.reset_detection()
            elif key == ord('s'):
                print("💾 Saving comprehensive AUTO-SAVE report...")
                detector.save_report()
            elif key == ord('p'):
                paused = not paused
                status = "⏸️  PAUSED - Auto-save stopped" if paused else "▶️  RESUMED - AUTO-SAVE ACTIVE"
                print(f"{status}")
            elif key == ord('+') or key == ord('='):
                detector.adjust_sensitivity(increase=True)  # More sensitive
            elif key == ord('-') or key == ord('_'):
                detector.adjust_sensitivity(increase=False)  # Less sensitive
            elif key == ord('h'):
                show_help = not show_help
                if show_help:
                    print("\n" + "="*75)
                    print("🔧 AUTO-SAVE RAILWAY DEFECT DETECTION HELP")
                    print("="*75)
                    print("'q' - Quit the application")
                    print("'r' - Reset detection counters (keep photo count)")
                    print("'s' - Save comprehensive detection report")
                    print("'p' - Pause/Resume detection (AUTO-SAVE paused too)")
                    print("'+' - Increase sensitivity (lower threshold)")
                    print("'-' - Decrease sensitivity (higher threshold)")
                    print("'h' - Toggle this help message")
                    print("")
                    print("📸 AUTO-SAVE FEATURES:")
                    print("✅ Photos automatically saved when defects detected")
                    print("🔴 All defects marked with BRIGHT RED indicators")
                    print("📁 Saved in 'detections/' folder with detailed filename")
                    print("🎯 Adjustable confidence threshold for sensitivity")
                    print("⏱️  Minimum save interval to prevent spam")
                    print("📊 Enhanced marking with severity and confidence")
                    print("="*75 + "\n")
                else:
                    print("Help hidden - AUTO-SAVE continues")
    
    except KeyboardInterrupt:
        print("\n🛑 AUTO-SAVE detection stopped by user")
    except Exception as e:
        print(f"❌ Error during detection: {e}")
        detector.logger.error(f"Detection error: {e}")
    
    finally:
        print("🧹 Cleaning up...")
        detector.end_session()
        cap.release()
        cv2.destroyAllWindows()
        
        # Final comprehensive report
        if detector.frame_count > 0:
            print("\n📊 FINAL AUTO-SAVE SESSION SUMMARY:")
            print("="*60)
            print(f"📹 Frames processed: {detector.frame_count}")
            print(f"🚨 Total defects found: {sum(detector.defect_counts.values())}")
            print(f"📸 Photos auto-saved: {detector.saved_photos_count}")
            
            # Calculate save rate
            if detector.frame_count > 0:
                save_rate = (detector.saved_photos_count / detector.frame_count) * 100
                print(f"📈 Photo save rate: {save_rate:.2f}% of frames")
            
            # Count actual saved files
            detections_dir = "detections"
            if os.path.exists(detections_dir):
                saved_files = [f for f in os.listdir(detections_dir) if f.endswith('.jpg')]
                print(f"💾 Files in detections folder: {len(saved_files)}")
                
                # Show recent saves
                if saved_files:
                    recent_files = sorted(saved_files)[-5:]  # Last 5 files
                    print("📂 Recent auto-saved files:")
                    for f in recent_files:
                        print(f"   📸 {f}")
            
            if detector.processing_times:
                avg_fps = 1.0 / np.mean(detector.processing_times)
                print(f"⚡ Average FPS: {avg_fps:.1f}")
            if detector.confidence_scores:
                avg_conf = np.mean(detector.confidence_scores)
                print(f"🎯 Average confidence: {avg_conf:.3f}")
            
            # Auto-save final comprehensive report
            detector.save_report()
        
        print("="*60)
        print("✅ AUTO-SAVE Railway Defect Detection shutdown complete!")
        print("📁 Check 'detections/' folder for all auto-saved defect photos!")
        print("📋 Check 'reports/' folder for detailed analysis reports!")
        print("🚂 Thank you for using AUTO-SAVE Defect Detection! 🚂")

if __name__ == "__main__":
    # Create all necessary directories
    directories = ["reports", "logs", "screenshots", "detections"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Directory ready: {directory}/")
    
    print("\n🚀 Starting AUTO-SAVE Railway Defect Detection System...")
    main()