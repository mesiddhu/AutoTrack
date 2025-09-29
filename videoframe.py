import cv2
import os
import time
import argparse
from pathlib import Path

class VideoFrameExtractor:
    def __init__(self):
        self.supported_formats = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v', '.3gp']
    
    def validate_video_path(self, video_path):
        """Validate video file path and format"""
        # Convert to Path object for better handling
        video_path = Path(video_path.strip().strip('"\''))
        
        # Check if file exists
        if not video_path.exists():
            return False, f"File does not exist: {video_path}"
        
        # Check if it's a file (not directory)
        if not video_path.is_file():
            return False, f"Path is not a file: {video_path}"
        
        # Check file extension
        if video_path.suffix.lower() not in self.supported_formats:
            return False, f"Unsupported format. Supported: {', '.join(self.supported_formats)}"
        
        # Check file size
        file_size = video_path.stat().st_size
        if file_size == 0:
            return False, "Video file is empty (0 bytes)"
        
        return True, str(video_path)
    
    def validate_output_directory(self, output_dir):
        """Validate and create output directory"""
        output_path = Path(output_dir.strip().strip('"\''))
        
        try:
            # Create directory if it doesn't exist
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Check if we can write to the directory
            test_file = output_path / "test_write.tmp"
            try:
                test_file.touch()
                test_file.unlink()
            except PermissionError:
                return False, f"No write permission for directory: {output_path}"
            
            return True, str(output_path)
            
        except Exception as e:
            return False, f"Error creating directory: {e}"
    
    def get_video_info(self, video_path):
        """Get comprehensive video information"""
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            return None, "Cannot open video file"
        
        info = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration': 0,
            'codec': None
        }
        
        # Calculate duration
        if info['fps'] > 0:
            info['duration'] = info['frame_count'] / info['fps']
        
        # Try to get codec information
        fourcc = cap.get(cv2.CAP_PROP_FOURCC)
        if fourcc:
            codec_bytes = int(fourcc).to_bytes(4, byteorder='little')
            try:
                info['codec'] = codec_bytes.decode('ascii').strip('\x00')
            except:
                info['codec'] = f"Unknown ({int(fourcc)})"
        
        cap.release()
        return info, None
    
    def extract_frames(self, input_video, output_directory, interval_seconds=1, max_frames=None, 
                      start_time=0, end_time=None, quality=95, prefix="frame"):
        """
        Extract frames from video with advanced options
        
        Args:
            input_video: Path to input video
            output_directory: Output directory path
            interval_seconds: Seconds between extracted frames
            max_frames: Maximum number of frames to extract
            start_time: Start time in seconds
            end_time: End time in seconds
            quality: JPEG quality (1-100)
            prefix: Filename prefix for frames
        """
        
        # Validate inputs
        valid, video_path = self.validate_video_path(input_video)
        if not valid:
            return False, video_path
        
        valid, output_path = self.validate_output_directory(output_directory)
        if not valid:
            return False, output_path
        
        # Get video info
        video_info, error = self.get_video_info(video_path)
        if error:
            return False, f"Error reading video: {error}"
        
        print(f"\n📹 Video Information:")
        print(f"   Resolution: {video_info['width']}x{video_info['height']}")
        print(f"   FPS: {video_info['fps']:.2f}")
        print(f"   Total frames: {video_info['frame_count']}")
        print(f"   Duration: {video_info['duration']:.2f} seconds")
        print(f"   Codec: {video_info['codec']}")
        print(f"   File size: {Path(video_path).stat().st_size / (1024*1024):.2f} MB")
        
        # Open video capture
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return False, "Failed to open video file"
        
        try:
            fps = video_info['fps']
            frame_interval = max(1, int(fps * interval_seconds))
            
            # Set start position
            if start_time > 0:
                start_frame = int(start_time * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            # Calculate end frame
            end_frame = None
            if end_time:
                end_frame = int(end_time * fps)
            
            success, image = cap.read()
            count = 0
            saved_count = 0
            failed_count = 0
            
            print(f"\n🎬 Starting frame extraction...")
            print(f"   Interval: {interval_seconds} seconds ({frame_interval} frames)")
            print(f"   Quality: {quality}%")
            print(f"   Output: {output_path}")
            print("-" * 50)
            
            start_extraction_time = time.time()
            
            while success and (max_frames is None or saved_count < max_frames):
                # Check if we've reached end time
                if end_frame and count >= end_frame:
                    break
                
                current_time = count / fps if fps > 0 else 0
                frame_filename = Path(output_path) / f"{prefix}_{saved_count:06d}_t{current_time:.2f}s.jpg"
                
                # Save frame with specified quality
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
                success_write = cv2.imwrite(str(frame_filename), image, encode_params)
                
                if success_write and frame_filename.exists():
                    file_size = frame_filename.stat().st_size
                    print(f"✅ Frame {saved_count:04d}: {frame_filename.name} ({file_size/1024:.1f} KB) - Time: {current_time:.2f}s")
                    saved_count += 1
                else:
                    print(f"❌ Failed to save frame {count}")
                    failed_count += 1
                
                # Skip frames according to interval
                for i in range(frame_interval):
                    success, image = cap.read()
                    count += 1
                    if not success:
                        break
            
            extraction_time = time.time() - start_extraction_time
            
            print("-" * 50)
            print(f"🏁 Extraction Complete!")
            print(f"   ✅ Frames saved: {saved_count}")
            print(f"   ❌ Failed saves: {failed_count}")
            print(f"   ⏱️  Time taken: {extraction_time:.2f} seconds")
            print(f"   📁 Output directory: {output_path}")
            
            return True, f"Successfully extracted {saved_count} frames"
            
        except Exception as e:
            return False, f"Error during extraction: {str(e)}"
        
        finally:
            cap.release()
    
    def interactive_extraction(self):
        """Interactive mode for frame extraction"""
        print("🎬 Video Frame Extractor")
        print("=" * 50)
        
        # Get video path
        while True:
            video_input = input("\n📁 Enter the path to the input video file: ").strip()
            if not video_input:
                print("❌ Please enter a valid path")
                continue
                
            valid, result = self.validate_video_path(video_input)
            if valid:
                video_path = result
                break
            else:
                print(f"❌ {result}")
                retry = input("Would you like to try again? (y/n): ")
                if retry.lower() != 'y':
                    return False, "User cancelled"
        
        # Get output directory
        while True:
            output_input = input("📁 Enter the path to the output directory: ").strip()
            if not output_input:
                print("❌ Please enter a valid path")
                continue
                
            valid, result = self.validate_output_directory(output_input)
            if valid:
                output_path = result
                break
            else:
                print(f"❌ {result}")
                retry = input("Would you like to try again? (y/n): ")
                if retry.lower() != 'y':
                    return False, "User cancelled"
        
        # Get extraction options
        try:
            interval = float(input("⏱️  Interval between frames (seconds) [default: 1.0]: ") or "1.0")
            max_frames_input = input("🔢 Maximum frames to extract [press Enter for all]: ").strip()
            max_frames = int(max_frames_input) if max_frames_input else None
            quality = int(input("🎨 JPEG quality (1-100) [default: 95]: ") or "95")
        except ValueError as e:
            print(f"❌ Invalid input: {e}")
            return False, "Invalid parameters"
        
        # Start extraction
        return self.extract_frames(video_path, output_path, interval, max_frames, quality=quality)


def main():
    parser = argparse.ArgumentParser(description="Advanced Video Frame Extractor")
    parser.add_argument("--input", "-i", help="Input video file path")
    parser.add_argument("--output", "-o", help="Output directory path")
    parser.add_argument("--interval", "-n", type=float, default=1.0, help="Interval between frames (seconds)")
    parser.add_argument("--max-frames", "-m", type=int, help="Maximum number of frames to extract")
    parser.add_argument("--start-time", "-s", type=float, default=0, help="Start time (seconds)")
    parser.add_argument("--end-time", "-e", type=float, help="End time (seconds)")
    parser.add_argument("--quality", "-q", type=int, default=95, help="JPEG quality (1-100)")
    parser.add_argument("--prefix", "-p", default="frame", help="Filename prefix")
    parser.add_argument("--info", action="store_true", help="Show video info only")
    
    args = parser.parse_args()
    
    extractor = VideoFrameExtractor()
    
    # If input provided via command line
    if args.input:
        if args.info:
            # Just show video info
            info, error = extractor.get_video_info(args.input)
            if error:
                print(f"❌ Error: {error}")
                return
            
            print("\n📹 Video Information:")
            for key, value in info.items():
                print(f"   {key}: {value}")
            return
        
        if not args.output:
            print("❌ Output directory required when using --input")
            return
        
        success, message = extractor.extract_frames(
            args.input, args.output, args.interval, args.max_frames,
            args.start_time, args.end_time, args.quality, args.prefix
        )
        
        if success:
            print(f"✅ {message}")
        else:
            print(f"❌ {message}")
    else:
        # Interactive mode
        success, message = extractor.interactive_extraction()
        if not success:
            print(f"❌ {message}")


if __name__ == "__main__":
    main()