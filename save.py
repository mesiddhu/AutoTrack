import cv2
import os
from datetime import datetime

def main():
    # Get save path from user
    save_path = input("Enter the path where you want to save the photo (e.g., /path/to/folder/): ")
    
    # Validate and create path if needed
    if not os.path.exists(save_path):
        try:
            os.makedirs(save_path, exist_ok=True)
            print(f"Created directory: {save_path}")
        except Exception as e:
            print(f"Error creating directory: {e}")
            return
    
    # Get filename from user (optional)
    filename = input("Enter filename (without extension, press Enter for timestamp): ")
    if not filename:
        # Use timestamp as filename if no name provided
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"photo_{timestamp}"
    
    # Full path for output file
    photo_path = os.path.join(save_path, f"{filename}.jpg")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)  # 0 for default camera
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Camera opened successfully!")
    print("Position yourself for the photo...")
    print("Press SPACE to capture photo or 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame from camera")
            break
        
        # Display the live camera feed
        cv2.imshow('Camera - Press SPACE to capture, Q to quit', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        # Capture photo on spacebar press
        if key == ord(' '):
            # Save the current frame
            success = cv2.imwrite(photo_path, frame)
            
            if success:
                print(f"Photo captured and saved to: {photo_path}")
                # Show the captured photo for 3 seconds
                cv2.imshow('Captured Photo', frame)
                cv2.waitKey(3000)  # Display for 3 seconds
                break
            else:
                print("Error: Could not save photo")
        
        # Quit on 'q' key press
        elif key == ord('q'):
            print("Photo capture cancelled")
            break
    
    # Release camera and close windows
    cap.release()
    cv2.destroyAllWindows()
    
    if os.path.exists(photo_path):
        # Get image dimensions for confirmation
        img = cv2.imread(photo_path)
        height, width, channels = img.shape
        file_size = os.path.getsize(photo_path)
        print(f"Photo details:")
        print(f"- Resolution: {width}x{height}")
        print(f"- File size: {file_size/1024:.1f} KB")
        print(f"- Format: JPEG")

if __name__ == "__main__":
    main()