import cv2
import numpy as np
from pyzbar.pyzbar import decode

def decoder(frame):
    """
    Detect and decode any QR or barcode in the given frame.
    Draws a polygon around the code and overlays the decoded text.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    barcodes = decode(gray)

    for obj in barcodes:
        # Draw polygon around detected code
        pts = np.array([obj.polygon], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], True, (0, 255, 0), 3)

        # Extract decoded data
        barcode_data = obj.data.decode("utf-8")
        barcode_type = obj.type
        x, y, w, h = obj.rect

        # Annotate the frame
        label = f"Data: {barcode_data} | Type: {barcode_type}"
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Log to console (for back-end integration or tracking database)
        print(f"Barcode: {barcode_data} | Type: {barcode_type}")

def main():
    # Open the default camera (change index if multiple cameras)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Camera not accessible")
        return

    print("▶ QR scanner started. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠ Frame grab failed")
            break

        decoder(frame)
        cv2.imshow('QR Code Scanner', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
