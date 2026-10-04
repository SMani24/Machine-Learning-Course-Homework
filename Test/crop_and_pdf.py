import cv2
import os
import json
from PIL import Image

# --- Configuration ---
INPUT_FOLDER = "Book Pics"
OUTPUT_PDF = "cropped_output.pdf"
CROP_MEMORY_FILE = "crop_memory.json"

# Max dimensions for the preview window to ensure it fits on your screen
MAX_DISPLAY_WIDTH = 1500
MAX_DISPLAY_HEIGHT = 850
# ---------------------

# Global variables for the interactive window
ref_point = []
cropping = False
image_for_display = None # Use a separate variable for the displayed image

def select_crop_area(event, x, y, flags, param):
    """Mouse callback function to draw the rectangle."""
    global ref_point, cropping, image_for_display

    if event == cv2.EVENT_LBUTTONDOWN:
        ref_point = [(x, y)]
        cropping = True
    elif event == cv2.EVENT_LBUTTONUP:
        ref_point.append((x, y))
        cropping = False
        cv2.rectangle(image_for_display, ref_point[0], ref_point[1], (255, 0, 0), 2)
        cv2.imshow("Select Crop Area", image_for_display)

def save_crop_coords(coords):
    """Saves crop coordinates to a JSON file."""
    with open(CROP_MEMORY_FILE, 'w') as f:
        json.dump({'crop_box': coords}, f)
    print("✅ Crop coordinates saved for future use.")

def load_crop_coords():
    """Loads crop coordinates from the JSON file if it exists."""
    if os.path.exists(CROP_MEMORY_FILE):
        try:
            with open(CROP_MEMORY_FILE, 'r') as f:
                data = json.load(f)
                return tuple(data['crop_box'])
        except (json.JSONDecodeError, KeyError):
            print(f"⚠️ Warning: '{CROP_MEMORY_FILE}' is corrupted. A new one will be created.")
            return None
    return None

def get_new_crop_coords(first_image_path):
    """Opens an interactive, scaled-down window to define a new crop area."""
    global image_for_display, ref_point
    
    # Load the original, full-resolution image
    original_image = cv2.imread(first_image_path)
    if original_image is None:
        print(f"❌ Error: Could not read the image at {first_image_path}")
        return None

    orig_height, orig_width = original_image.shape[:2]
    
    # Calculate the scaling factor to fit the image on the screen
    scale = min(MAX_DISPLAY_WIDTH / orig_width, MAX_DISPLAY_HEIGHT / orig_height)
    if scale >= 1.0:
        # Image is smaller than max display size, no need to resize
        scale = 1.0
        image_for_display = original_image.copy()
    else:
        # Image is larger, resize it for display
        display_width = int(orig_width * scale)
        display_height = int(orig_height * scale)
        image_for_display = cv2.resize(original_image, (display_width, display_height), interpolation=cv2.INTER_AREA)

    clone = image_for_display.copy()
    cv2.namedWindow("Select Crop Area")
    cv2.setMouseCallback("Select Crop Area", select_crop_area)

    print("\n✅ A (potentially scaled) image window has opened.")
    print("INSTRUCTIONS:")
    print("1. Click and drag to draw a box.")
    print("2. Press 'r' to reset.")
    print("3. Press 'c' to confirm.")

    while True:
        cv2.imshow("Select Crop Area", image_for_display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("r"):
            image_for_display = clone.copy()
        elif key == ord("c"):
            if len(ref_point) == 2:
                break
            else:
                print("   ⚠️ Please select an area first.")
    cv2.destroyAllWindows()

    # Convert the coordinates from the resized preview back to the original image scale
    x1_display, y1_display = ref_point[0]
    x2_display, y2_display = ref_point[1]
    
    x1_orig = int(min(x1_display, x2_display) / scale)
    y1_orig = int(min(y1_display, y2_display) / scale)
    x2_orig = int(max(x1_display, x2_display) / scale)
    y2_orig = int(max(y1_display, y2_display) / scale)
    
    crop_box = (x1_orig, y1_orig, x2_orig, y2_orig)
    save_crop_coords(crop_box)
    return crop_box

def main():
    """Main function to run the cropping and PDF conversion process."""
    if not os.path.exists(INPUT_FOLDER):
        print(f"❌ Error: The folder '{INPUT_FOLDER}' was not found.")
        return

    try:
        image_files = sorted([f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if not image_files:
            raise FileNotFoundError
    except FileNotFoundError:
        print(f"❌ Error: No image files found in '{INPUT_FOLDER}'.")
        return

    first_image_path = os.path.join(INPUT_FOLDER, image_files[-300])
    crop_box = None
    saved_coords = load_crop_coords()

    if saved_coords:
        print(f"💾 Found saved crop coordinates: {saved_coords}")
        while True:
            choice = input("Do you want to (R)euse these coordinates or define a (N)ew one? [R/N]: ").lower()
            if choice in ['r', 'n']:
                break
            print("   Invalid input. Please enter 'R' or 'N'.")
        
        if choice == 'r':
            crop_box = saved_coords
            print("👍 Reusing saved coordinates.")
        else:
            crop_box = get_new_crop_coords(first_image_path)
    else:
        print("No saved crop found. Please define a new crop area.")
        crop_box = get_new_crop_coords(first_image_path)

    if not crop_box:
        print("❌ Error: No crop area was defined. Exiting.")
        return

    print(f"\nProcessing images with crop area: {crop_box}...")
    cropped_images_for_pdf = []
    for filename in image_files:
        print(f"   - Cropping {filename}")
        img_path = os.path.join(INPUT_FOLDER, filename)
        pil_img = Image.open(img_path)
        cropped_pil_img = pil_img.crop(crop_box)
        
        if cropped_pil_img.mode in ('RGBA', 'P'):
            cropped_pil_img = cropped_pil_img.convert('RGB')
        
        cropped_images_for_pdf.append(cropped_pil_img)

    if cropped_images_for_pdf:
        print(f"\nSaving all cropped images to '{OUTPUT_PDF}'...")
        cropped_images_for_pdf[0].save(
            OUTPUT_PDF,
            save_all=True,
            append_images=cropped_images_for_pdf[1:]
        )
        print("🎉 Done! Your PDF has been created successfully.")
    else:
        print("No images were processed to create a PDF.")

if __name__ == '__main__':
    main()
