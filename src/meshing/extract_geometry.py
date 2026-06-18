import numpy as np
import cv2
import os

def extract():
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    assets_dir = os.path.join(PROJECT_ROOT, 'assets')
    img_path = os.path.join(assets_dir, 'glomerulus_cross_section.png')
    
    if not os.path.exists(img_path):
        print("Image not found:", img_path)
        return
        
    # Read as grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # Resize to a manageable grid size for CFD (e.g. 100x100)
    img = cv2.resize(img, (100, 100))
    
    # Thresholding: Assuming dark regions are fluid, bright are walls
    # Otsu's thresholding
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Convert to binary mask: 1 for solid (walls), 0 for fluid
    # Let's say bright is solid
    mask = (thresh > 127).astype(np.int32)
    
    # Ensure inlet (left) and outlet (right) are open for flow
    mask[:, 0] = 0
    mask[:, -1] = 0
    
    # Save the mask
    os.makedirs(assets_dir, exist_ok=True)
    out_path = os.path.join(assets_dir, 'mask.npy')
    np.save(out_path, mask)
    print("Mask extracted and saved to", out_path, "Shape:", mask.shape)

if __name__ == "__main__":
    extract()
