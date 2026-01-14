
import os
import pandas as pd
import numpy as np
import cv2
import easyocr
from glob import glob
import tempfile
import shutil
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)


def extractor(g2raw_path, helicoid_path, output_folder):
    """
    Extracts angio frames from a .g2raw file using Helicoid-Magick and saves them as images in the specified output folder.

    Args:
        g2raw_path (str): Path to the input .g2raw file.
        helicoid_path (str): Path to the Helicoid-Magick executable.
        output_folder (str): Directory to save the extracted angio frames.
    """
    output_file = os.path.join(output_folder, "angio_frames.png")
    logging.info(f"[extractor] Extracting angio frames from: {g2raw_path}")
    logging.info(f"[extractor] Output file: {output_file}")
    cmd = f'{helicoid_path} "{g2raw_path}" --angio "{output_file}"'
    logging.info(f"[extractor] Running command: {cmd}")
    os.system(cmd)
    logging.info(f"[extractor] Extraction complete.")


def clahe(img):
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance image contrast.

    Args:
        img (np.ndarray): Input BGR image.
    Returns:
        np.ndarray: CLAHE-enhanced BGR image.
    """
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    c = cv2.createCLAHE(2.0, (8, 8)).apply(g)
    return cv2.cvtColor(c, cv2.COLOR_GRAY2BGR)

def invert(img):
    """
    Inverts the pixel values of an image.

    Args:
        img (np.ndarray): Input image.
    Returns:
        np.ndarray: Inverted image.
    """
    return 255 - img

def upscale(img, scale=2):
    """
    Upscales an image by a given scale factor using cubic interpolation.

    Args:
        img (np.ndarray): Input image.
        scale (int): Scale factor.
    Returns:
        np.ndarray: Upscaled image.
    """
    h, w = img.shape[:2]
    return cv2.resize(img, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)



def detect(img, reader, conf=0.15):
    """
    Uses OCR to detect text regions in an image.

    Args:
        img (np.ndarray): Input image.
        reader (easyocr.Reader): EasyOCR reader instance.
        conf (float): Minimum confidence threshold for detection.
    Returns:
        list: List of bounding boxes for detected text regions.
    """
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hits = []
    for b, t, c in reader.readtext(rgb):
        if c >= conf:
            hits.append(b)
    return hits



def mask_from_boxes(shape, boxes, pad=4):
    """
    Creates a binary mask from a list of bounding boxes.

    Args:
        shape (tuple): Shape of the image (height, width, channels).
        boxes (list): List of bounding boxes.
        pad (int): Padding for dilation.
    Returns:
        np.ndarray: Binary mask with masked regions set to 255.
    """
    mask = np.zeros(shape[:2], dtype=np.uint8)
    for b in boxes:
        pts = np.array(b, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)

    k = 2*pad + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(mask, kernel, 1)


def anonymize(image, reader, blur_passes = 2):
    """
    Detects and anonymizes text regions in an image by blurring them.

    Args:
        image (np.ndarray): Input image.
        reader (easyocr.Reader): EasyOCR reader instance.
        blur_passes (int): Number of blur passes to apply.
    Returns:
        tuple: (anonymized image, mask)
    """
    boxes = detect(image, reader)
    mask = mask_from_boxes(image.shape, boxes)
    for i in range(blur_passes):
        blurred = cv2.GaussianBlur(image, (0,0), 7)
    out = image.copy()
    out[mask == 255] = blurred[mask == 255]
    return out, mask



def batch_anonymize(input_folder, output_folder, reader):
    """
    Batch processes all images in a folder, anonymizing text regions and saving the results.

    Args:
        input_folder (str): Folder containing input images.
        output_folder (str): Folder to save anonymized images.
        reader (easyocr.Reader): EasyOCR reader instance.
    """
    logging.info(f"[batch_anonymize] Starting batch anonymization from {input_folder} to {output_folder}")
    os.makedirs(output_folder, exist_ok=True)
    image_files = sorted(glob(os.path.join(input_folder, "*.*")))
    logging.info(f"[batch_anonymize] Found {len(image_files)} image files.")

    for path in image_files:
        logging.info(f"[batch_anonymize] Processing: {path}")
        img = cv2.imread(path)
        if img is None:
            logging.warning(f"[batch_anonymize] Skipping {path}, could not read image.")
            continue

        anon_img, mask = anonymize(img, reader)
        base_name = os.path.basename(path)
        save_path = os.path.join(output_folder, base_name)
        cv2.imwrite(save_path, anon_img)
        logging.info(f"[batch_anonymize] Anonymized and saved: {save_path}")

    logging.info(f"[batch_anonymize] All images processed and saved to: {output_folder}")



def import_anonymized_angio(original_g2raw, helicoid_path, anonymized_folder, output_dir):
    """
    Imports anonymized angio frames back into a new .g2raw file using Helicoid-Magick.

    Args:
        original_g2raw (str): Path to the original .g2raw file.
        helicoid_path (str): Path to the Helicoid-Magick executable.
        anonymized_folder (str): Folder containing anonymized angio images (should be sequentially named PNGs).
        output_dir (str): Directory to save the new anonymized .g2raw file.

    Returns:
        str or None: Path to the new anonymized .g2raw file if successful, else None.
    """
    import subprocess
    logging.info(f"[import_anonymized_angio] Importing anonymized angio frames from {anonymized_folder} into new g2raw file.")
    abs_output_dir = os.path.abspath(output_dir)
    abs_anonymized_folder = os.path.abspath(anonymized_folder)
    abs_g2raw = os.path.abspath(original_g2raw)
    abs_helicoid = os.path.abspath(helicoid_path)
    os.makedirs(abs_output_dir, exist_ok=True)
    base_name = os.path.basename(abs_g2raw)
    new_g2raw = os.path.join(abs_output_dir, f"anonymizeDB-{base_name}")

    logging.info(f"[import_anonymized_angio] abs_output_dir: {abs_output_dir}")
    logging.info(f"[import_anonymized_angio] abs_anonymized_folder: {abs_anonymized_folder}")
    logging.info(f"[import_anonymized_angio] abs_g2raw: {abs_g2raw}")
    logging.info(f"[import_anonymized_angio] abs_helicoid: {abs_helicoid}")
    logging.info(f"[import_anonymized_angio] new_g2raw: {new_g2raw}")
    logging.info(f"[import_anonymized_angio] Current working directory: {os.getcwd()}")

    cmd = [
        abs_helicoid,
        abs_g2raw,
        "--import-angio", abs_anonymized_folder,
        "-o", new_g2raw
    ]
    logging.info(f"[import_anonymized_angio] Running command (as list): {cmd}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, shell=False)
        logging.info(f"[import_anonymized_angio] Return code: {result.returncode}")
        logging.info(f"[import_anonymized_angio] STDOUT:\n{result.stdout}")
        logging.info(f"[import_anonymized_angio] STDERR:\n{result.stderr}")
    except Exception as e:
        logging.error(f"[import_anonymized_angio] Exception occurred: {e}")
        return None

    if os.path.exists(new_g2raw):
        logging.info(f"[import_anonymized_angio] Anonymized g2raw created: {new_g2raw}")
        return new_g2raw
    else:
        logging.error(f"[import_anonymized_angio] Failed to create anonymized g2raw. Check if the import command ran successfully and if the output file exists.")
        return None
    

def secure_processing_pipeline(g2raw_path, helicoid_path, output_folder, reader, new_raw_folder):
    """
    Main pipeline for secure anonymization of angio frames in a .g2raw file.

    Steps:
        1. Create a temporary folder for PHI (Protected Health Information).
        2. Extract raw angio frames from the .g2raw file into the temp folder.
        3. Anonymize all images and save to the verification/output folder.
        4. Wait for user to review and approve anonymized images.
        5. Delete PHI temp data automatically.
        6. Import anonymized angio frames back into a new .g2raw file.

    Args:
        g2raw_path (str): Path to the input .g2raw file.
        helicoid_path (str): Path to the Helicoid-Magick executable.
        output_folder (str): Folder to save anonymized images and final .g2raw.
        reader (easyocr.Reader): EasyOCR reader instance.
        new_raw_folder (str): Folder to save the new anonymized .g2raw file.

    Returns:
        str or None: Path to the new anonymized .g2raw file if successful, else None.
    """

    logging.info(f"[secure_processing_pipeline] Starting pipeline for: {g2raw_path}")
    with tempfile.TemporaryDirectory() as temp_phi_dir:
        logging.info(f"[secure_processing_pipeline] TEMP PHI DIR: {temp_phi_dir}")

        # Step 1: extract frames with PHI
        logging.info("[secure_processing_pipeline] Step 1: Extracting frames with PHI...")
        extractor(g2raw_path, helicoid_path, temp_phi_dir)

        # Step 2: anonymize into safe folder
        logging.info("[secure_processing_pipeline] Step 2: Anonymizing images...")
        batch_anonymize(temp_phi_dir, output_folder, reader)

        logging.info("[secure_processing_pipeline] Review anonymized images here:")
        logging.info(os.path.abspath(output_folder))

        input("[secure_processing_pipeline] Press ENTER to delete original PHI images...")

        # leaving `with` automatically deletes temp dir
        logging.info("[secure_processing_pipeline] PHI temporary directory securely deleted.")

    logging.info("[secure_processing_pipeline] Step 3: Importing anonymized angio frames back into g2raw...")
    anonymized_g2raw = import_anonymized_angio(
        original_g2raw=g2raw_path,
        helicoid_path=helicoid_path,
        anonymized_folder=output_folder,
        output_dir=new_raw_folder)
    if anonymized_g2raw:
        logging.info(f"[secure_processing_pipeline] Pipeline complete. Anonymized g2raw at: {anonymized_g2raw}")
    else:
        logging.error(f"[secure_processing_pipeline] Pipeline failed to create anonymized g2raw.")
    return anonymized_g2raw


def main():
    """
    Entry point for the anonymization pipeline.
    Supports three input types:
      1. Single file
      2. Folder of files
      3. Excel/CSV list of files
    """
    import sys
    input_path = r"S:\01 - Clinical G2 Data\2022 - CFD\04-058 - CFD\04-058 - OCT\04-058-Anonymous-Anonymous-20240329-093034.g2raw"  # Example: can be file, folder, or Excel/CSV
    helicoid_path = r"C:\Users\dbhagria\Genshi-v25.5.5\bin\helicoid-magick.exe"
    output_folder = 'review_anonymized'
    raw_folder = 'raw_folder'
    reader = easyocr.Reader(['en'], gpu=False)
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(raw_folder, exist_ok=True)

    # Detect input type
    if os.path.isfile(input_path) and input_path.lower().endswith('.g2raw'):
        # Single file mode
        logging.info(f"Processing single file: {input_path}")
        result = secure_processing_pipeline(input_path, helicoid_path, output_folder, reader, raw_folder)
        if result:
            logging.info(f"Anonymized .g2raw file saved at: {result}")
        else:
            logging.error("Anonymized .g2raw file was not created. Please check for errors.")

    elif os.path.isdir(input_path):
        # Folder mode
        g2raw_files = sorted(glob(os.path.join(input_path, '*.g2raw')))
        logging.info(f"Processing folder: {input_path} with {len(g2raw_files)} .g2raw files")
        for g2raw_file in g2raw_files:
            logging.info(f"Processing file: {g2raw_file}")
            result = secure_processing_pipeline(g2raw_file, helicoid_path, output_folder, reader, raw_folder)
            if result:
                logging.info(f"Anonymized .g2raw file saved at: {result}")
            else:
                logging.error(f"Failed to anonymize: {g2raw_file}")

    elif input_path.lower().endswith(('.xlsx', '.xls', '.csv')) and os.path.isfile(input_path):
        # Excel/CSV mode
        logging.info(f"Processing spreadsheet: {input_path}")
        if input_path.lower().endswith('.csv'):
            df = pd.read_csv(input_path)
        else:
            df = pd.read_excel(input_path)
        # Assume column 'g2raw_path' contains file paths
        if 'g2raw_path' not in df.columns:
            logging.error("Spreadsheet must contain a 'g2raw_path' column with file paths.")
            return
        for g2raw_file in df['work_path'].dropna():
            if not os.path.isfile(g2raw_file):
                logging.warning(f"File not found: {g2raw_file}")
                continue
            logging.info(f"Processing file: {g2raw_file}")
            result = secure_processing_pipeline(g2raw_file, helicoid_path, output_folder, reader, raw_folder)
            if result:
                logging.info(f"Anonymized .g2raw file saved at: {result}")
            else:
                logging.error(f"Failed to anonymize: {g2raw_file}")
    else:
        logging.error(f"Input path not recognized or supported: {input_path}")



if __name__ == "__main__":
    main()


    