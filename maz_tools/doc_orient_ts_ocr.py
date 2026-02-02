import time
import argparse
import os
import numpy as np
import cv2

# PaddlePaddle CPU optimization flags for Lambda
# These prevent segmentation faults on Lambda's CPU architecture
os.environ['FLAGS_use_mkldnn'] = 'false'  # Disable MKL-DNN to avoid CPU compatibility issues
os.environ['FLAGS_cpu_deterministic'] = 'true'  # Force deterministic behavior
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Allow duplicate library loading
os.environ['OMP_NUM_THREADS'] = '1'  # Limit OpenMP threads
os.environ['MKL_NUM_THREADS'] = '1'  # Limit MKL threads
os.environ['OPENBLAS_NUM_THREADS'] = '1'  # Limit OpenBLAS threads

from paddleocr import TextDetection
from paddleocr import TextRecognition

def detect_rot(img_path, ocr_model_dir, ts_model_dir):
    # Initialize text spotting model
    s = time.time()
    text_detector = TextDetection(device='cpu', model_dir=ts_model_dir, model_name="PP-OCRv5_mobile_det", limit_side_len=64, limit_type='min',
                             thresh=0.1, box_thresh=0.3, unclip_ratio=1.5, cpu_threads=1, enable_mkldnn=False, mkldnn_cache_capacity=0)
    # Initialize OCR model
    text_ocr = TextRecognition(model_name="en_PP-OCRv5_mobile_rec",
                            device='cpu',
                            cpu_threads=1,
                            enable_mkldnn=False,
                            mkldnn_cache_capacity=0,
                            model_dir=ocr_model_dir)

    # read image a scale it
    img = cv2.imread(img_path)

    if img is None:
        #raise FileNotFoundError(f"Image not found at {img_path}")
        print(f"Image not found at {img_path}")
        return -1

    max_side_len = 1280
    h, w, _ = img.shape
    if max(h, w) > max_side_len:
        ratio = max_side_len / max(h, w)
        img = cv2.resize(img, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_AREA)

    # take rectangle from the middle of the image, start with 20% of size, if there will no black make it 40%
    h, w, _ = img.shape
    for scale in [0.2, 0.4, 0.6, 1.]:
        all_confs = []
        try:
            # perform rotation image
            for angle in [0, 90, 180, 270]:
                if angle == 0:
                    rotated_img = img
                else:
                    # rotate image
                    center = (img.shape[1] // 2, img.shape[0] // 2)
                    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                    rotated_img = cv2.warpAffine(img, rot_matrix, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

                start_x = int(w * (1 - scale) / 2)
                start_y = int(h * (1 - scale) / 2)
                end_x = int(w * (1 + scale) / 2)
                end_y = int(h * (1 + scale) / 2)
                crop_img = rotated_img[start_y:end_y, start_x:end_x]

                # check if in cropped image is more then 10% of black pixels
                gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                _, thresh_img = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
                black_pixels = np.sum(thresh_img == 0)
                total_pixels = thresh_img.shape[0] * thresh_img.shape[1]
                black_ratio = black_pixels / total_pixels
                if scale != 1. and black_ratio < 0.05:
                    raise ValueError("no_black")

                # detect text in the cropped image
                confs = []
                ts_res = text_detector.predict(crop_img)
                print(f"Angle {angle} degrees, detected {len(ts_res[0]['dt_polys'])} text boxes")
                for i, bbox in enumerate(ts_res[0]["dt_polys"], 1):
                    box = np.reshape(np.array(bbox), [-1, 1, 2]).astype(np.int64)
                    # crop the detected text region
                    x, y, w_box, h_box = cv2.boundingRect(box)
                    word_img = crop_img[y:y+h_box, x:x+w_box]
                    # if word is verical skip it
                    if h_box > 1.5 * w_box:
                        continue
                    # recognize text in the cropped region
                    ocr_res = text_ocr.predict(word_img, batch_size=1, return_word_box=True)
                    for res in ocr_res:
                        print(f"{res.get('rec_text', 'no_text')[0]}, {res.get('rec_score', 0.0)}")
                        conf = res.get('rec_score', 0.0)
                        text = res.get('rec_text', ' ')[0]
                        if len(text) < 3 and conf < 0.5:
                            # skip very short low confidence words
                            continue
                        if len(text) < 3:
                            # decrease confidence for very short words
                            conf *= 0.75
                        # count the spaces in the text
                        space_count = text.count(' ')
                        if space_count > 0:
                            confs.extend([conf]*space_count)
                        confs.append(conf)
                    if 10 < len(confs):
                        # limit to first 10 words
                        break
                all_confs.append(confs)
                print("-------")

            # heigh average confidence will give the rotation, height will be according to the number of detected words
            avg_confs = []
            words_counts = []
            for confs in all_confs:
                if len(confs) == 0:
                    avg_confs.append(0.0)
                else:
                    avg_confs.append(sum(confs) / len(confs))
                # count the highest word counts
                words_counts.append(len(confs))
            print(avg_confs)
            max_word_counts = max(max(words_counts),1)
            print(max_word_counts)
            avg_confs = [(avg / max_word_counts) * len(all_confs[i]) for i, avg in enumerate(avg_confs)]
            best_angle_idx = np.argmax(avg_confs)
            best_angle = [0, 90, 180, 270][best_angle_idx]
            print(avg_confs)
            print(f"Detected rotation: {best_angle} degrees with average confidence {avg_confs[best_angle_idx]:.4f}")
            return best_angle, time.time() - s
        except ValueError as ve:
            if str(ve) == "no_black":
                print(f"Scale {scale:.2f} - no black pixels detected, trying larger area...")
                continue
            else:
                print(f"Error during processing: {str(ve)}")
                return -1

    return -1

def main():
    parser = argparse.ArgumentParser(description='Run Paddle doc-orientation locally.')
    parser.add_argument('--img', help='Input image',
                        type=str, required=True)
    parser.add_argument('--ts_model_dir', help='TS Model dir where model with name PP-OCRv5_mobile_det will be stored',
                        type=str, required=True)
    parser.add_argument('--ocr_model_dir', help='OCR Model dir where model with name en_PP-OCRv5_mobile_rec will be stored',
                        type=str, required=True)

    flags = parser.parse_args()
    detect_rot(flags.img, flags.ocr_model_dir, flags.ts_model_dir)

if __name__ == "__main__":
    main()
