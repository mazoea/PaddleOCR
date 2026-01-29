import time
import argparse
from paddleocr import TextRecognition
import os
import cv2
import numpy as np
from bbox_from_ocr_response import get_char_bboxes, get_word_bboxes, visualize_bboxes, process_ocr_response, visualize_words

def ocr(img_path, model_dir):
    if "PP-OCRv5_mobile_rec" in model_dir:
        model_name = "PP-OCRv5_mobile_rec"
    if "PP-OCRv5_server_rec" in model_dir:
        model_name = "PP-OCRv5_server_rec"
    if  "en_PP-OCRv5_mobile_rec" in model_dir:
        model_name = "en_PP-OCRv5_mobile_rec"
    if "latin_PP-OCRv5_mobile_rec" in model_dir:
        model_name = "latin_PP-OCRv5_mobile_rec"
    s = time.time()
    model = TextRecognition(model_name=model_name,
                            device='cpu',
                            cpu_threads=1,
                            enable_mkldnn=False,
                            mkldnn_cache_capacity=0,
                            model_dir=model_dir)
    mode_read = time.time()
    # We found the gold - return_word_box which return position of letters
    output = model.predict(input=img_path,  batch_size=1, return_word_box=True)
    #output = model.predict(input=img_path,  batch_size=1)
    infer_time = time.time() - mode_read
    total_time = time.time() - s
    model_read = mode_read - s
    print(f"Model load time: {model_read:.3f}s, Inference time: {infer_time:.3f}s, Total time: {total_time:.3f}s")

    # get image dimmensions
    img = cv2.imread(img_path)
    h_img, w_img = img.shape[0:2]

    ret = []
    for res in output:
        print(f"{res.get('rec_text', 'no_text')}, {res.get('rec_score', 0.0)}")
        text_info = res.get('rec_text', [])
        #char_bbs = get_char_bboxes(text_info, w_img, h_img)
        #word_bbs = get_word_bboxes(text_info, w_img, h_img)
        # visualize character bboxes
        #visualize_bboxes(img_path, char_bbs, f"{img_path}_chars_bbox.png")
        #visualize_bboxes(img_path, word_bbs, f"{img_path}_words_bbox.png")
        words = process_ocr_response(text_info, w_img, h_img)
        #visualize_words(img_path, words, f"{img_path}_words.png")

        #ret.append((res.get('rec_text', 'no_text'), res.get('rec_score', 0.0)))
        #res.print()
        #res.save_to_img(save_path="./___output/")
        #res.save_to_json(save_path="./___output/res.json")
    return ret

def main():
    parser = argparse.ArgumentParser(description='Run Paddle textspotting locally. Possibility to set model '
                                                 'for test and output path, where to save results.')
    parser.add_argument('--img', help='Input image',
                        type=str, required=True)
    parser.add_argument('--model_dir', help='Model dir where model with name PP-OCRv5_mobile_det will be stored',
                        type=str, required=True)

    flags = parser.parse_args()
    ocr(flags.img, flags.model_dir)

if __name__ == "__main__":
    main()
