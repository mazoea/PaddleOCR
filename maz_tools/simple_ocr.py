import time
import argparse
from paddleocr import TextRecognition

def ocr(img_path, model_dir):
    model_name = "PP-OCRv5_server_rec" if 'server' in model_dir else "PP-OCRv5_mobile_rec"
    s = time.time()
    model = TextRecognition(model_name=model_name, device='cpu', cpu_threads=1, enable_mkldnn=False, mkldnn_cache_capacity=0, model_dir=model_dir)
    mode_read = time.time()
    # We found the gold - return_word_box which return position of letters
    output = model.predict(input=img_path,  batch_size=1, return_word_box=True)
    #output = model.predict(input=img_path,  batch_size=1)
    infer_time = time.time() - mode_read
    total_time = time.time() - s
    model_read = mode_read - s
    print(f"Model load time: {model_read:.3f}s, Inference time: {infer_time:.3f}s, Total time: {total_time:.3f}s")
    ret = []
    for res in output:
        print(f"{res.get('rec_text', 'no_text')}, {res.get('rec_score', 0.0)}")
        ret.append((res.get('rec_text', 'no_text'), res.get('rec_score', 0.0)))
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
