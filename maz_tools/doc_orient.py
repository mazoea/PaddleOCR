import time
import argparse
from paddleocr import DocImgOrientationClassification


def orient_detect(img_path, model_dir):
    s = time.time()
    model = DocImgOrientationClassification(model_name="PP-LCNet_x1_0_doc_ori", model_dir=model_dir, device="cpu")
    output = model.predict(img_path, batch_size=1)
    for res in output:
        res.print(json_format=False)
    infer_time = time.time() - s
    print(f"Inference time: {infer_time:.3f}s")

def main():
    parser = argparse.ArgumentParser(description='Run Paddle doc-orientation locally.')
    parser.add_argument('--img', help='Input image',
                        type=str, required=True)
    parser.add_argument('--model_dir', help='Model dir where model with name PP-OCRv5_mobile_det will be stored',
                        type=str, required=True)

    flags = parser.parse_args()
    orient_detect(flags.img, flags.model_dir)

if __name__ == "__main__":
    main()
