import json
import os
import argparse
import shutil
import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dataset_json', help='Input dataset JSON file with image paths. e.g. d:\projects\dataset-words-OCR\__dataset.words.zebra_words_phrases_lite.json',
                        type=str, required=True)
    parser.add_argument("--output_dir", help="Output directory where to save the training dataset.",
                        type=str, required=True)
    parser.add_argument("--perc_val", help="Percentage of data to use for validation set. e.g. 20",
                        type=float, default=20.0)

    flags = parser.parse_args()
    input_json = flags.input_dataset_json
    output_dir = flags.output_dir
    perc_val = flags.perc_val

    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    data = data.get('data', [])

    if 0 >= len(data):
        print(f"No data found in input dataset JSON: {input_json}")
        return

    #os.remove(output_dir) if os.path.exists(output_dir) else None
    os.makedirs(output_dir, exist_ok=True)

    num_val = int(len(data) * (perc_val / 100.0))
    every_val = len(data) // num_val

    # create train and val files
    train_file = os.path.join(output_dir, 'train.txt')
    val_file = os.path.join(output_dir, 'val.txt')
    with open(train_file, 'w', encoding='utf-8') as f_train, open(val_file, 'w', encoding='utf-8') as f_val:
        for i, item in tqdm.tqdm(enumerate(data)):
            if i % every_val == 0:
                f_val.write(f"{item['id']}\t{item['gt']}\n")
            else:
                f_train.write(f"{item['id']}\t{item['gt']}\n")
            # copy files to output dir
            src_path = os.path.join(os.path.dirname(input_json), item['id'])
            dst_path = os.path.join(output_dir, item['id'])
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy(src_path, dst_path)

if __name__ == '__main__':
    main()