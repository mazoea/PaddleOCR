# Maz Tools - PaddleOCR Utilities

This folder contains utility scripts for testing trained models.

## Table of Contents

- [How to train](#how-to-train)
- [How to Run Text Spotting on Lambda](#how-to-run-text-spotting-on-lambda)
- [Compare New Model to QA Baseline](#compare-new-model-to-qa-baseline)
- [Compare New Model to Validation Dataset](#compare-new-model-to-validation-dataset)
- [Additional Tools](#additional-tools)

---
## How to train
- in PaddleOCR folder
```bash
python tools/train.py -c configs/det/PP-OCRv5/PP-OCRv5_mobile_det.yml -o Global.pretrained_model=./PP-OCRv5_mobile_det_pretrained/PP-OCRv5_mobile_det_pretrained.pdparams Train.dataset.data_dir=./___dataset_for_training Train.dataset.label_file_list="[./___dataset_for_training/Label.txt]" Eval.dataset.data_dir=./___dataset_for_val Eval.dataset.label_file_list="[./___dataset_for_val/Label.txt]"
```
### visualtion of training process
- copy `visualize_training.py` into folder, where training happened, where `train.log` is presented (`d:\projects\PaddleOCR\output\PP-OCRv5_mobile_det\`) and run
```bash
python visualize_training.py
```
several images are generated and show (also stored).

### export trained model
```bash
python tools/export_model.py -c configs/det/PP-OCRv5/PP-OCRv5_mobile_det.yml -o Global.pretrained_model=output/PP-OCRv5_mobile_det_11_3_2025/best_accuracy.pdparams Global.save_inference_dir="./PP-OCRv5_mobile_det_infer/"
```

## How to Run Text Spotting on Lambda

### Prerequisites

- AWS CLI configured with appropriate credentials
- Docker installed (for building Lambda deployment packages)
- PaddleOCR model inference files (`inference.pdparams`, `inference.pdiparams`, `inference.json`)

### Step 1: Build and Deploy Lambda Package
- copy model into maz_tools and name folder `PP-OCRv5_mobile_det_infer`
- docker expecting model in this folder with this name
37.000013.pdf.raw.json.bboxes.png
```bash
# Build Docker image with deployment package
cd maz_tools
docker build -t paddleocr-lambda -f Dockerfile .

# Deploy using the build script
./build_and_deploy.bat
```cmp_model_to_qa.bat

### Step 2: Invoke Lambda Function

Use the `invoke_lambda.py` script to test your deployed Lambda function:

**Example:**
```bash
python invoke_lambda.py input.png te-advent-tm-v2 eu-central-1
```
### Step 3: Test Lambda Handler Locally

Before deploying, you can test the Lambda handler locally:

```bash
python test_lambda.py <image_path>
```

## Compare New Model to QA Baseline

This comparison evaluates how well a newly trained model performs against QA baseline results by measuring bounding box coverage.

### Compare ts_words from raw.json with QA Baseline

#### Script: `cmp_ts_words_with_qa_baseline.py`

This script compares ts_words bounding boxes from raw.json files with QA baseline data. It's useful when you have raw.json files with `ts_words` key containing text spotting results.

#### Usage

```bash
python cmp_ts_words_with_qa_baseline.py \
    --raw-json-dir <path-to-raw-json-files> \
    --baseline-dir <path-to-qa-baseline-dir> \
    --overlap-threshold 0.5 \
    --show-details
```

**Parameters:**
- `--raw-json-dir`: Directory containing raw.json files with ts_words (default: `d:\projects\issues\jira-515\newst_paddle`)
- `--baseline-dir`: Directory containing QA baseline JSON files (default: `d:\projects\issues\jira-515\newst_paddle\qa_dp_advent_baseline`)
- `--overlap-threshold`: Minimum overlap ratio to consider a bbox as covered (default: 0.5)
- `--show-details`: Show detailed per-file statistics including low coverage files

#### Example

```bash
python cmp_ts_words_with_qa_baseline.py \
    --raw-json-dir D:\projects\issues\jira-515\newst_paddle \
    --baseline-dir D:\projects\issues\jira-515\newst_paddle\qa_dp_advent_baseline \
    --overlap-threshold 0.5 \
    --show-details
```

#### Input Format

**Raw JSON file (*.raw.json):**
```json
{
    "ts_words": [
        {
            "bbox": {"x": 100, "y": 200, "w": 150, "h": 30},
            "text": "Sample text",
            "score": 0.95
        },
        {
            "bbox": {"x": 300, "y": 250, "w": 200, "h": 40},
            "text": "Another text",
            "score": 0.92
        }
    ],
    "img_w": 1240,
    "img_h": 1754
}
```

**QA Baseline JSON:**
```json
{
    "bboxes": [
        {"x": 100, "y": 200, "w": 150, "h": 30},
        {"x": 300, "y": 250, "w": 200, "h": 40}
    ],
    "deskew": -2.5,
    "rotation": 0,
    "page_bbox": {"x": 0, "y": 0, "w": 2480, "h": 3508}
}
```

#### Output

The script provides:
- Total files processed
- Total baseline bounding boxes
- Covered bounding boxes (detected in ts_words)
- Uncovered bounding boxes (missed in ts_words)
- Overall coverage rate percentage
- Per-file statistics with `--show-details` flag showing:
  - Baseline bbox count
  - TS words bbox count
  - Covered count
  - Uncovered count
  - Coverage percentage
- List of files with coverage < 80%

#### How It Works

1. Finds matching files between raw.json and baseline directories by filename
2. Extracts ts_words bboxes from raw.json files
3. Transforms ts_words bboxes to baseline coordinate system (scaling + deskew)
4. Compares each baseline bbox against all transformed ts_words bboxes
5. Uses `bbox.overlap_min()` with threshold to determine coverage
6. Reports comprehensive statistics

---

### Creating baseline from QA 
- it needs to be done only once
```bash
python create_qa_baseline.py --input <qa_dir_raw_json_files> --output <output_dir>
```
It will produce two files for every input raw json `*.bboxes.json` and `*.raw.json.bboxes.png`. Only json is needed, but png is for debug purpouse.

### cmp_model_to_qa.bat
- batch script is subject to change, specially folders and paths
- it runs two scripts to run textspotting with selected model and script which compare the result
- it shows also baseline result for original model with settings of inferening `thresh=0.1, box_thresh=0.3`
- as an argument is a path into model to check
- variables:
    - QA_INPUT - folder with png files of QA
    - OUTPUTPATH - folder where will be stored processed QA images
    - QA_BASELINE - folder where are results from running script `create_qa_baseline.py` from previous step
- scripts runs approximally 15 minutes
- it checks how many bounding boxes are not covered by textspotted bounding boxes by new model

#### Script: `cmp_ts_rs_with_qa_baseline.py`

This script compares text spotting output with QA baseline data, accounting for coordinate transformations (scaling, deskewing, rotation).

#### Usage

```bash
python cmp_ts_rs_with_qa_baseline.py \
    --baseline <path-to-qa-baseline-dir-created-in-previous-step> \
    --output <path-to-model-output-dir> \
    --overlap-threshold 0.5 \
    --show-details
```

**Parameters:**
- `--baseline`: Directory containing QA baseline JSON files with bboxes, deskew, rotation, and page_bbox information
- `--output`: Directory containing model output JSON files with detected bboxes
- `--overlap-threshold`: Minimum overlap ratio to consider a bbox as covered (default: 0.5)
- `--show-details`: Show detailed per-file uncovered bbox counts

#### Example

```bash
python cmp_ts_rs_with_qa_baseline.py \
    --baseline D:\qa_baseline\results \
    --output D:\model_output\detections \
    --overlap-threshold 0.5 \
    --show-details
```

### Input Format

**QA Baseline JSON:**
```json
{
    "bboxes": [
        {"x": 100, "y": 200, "w": 150, "h": 30},
        {"x": 300, "y": 250, "w": 200, "h": 40}
    ],
    "deskew": -2.5,
    "rotation": 0,
    "page_bbox": {"x": 0, "y": 0, "w": 2480, "h": 3508}
}
```

**Model Output JSON:**
```json
{
    "bboxes": [
        {"x": 50, "y": 100, "w": 150, "h": 30},
        {"x": 150, "y": 125, "w": 200, "h": 40}
    ],
    "img_w": 1240,
    "img_h": 1754
}
```

### Output Metrics

The script reports:
- **Total baseline files** and **output files**
- **Matched files** between baseline and output
- **Total baseline bounding boxes**
- **Covered bounding boxes** (detected by model)
- **Uncovered bounding boxes** (missed by model)
- **Coverage rate** (percentage of bboxes detected)
- **Per-file breakdown** (with `--show-details`)

---

## Compare New Model to Validation Dataset

This comparison evaluates model performance against the validation dataset used during training.

## cmp_model_to_val.bat
- batch script is subject to change, specially folders and paths
- it runs two scripts to run textspotting with selected model and script which compare the result
- it shows also baseline result for original model with settings of inferening `thresh=0.1, box_thresh=0.3`
- as an argument is a path into model to check
- variables:
    - VAL_DATASET - validation dataset used during training in Paddle format
    - VAL_INPUT - validation dataset only images input
    - OUTPUTPATH - folder where will be stored processed validation images
- scripts runs approximally 2 minutes
- it checks how many bounding boxes are not covered by textspotted bounding boxes by new model


#### Script: `cmp_ts_val_dataset.py`

This script compares model output with the validation dataset (Label.txt format) without coordinate transformations, since the input images are the same.

#### Usage

```bash
python cmp_ts_val_dataset.py \
    --baseline <path-to-validation-dataset-dir> \
    --output <path-to-model-output-dir> \
    --overlap-threshold 0.5 \
    --show-details
```

**Parameters:**
- `--baseline`: Path to validation dataset directory (must contain `Label.txt`)
- `--output`: Directory containing model output JSON files with detected bboxes
- `--overlap-threshold`: Minimum overlap ratio to consider a bbox as covered (default: 0.5)
- `--show-details`: Show detailed per-file uncovered bbox counts

#### Example

```bash
python cmp_ts_val_dataset.py \
    --baseline D:\projects\PaddleOCR\___dataset_for_val \
    --output D:\projects\issues\jira-515\newst_paddle\_val_output \
    --overlap-threshold 0.5 \
    --show-details
```

#### Input Format

**Validation Dataset (Label.txt):**
```
dataset_for_val/image1.png	[{"transcription": "Hello", "points": [[10, 20], [100, 20], [100, 50], [10, 50]], "difficult": false}]
dataset_for_val/image2.png	[{"transcription": "World", "points": [[50, 100], [150, 100], [150, 130], [50, 130]], "difficult": false}]
```

**Model Output JSON:**
```json
{
    "bboxes": [
        {"x": 10, "y": 20, "w": 90, "h": 30}
    ]
}
```

#### Output Metrics

The script reports:
- **Total baseline files** and **output files**
- **Matched files** between baseline and output
- **Unmatched files** (present in baseline but not in output)
- **Total baseline bounding boxes**
- **Covered bounding boxes** (detected by model)
- **Uncovered bounding boxes** (missed by model)
- **Coverage rate** (percentage of bboxes detected)
- **Per-file breakdown** (with `--show-details`)

#### Key Differences from QA Comparison

- **No coordinate transformations:** Since the input is identical, no scaling or deskewing is needed
- **Label.txt format:** Parses PaddleOCR's training dataset format
- **Direct comparison:** Bboxes are compared in their original coordinate system

---

## Additional Tools

### Evaluate OCR on Dataset

**Script:** `evaluate_ocr_dataset.py`

Evaluate OCR recognition model performance on a labeled dataset with multi-threading support. This script performs OCR on each image and compares results with ground truth to calculate accuracy metrics.

#### Usage

```bash
python evaluate_ocr_dataset.py \
    --dataset <path-to-dataset.json> \
    --model-dir <path-to-recognition-model> \
    --threads <num-threads> \
    --show-errors
```

**Parameters:**
- `--dataset`: Path to dataset JSON file (default: `d:\projects\dataset-words-OCR\__dataset.words.IBedits.json`)
- `--model-dir`: Path to OCR recognition model directory (e.g., `PP-OCRv5_mobile_rec`)
- `--threads`: Number of parallel threads for processing (default: 4)
- `--output`: Path to save detailed results JSON (optional)
- `--show-errors`: Show detailed mismatches between predictions and ground truth
- `--limit`: Limit number of images to process (useful for testing)

#### Example

```bash
# Evaluate with 8 threads
python evaluate_ocr_dataset.py \
    --dataset D:\projects\dataset-words-OCR\__dataset.words.IBedits.json \
    --model-dir ./PP-OCRv5_mobile_rec \
    --threads 8 \
    --show-errors

# Save results to file
python evaluate_ocr_dataset.py \
    --dataset dataset.json \
    --model-dir ./PP-OCRv5_mobile_rec \
    --output evaluation_results.json

# Test with limited images
python evaluate_ocr_dataset.py \
    --dataset dataset.json \
    --model-dir ./PP-OCRv5_mobile_rec \
    --limit 100 \
    --threads 4
```

#### Dataset Format

The script supports JSON files with the following format:

**Option 1 - Array of objects:**
```json
[
    {
        "image": "images/word001.png",
        "gt": "Hello"
    },
    {
        "image": "images/word002.png",
        "gt": "World"
    }
]
```

**Option 2 - Object with data key:**
```json
{
    "data": [
        {"image": "images/word001.png", "gt": "Hello"},
        {"image": "images/word002.png", "gt": "World"}
    ]
}
```

Alternative key names supported:
- Image path: `"image"`, `"img"`, or `"path"`
- Ground truth: `"gt"` or `"text"`

#### Output Metrics

The script reports:
- **Total images processed**
- **Exact matches**: Number and percentage of perfect matches
- **Character accuracy**: Percentage of correctly recognized characters
- **Processing time**: Total and average per image
- **Detailed errors**: With `--show-errors`, shows first 20 mismatches with ground truth vs. predicted text

#### Multi-Threading

The script uses `ThreadPoolExecutor` for parallel processing:
- Each thread initializes its own OCR model instance
- Thread-safe statistics collection using locks
- Progress bar shows real-time processing status
- Recommended threads: 4-8 depending on CPU cores and memory

---

### Evaluate OCR on Dataset (GPU with Batching)

**Script:** `evaluate_ocr_dataset_gpu.py`

GPU-accelerated version of the OCR evaluation script. Processes images in batches using CUDA for significantly faster evaluation compared to CPU multi-threading.

#### Usage

```bash
python evaluate_ocr_dataset_gpu.py \
    --dataset <path-to-dataset.json> \
    --model-dir <path-to-recognition-model> \
    --batch-size <batch-size>
```

**Parameters:**
- `--dataset`: Path to dataset JSON file (default: `d:\projects\dataset-words-OCR\__dataset.words.IBedits.json`)
- `--model-dir`: Path to OCR recognition model directory (e.g., `PP-OCRv5_mobile_rec`)
- `--batch-size`: Number of images to process in parallel (default: 8)
- `--output`: Path to save detailed results JSON (optional)
- `--show-errors`: Show detailed mismatches between predictions and ground truth
- `--limit`: Limit number of images to process (useful for testing)

#### Example

```bash
# Evaluate with default batch size (8)
python evaluate_ocr_dataset_gpu.py \
    --dataset D:\projects\dataset-words-OCR\__dataset.words.IBedits.json \
    --model-dir ./PP-OCRv5_mobile_rec

# Use larger batch size for faster processing
python evaluate_ocr_dataset_gpu.py \
    --dataset dataset.json \
    --model-dir ./PP-OCRv5_mobile_rec \
    --batch-size 16 \
    --show-errors

# Save results and show errors
python evaluate_ocr_dataset_gpu.py \
    --dataset dataset.json \
    --model-dir ./PP-OCRv5_mobile_rec \
    --batch-size 8 \
    --output gpu_results.json \
    --show-errors
```

#### Prerequisites

- **CUDA-enabled GPU** with appropriate drivers installed
- **paddlepaddle-gpu** installed instead of regular paddlepaddle:
  ```bash
  # Install paddlepaddle-gpu (CUDA 11.8 example)
  python -m pip install paddlepaddle-gpu==3.0.0b1 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
  ```

#### Key Features

- **GPU Acceleration**: Uses CUDA for significantly faster inference
- **Batch Processing**: Processes multiple images simultaneously
- **Optimized Performance**: Typically 3-5x faster than CPU multi-threading
- **Same Metrics**: Reports exact matches, character accuracy, and processing times
- **Progress Tracking**: Real-time progress bar with batch processing status

#### Output Metrics

Same as CPU version, plus:
- **Average time per batch**: Shows batch processing efficiency
- **Device**: Confirms GPU (CUDA) usage

#### Performance Comparison

Typical performance (depends on GPU):
- **CPU (4 threads)**: ~0.3-0.5s per image
- **GPU (batch size 8)**: ~0.05-0.1s per image
- **Speedup**: 3-5x faster with GPU batching

#### Batch Size Recommendations

- **Batch size 8**: Good balance for most GPUs (default)
- **Batch size 16-32**: For high-end GPUs with more VRAM
- **Batch size 4**: For GPUs with limited memory
- Monitor GPU memory usage and adjust accordingly

---

### Batch Processing

**Script:** `batch_process.py`

Process multiple images with PaddleOCR model:

```bash
python batch_process.py \
    --input-dir <path-to-images> \
    --output-dir <path-to-output> \
    --model-dir <path-to-model>
```

### Resize Images

**Script:** `resize_images.py`

Resize images in dataset for training:

```bash
python resize_images.py \
    --input-dir <path-to-original-images> \
    --output-dir <path-to-resized-images> \
    --max-size 1024
```

### Visualize Training Results

**Script:** `visualize_training.py`

Visualize training metrics and generate plots:

```bash
python visualize_training.py \
    --log-file <path-to-training-log> \
    --output-dir <path-to-output-plots>
```

### Test Model
- doing text spotting
- important 

**Script:** `ts_v3.py`

Test PaddleOCR v3 model inference:

```bash
python ts_v3.py \
    --input <path-to-test-image> \
    --output <where-should-be-result-stored> \
    --model_dir <path-to-model> \
    --bbs=1 <should-be-also-file-with-bounding-boxes-created>
```

---

## Dependencies

Install required Python packages:

```bash
pip install -r requirements.txt
```

**Key dependencies:**
- `paddlepaddle` or `paddlepaddle-gpu`
- `paddleocr`
- `coords` (custom bbox transformation library)
- `tqdm` (progress bars)
- `Pillow` (image processing)
- `numpy`

---
