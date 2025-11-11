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
