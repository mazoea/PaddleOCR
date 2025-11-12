echo off
rem check if first argument is set
if "%1X"=="X" (
        echo Usage: cmp_model_to_qa.bat [MODEL_PATH]
echo Example: cmp_model_to_qa.bat d:\models\textspotting\my_model
pause
exit /b 1
)

set MODEL_PATH=%1

set VAL_DATASET=d:\projects\PaddleOCR\___dataset_for_val
set VAL_INPUT=d:\projects\PaddleOCR\___dataset_for_val\dataset_for_val
set OUTPUTPATH=d:\projects\issues\jira-515\newst_paddle\_val_output

rem run textspotting on the model
rm -rf %OUTPUTPATH%
rem d:\projects\PaddleOCR\___dataset_for_val  is in paddle like format
python batch_ts_v3_local.py --input %VAL_INPUT% --output %OUTPUTPATH% --model_dir %MODEL_PATH% --bbs=1 --threads=15
rem check if errorlevel is not 0
if not errorlevel 0 (
echo Textspotting failed
pause
exit /b 1
)

rem baseline_dir = results from QA, by using python create_qa_baseline.py
python cmp_ts_val_dataset.py --baseline %VAL_DATASET% --output %OUTPUTPATH%
if not errorlevel 0 (
echo Textspotting failed
pause
exit /b 1
)
echo "Model %MODEL_PATH% comparison to QA baseline completed."


rem echo "----------------------------------------------"
rem echo "Results in PP-OCRv5_mobile_det_infer with thresh=0.1, box_thresh=0.3"
rem echo "COMPARISON RESULTS"
rem echo "Total baseline files: 21"
rem echo "Total output files: 21"
rem echo "Matched files: 21"
rem echo "Unmatched files: 0"
rem echo "Total baseline bounding boxes: 4336"
rem echo "Covered bounding boxes: 3897"
rem echo "Uncovered bounding boxes: 439"
rem echo "Coverage rate: 89.88%"
rem echo "----------------------------------------------"

pause
