echo off
rem check if first argument is set
if "%1X"=="X" (
    echo Usage: cmp_model_to_qa.bat [MODEL_PATH]
    echo Example: cmp_model_to_qa.bat d:\models\textspotting\my_model
    pause
    exit /b 1
)

set MODEL_PATH=%1

SET QA_INPUT=d:\projects\issues\jira-515\newst_paddle\qa_input
set OUTPUTPATH=d:\projects\issues\jira-515\newst_paddle\qa_output
set QA_BASELINE=d:\projects\issues\jira-515\newst_paddle\qa_dp_advent_baseline

rem run textspotting on the model
rm -rf %OUTPUTPATH%
python batch_ts_v3_local.py --input %QA_INPUT% --output %OUTPUTPATH% --model_dir %MODEL_PATH% --bbs=1 --threads=15
rem check if errorlevel is not 0
if not errorlevel 0 (
    echo Textspotting failed
    pause
    exit /b 1
)

rem baseline_dir = results from QA, by using python create_qa_baseline.py
python cmp_ts_rs_with_qa_baseline.py --baseline-dir %QA_BASELINE% --output-dir %OUTPUTPATH%
if not errorlevel 0 (
    echo Textspotting failed
    pause
    exit /b 1
)
echo "Model %MODEL_PATH% comparison to QA baseline completed."


echo "----------------------------------------------"
echo "Results in PP-OCRv5_mobile_det_infer with thresh=0.1, box_thresh=0.3"
echo "Total files compared: 470"
echo "Total baseline bboxes: 110682"
echo "Total output bboxes: 104503"
echo "Total covered bboxes: 110562"
echo "Total uncovered bboxes: 120"
echo "Overall coverage: 99.89%"

pause