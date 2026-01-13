REM set MODEL2=./output/v3_en_mobile_2024.04.02/best_accuracy
set MODEL1=d:\projects\PaddleOCR\output\v5_en_mobile_2025_12_31\best_model\model.pdparams  
set MODEL2=d:\projects\PaddleOCR\output\v5_en_mobile_2025_12_31\step_012300.metric_0.9725.pdparams
REM set MODEL2=./output/latin_rec_ppocr_v5/latest
python MAZbatchtest.py --model1=%MODEL1% --model2=%MODEL2%