@echo off

REM Execute export
python export_model.py -c ../configs/rec/en_PP-OCRv4_rec_train/en_PP-OCRv4_rec.yml -o Global.pretrained_model=output/rec_ppocr_v4/latest -o Global.save_inference_dir=./inference/rec_ppocr_v4

REM Pause the console to keep it open after execution
pause