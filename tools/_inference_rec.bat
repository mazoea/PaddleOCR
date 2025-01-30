@echo off

REM Execute export
python export_model.py -c ../configs/rec/en_PP-OCRv4_rec_train/en_PP-OCRv4_rec.yml -o Global.pretrained_model=output/rec_ppocr_v4/latest Global.infer_img=./train_data/dataset.20250113_143943/99926b5b173440e28cba9b6dd4caaafc_478_(63739-434.bin.png Global.use_gpu=False

REM Pause the console to keep it open after execution
pause