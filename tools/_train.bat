@echo off

REM Execute training
python train.py -c ../configs/rec/en_PP-OCRv4_rec_train/en_PP-OCRv4_rec.yml -o Global.checkpoints=../configs/rec/en_PP-OCRv4_rec_train/best_accuracy

REM Pause the console to keep it open after execution
pause