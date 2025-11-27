REM set MODEL2=./output/v3_en_mobile_2024.04.02/best_accuracy
set MODEL1=./output/en_rec_ppocr_v5_store/best_accuracy
set MODEL2=./output/en_rec_ppocr_v5_store/best_accuracy
REM set MODEL2=./output/latin_rec_ppocr_v5/latest
python MAZbatchtest.py --model1=%MODEL1% --model2=%MODEL2%