See https://github.com/mazoea/PaddleOCR/pull/2

0. ```
    python -m pip install paddlepaddle-gpu==2.6.1.post120 -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html
   ```
1. Read https://paddlepaddle.github.io/PaddleOCR/main/en/ppocr/model_train/training.html
2. Download https://paddlepaddle.github.io/PaddleOCR/latest/en/model/index.html, see __maz
3. Copy and update .yml training configuration
4. ```
   pip install orjson
   ```
5. Update `en_PP-OCRv3_rec.yml` and run
   ```
   python tools/train.py -c ./__maz/en_PP-OCRv3_rec_train/en_PP-OCRv3_rec.yml -o Global.checkpoints=./__maz/en_PP-OCRv3_rec_train/best_accuracy
   ```
6. Test model
   ```
   python tools/infer_rec.py -c ./__maz/en_PP-OCRv3_rec_train/en_PP-OCRv3_rec.yml -o Global.checkpoints=./output/v3_en_mobile/best_accuracy Global.infer_img=c:/data/woec_dataset/multi/IBedits/000000_pad/Acid)_007b404c037049c7862071900aa85a09_20168.in.pad-3.png_020168.pad-3.png Global.use_gpu=False
   ```
7. Eval orig vs finetuned
   see `MAZtest.bat`
6. 
7. Export model
   ```
   python tools/export_model.py -c ./__maz/en_PP-OCRv3_rec_train/en_PP-OCRv3_rec.yml -o Global.pretrained_model=./output/v3_en_mobile/best_accuracy Global.save_inference_dir=./output/v3_en_mobile/inference-2/
   ```


Other resources:
- https://gist.github.com/leonbora167/049ac6622b7a2fb5c23ec48070af486f
- https://paddlepaddle.github.io/PaddleOCR/main/en/ppocr/blog/inference_args.html


# Label Studio for annotation/correction

1. 
```
λ docker run -it -p 8080:8080 -v %cd%/labelling:/label-studio/data heartexlabs/label-studio:latest
```

2.
Add new project->Settings->Labeling Interface->Code
```
<View>
  <Style>
        .text-container {
          background-color: white;
          border-radius: 10px;
          box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
          padding: 20px;
          font-family: 'Courier New', monospace;
          line-height: 1.6;
          font-size: 16px;
        }
    </Style>  
  <Image name="image" value="$image"/>

  <View className="text-container">
     <TextArea name="transcription" toName="image"
               editable="true"
               required="true"
               value="$gt"
               maxSubmissions="1"
               rows="2"
               placeholder="Recognized Text"
               />

<Header value="GT remove"/>
    <Choices name="gt_check" toName="image"
             choice="single" showInLine="true">
    <Choice value="remove"/>
    <Choice value="ok"/>
</Choices>

     <Labels name="paddle" toName="text-paddle"><Label value="Paddle" /></Labels>
     <Text name="text-paddle" value="$paddletext" granularity="word" highlightColor="#ff0000" />
   
     <Labels name="T3" toName="text-t3"><Label value="T3" /></Labels>
     <Text name="text-t3" value="$t3text" granularity="word" highlightColor="#ff0000" />
   
     <Labels name="T4" toName="text-t4"><Label value="T4" /></Labels>
     <Text name="text-t4" value="$t4text" granularity="word" highlightColor="#ff0000" />
   </View>
    
</View>
```

3. Create labels
See README.md in `te-dataset-finetune`



## dev-10.pc

```
docker run  -e LABEL_STUDIO_HOST=http://dev-10.pc/labels/ -d --restart always -p 8090:8080 -v /opt/labelling:/label-studio/data heartexlabs/label-studio:latest
```


# Debugging paddle detection

See https://paddlepaddle.github.io/PaddleOCR/main/en/ppocr/quick_start.html#211-chinese-and-english-model
Download test images and: 

```
import glob
import os
import sys
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont


font = ImageFont.truetype("./fonts/simfang.ttf", size=20)  # Adjust size as needed

def viz(img_path, results):
    # Load image
    image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    # Process and draw results
    for res in results:
        for line in res:
            box = [tuple(point) for point in line[0]]
            # Finding the bounding box
            box = [(min(point[0] for point in box), min(point[1] for point in box)),
                   (max(point[0] for point in box), max(point[1] for point in box))]
            txt = line[1][0]
            draw.rectangle(box, outline="red", width=2)  # Draw rectangle
            draw.text((box[0][0], box[0][1] - 5), txt, fill="blue", font=font)  # Draw text above the box
    # Save result
    image.save(os.path.basename(img_path))


# Initialize OCR engine
ocr = PaddleOCR(use_angle_cls=False, lang="en")
img_paths = glob.glob(sys.argv[1])
for img_path in img_paths:
    results = ocr.ocr(img_path, rec=True, cls=False)
    viz(img_path, results)
```

Adjust `DBPostProcess::box_score_slow` in `ppocr/postprocess/db_postprocess.py`.
