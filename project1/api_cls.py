# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.components import processors
from mediapipe.tasks.python import vision

# STEP 2: Create an ImageClassifier object. 추론기를 객체로 만든다
base_options = python.BaseOptions(model_asset_path='models\efficientnet_lite0.tflite')
options = vision.ImageClassifierOptions(
    base_options=base_options, max_results=3)
classifier = vision.ImageClassifier.create_from_options(options)

from fastapi import FastAPI, File, UploadFile

app = FastAPI()

import cv2
import numpy as np
@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    contents = await file.read()
    #텍스상태 이미지 디코딩이안댐 

    # STEP 3: Load the input image.
    binary = np.fromstring(contents, dtype = np.uint8)  
   # 위에걸로 바뀜 
    cv_mat = cv2.imdecode(binary, cv2.IMREAD_COLOR)
    #오픈 CV 로바꿈 이미지 디코딩 해주고 컬러를 넣어준다
    rgb_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv_mat)
    # image = mp.Image.create_from_file('burger.jpg')
    # 추론기로 넣을수있는 파일로 바꿀수 있고 대입해서 추론할 수 있다 .

    # STEP 4: Classify the input image.
    classification_result = classifier.classify(rgb_frame)
    # print(classification_result)


    # STEP 5: Process the classification result. In this case, visualize it.
    top_category = classification_result.classifications[0].categories[0]
    # print(f"{top_category.category_name} ({top_category.score:.2f})")
    return {"category": top_category.category_name,
            "score": f"{top_category.score:.2f}"}