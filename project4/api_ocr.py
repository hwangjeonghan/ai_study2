from fastapi import FastAPI, File, UploadFile
import easyocr
import cv2
import numpy as np
reader = easyocr.Reader(['en','ko']) # this needs to run only once to load the model into memory
app = FastAPI()

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    contents = await file.read()

    binary = np.fromstring(contents, np.uint8)
    cv_mat= cv2.imdecode(binary,cv2.IMREAD_COLOR)
    

    result = reader.readtext(cv_mat)

    print(result)

    return {"filename": result}