# STEP 1 : import modules
import argparse
import cv2
import sys
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
# STEP 2 : create inference object(instance)
face = FaceAnalysis()
face.prepare(ctx_id=0, det_size=(640,640))

from fastapi import FastAPI, File, UploadFile
app = FastAPI()

@app.post("/uploadfile/")
async def create_upload_file(file1: UploadFile, file2: UploadFile):
    contents1 = await file1.read()
    contents2 = await file2.read()

# 여기부터 통신구간 연결이되는것

    # STEP 3 : load data
    binary1 = np.fromstring(contents1, dtype=np.uint8)
    binary2 = np.fromstring(contents2, dtype=np.uint8)
    
    img1 = cv2.imdecode(binary1, cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(binary2, cv2.IMREAD_COLOR)
    # STEP 4 : inference
    faces1 = face.get(img1)
    assert len(faces1)==1
    faces2 = face.get(img2)
    assert len(faces2)==1
    # STEP 5
    emb1 = faces1[0].normed_embedding
    emb2 = faces2[0].normed_embedding
    np_emb1 = np.array(emb1, dtype=np.float32)
    np_emb2 = np.array(emb2, dtype=np.float32)
    sims = np.dot(np_emb1, np_emb2.T)
    print(sims)
    return {"similarity": sims.item()}