#step1 : import modules
import argparse
import cv2
import sys
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

# step 2 : create inference object(instance)
app = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(640,640))

#steo 3 : load data
# img = ins_get_image('t1')
img1 = cv2.imread('chu.jpg')
img2 = cv2.imread('1OG1QBQZD0_3.jpg')
# 파일오픈
# 디코드 이미지  두개 합쳐진에 아임 리드



# step 4 : inference
faces1 = app.get(img1)
faces2 = app.get(img2)
assert len(faces1)==1


# step 5 : post processing (application)
# rimg = app.draw_on(img, faces)
# cv2.imwrite("./t1_output.jpg", rimg)

# then print all-to-all face similarity
emb1 = faces1[0].normed_embedding
emb2 = faces2[0].normed_embedding

np_emb1 = np.array(emb1, dtype=np.float32)
np_emb2 = np.array(emb2, dtype=np.float32)

sims = np.dot(np_emb1, np_emb2.T)
print(sims)

