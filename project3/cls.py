
#step 1 : import module
from transformers import pipeline

#step 2  : create inference object(instance)
classifier = pipeline("sentiment-analysis", model="snunlp/KR-FinBert-SC") # 그 사람의 / 모델

#step 3 : prepare data
text = "북한과 한국의 전쟁"


# step 4 : infreence
result = classifier(text)

# step 5 : post processing
print(result)