#step 1
from sentence_transformers import SentenceTransformer

#step 2
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# step 3
sentences = [
    "안녕 하세요",
    "집에 가고싶다",
    "이게 맞냐",
]

#sentences 1 = "The weather is lovely today."
#sentences 2 = "It's so sunny outside!" 따로 나눠서 배열로안하면 이렇게 나눠줄수 있음


#step 4
embeddings = model.encode(sentences)

# embedding1 = model.encode(sentence1)


print(embeddings.shape)

#[1,384] * 3
# [3, 384]

# 3. Calculate the embedding similarities
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.6660, 0.1046],
#         [0.6660, 1.0000, 0.1411],
#         [0.1046, 0.1411, 1.0000]])