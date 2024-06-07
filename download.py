from sentence_transformers import SentenceTransformer

# 下载模型并保存到本地
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
model.save('./model/sentence-transformers/all-MiniLM-L6-v2')
