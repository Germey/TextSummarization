import gensim, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

sentences = [['first', 'sentence'], ['second', 'sentence']]
# train word2vec on the two sentences
model = gensim.models.Word2Vec(sentences, min_count=1, size=200)

model.save('test.model')

# print(model['first'])
# print(model['sentence'])
# print(model['second'])

print(model.similarity('first', 'second'))
print(model.similarity('first', 'sentence'))