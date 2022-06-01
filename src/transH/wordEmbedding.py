from gensim.models import Word2Vec
import numpy as np
import pickle

def WordEmbedding(entity_file: str, relation_file: str, train_file: str, test_file: str, vector_size: int=100):
    entity_ids = []
    entity_sentences = []
    relation_ids = []
    relation_sentences = []
    train_entity_ids = set() # there are certain duplication like 13133
    train_relation_ids = set()  # pre process
    with open(entity_file, "r", encoding="utf-8") as fr:
        for line in fr:
            id, sentence = line.strip().split("\t")
            entity_ids.append(int(id))
            entity_sentences.append(sentence.split())
    with open(relation_file, "r", encoding="utf-8") as fr:
        for line in fr:
            id, sentence = line.strip().split("\t")
            relation_ids.append(int(id))
            relation_sentences.append(sentence.split())
    with open(train_file, "r", encoding="utf-8") as fr:
        for line in fr:
            h, r, t = line.strip().split("\t")
            train_entity_ids.add(int(h))
            train_entity_ids.add(int(t))
            train_relation_ids.add(int(r))
    with open(test_file, "r", encoding="utf-8") as fr:
        for line in fr:
            h, r, t = line.strip().split("\t")
            train_entity_ids.add(int(h))
            train_relation_ids.add(int(r))
    sentence = entity_sentences
    sentence.extend(relation_sentences)
    model = Word2Vec(sentences = sentence, vector_size = vector_size, min_count = 1) # Word2Vec embedding

    word2index = {}
    vectors = []
    for idx, (id, sentence) in enumerate(zip(entity_ids, entity_sentences)):
        word2index[id] = idx
        mean_vector = np.mean([model.wv[word] for word in sentence], axis = 0)
        vectors.append(mean_vector)
    for id in train_entity_ids:
        if id not in entity_ids:
            word2index[id] = len(word2index)
            vectors.append(np.random.normal(0, 0.5, vector_size)) # give it a random vector rather than eliminate it
    vectors = np.stack(vectors)
    entity_embedding = {"word2index": word2index, "embedded": vectors}
    pickle.dump(entity_embedding, open("./output/entity_embedded.bin", "wb"))

    word2index = {}
    vectors = []
    for idx, (id, sentence) in enumerate(zip(relation_ids, relation_sentences)):
        word2index[id] = idx
        mean_vector = np.mean([model.wv[word] for word in sentence], axis = 0)
        vectors.append(mean_vector)
    vectors = np.stack(vectors)
    entity_embedding = {"word2index": word2index, "embedded": vectors}
    pickle.dump(entity_embedding, open("./output/relation_embedded.bin", "wb"))
