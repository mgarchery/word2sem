import sparql
import examples
import time
import numpy as np
from scipy import spatial
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from os.path import join, dirname
from dbpedia import get_related_entities, DBPEDIA_PREFIX



def main():
    # examples.word2vec_example()
    # examples.sparql_example()
    print 'Loading model...'
    model_path = '../data/dbpedia_noCats_model_sg_400.bin'  # '../data/dbpedia_Cats_model_sg_400.bin'
    n_entities = 100

    #
    # print 'Entities:', len(model.vocab.keys())
    #

    # for word in model.vocab.keys()[:3]:
    #     print 'Finding relations for word', word
    #     for (relation, related_entity) in get_related_entities(word):
    #         if related_entity in model.vocab.keys():
    #             print model.vocab[related_entity]

    # vector_entities = []
    model = Word2Vec.load_word2vec_format(model_path, binary=True)

    start_time = time.time()

    relation_vectors = dict()
    for base_entity in model.vocab.keys()[:n_entities]:
        print base_entity
        for (relation, related_entity) in get_related_entities(base_entity):
            try:
                if str(related_entity).encode('utf-8').startswith(DBPEDIA_PREFIX):
                    related_entity_without_prefix = related_entity[len(DBPEDIA_PREFIX):]
                    if related_entity_without_prefix in model:
                        # vector_entities.append(related_entity_without_prefix)

                        v1, v2 = model[base_entity], model[related_entity_without_prefix]
                        if relation in relation_vectors:
                            relation_vectors[relation].append(v2 - v1)
                        else:
                            relation_vectors[relation] = [v2 - v1]

            except UnicodeEncodeError:
                # TODO handle non unicode characters
                # print 'unicode encoding error with entity', related_entity
                continue

    # print vector_entities
    relations_statistics = []
    for relation in relation_vectors:
        vectors = relation_vectors[relation]
        cosine_distances = []
        for i, vi in enumerate(vectors):
            for j, vj in enumerate(vectors[i+1:]):
                cosine_distances.append(1.0 - spatial.distance.cosine(vi, vj))
        if len(cosine_distances) > 1:
            #print cosine_distances
            avg_cos, std_cos, count = np.mean(cosine_distances), np.std(cosine_distances), len(relation_vectors[relation])
            #print relation, 'count', count , ' / avg cos_sim', avg_cos, ' / std cos_sim', std_cos
            relations_statistics.append((relation, count, avg_cos, std_cos))

    relations_statistics.sort(key=lambda x: x[2], reverse=True)
    print relations_statistics
    print '{0:.2f}'.format(time.time() - start_time), 'seconds'

if __name__ == '__main__':
    main()
