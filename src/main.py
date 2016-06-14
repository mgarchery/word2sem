#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv

import time
import numpy as np
from scipy import spatial
from gensim.models import Word2Vec
from dbpedia import get_related_entities, DBPEDIA_PREFIX


def write_csv(relations_statistics, csv_path):
    with open(csv_path, 'wb') as f:
        writer = csv.writer(f, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        header = ['Relation', 'Count', 'Average cosine sim.', 'Std. Dev. cosine sim.']
        writer.writerow(header)
        for (relation, count, avg_cos, std_cos) in relations_statistics:
            writer.writerow((relation, count, avg_cos, std_cos))

def main():
    # examples.word2vec_example()
    # examples.sparql_example()

    model_path = '../data/WikiEntityModel_400_neg10_iter5.seq'
    #'../data/dbpedia_noCats_model_sg_400.bin'  # '../data/dbpedia_Cats_model_sg_400.bin'
    n_entities = 100
    out_path = '../data/word2sem_.csv'

    print 'Loading model...'
    model = Word2Vec.load_word2vec_format(model_path, binary=True)

    start_time = time.time()

    relation_vectors = dict()
    if n_entities > 0:
        relations = model.vocab.keys()[:n_entities]
    else:
        relations = model.vocab.keys()

    for i, base_entity in enumerate(relations):
        print i, base_entity
        for (relation, related_entity) in get_related_entities(base_entity):
            related_entity = unicode(related_entity).encode('utf8')
            if related_entity.startswith(DBPEDIA_PREFIX):
                related_entity_without_prefix = related_entity[len(DBPEDIA_PREFIX):]
                if related_entity_without_prefix in model:
                    v1, v2 = model[base_entity], model[related_entity_without_prefix]
                    if relation in relation_vectors:
                        relation_vectors[relation].append(v2 - v1)
                    else:
                        relation_vectors[relation] = [v2 - v1]

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
    write_csv(relations_statistics, out_path)
    #print relations_statistics
    print '{0:.2f}'.format(time.time() - start_time), 'seconds'

if __name__ == '__main__':
    main()
