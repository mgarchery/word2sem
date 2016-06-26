#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import cPickle
import examples
import random
import time
import numpy as np
from os import path
from scipy import spatial
from gensim.models import Word2Vec
from dbpedia import get_relations_from_base_entity, DBPEDIA_PREFIX


def write_csv(relations_statistics, csv_path):
    with open(csv_path, 'wb') as f:
        writer = csv.writer(f, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        header = ['Relation', 'Count', 'Average cosine sim.', 'Std. Dev. cosine sim.']
        writer.writerow(header)
        for (relation, count, avg_cos, std_cos) in relations_statistics:
            writer.writerow((relation, count, avg_cos, std_cos))


def extract_relations(n_entities, min_relation_count, model_path, out_path, shuffle, generate_relation_vectors):
    print 'Loading model...'
    model = Word2Vec.load_word2vec_format(model_path, binary=True)
    print 'Finished loading model'

    relation_vectors = dict()
    if n_entities > 0:
        if shuffle:
            base_entities = random.sample(model.vocab.keys(), n_entities)
        else:
            base_entities = model.vocab.keys()[:n_entities]
    else:
        base_entities = model.vocab.keys()

    for i, base_entity in enumerate(base_entities):
        print i, base_entity
        for (relation, related_entity) in get_relations_from_base_entity(base_entity):
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
    mean_relation_vectors = dict()

    for relation in relation_vectors:
        vectors = relation_vectors[relation]

        if len(vectors) > min_relation_count:
            cosine_distances = []
            for i, vi in enumerate(vectors):
                for j, vj in enumerate(vectors[i + 1:]):
                    cosine_distances.append(1.0 - spatial.distance.cosine(vi, vj))
            if len(cosine_distances) > 1:
                # print cosine_distances
                avg_cos, std_cos = np.mean(cosine_distances), np.std(cosine_distances)
                count = len(vectors)
                # print relation, 'count', count , ' / avg cos_sim', avg_cos, ' / std cos_sim', std_cos
                relations_statistics.append((relation, count, avg_cos, std_cos))

            if generate_relation_vectors:
                mean_relation_vectors[relation] = np.mean(vectors, axis=0)

    relations_statistics.sort(key=lambda x: x[2], reverse=True)
    write_csv(relations_statistics, out_path)

    if generate_relation_vectors:
        f = open(out_path + '.vectors.pkl', 'wb')
        cPickle.dump(mean_relation_vectors, f)


def main():
    start_time = time.time()
    # examples.word2vec_example()
    # examples.sparql_example()
    # examples.dbpedia_example()

    model_path = '../data/dbpedia_Cats_model_sg_400.bin'
    # model_path = '../data/dbpedia_noCats_model_sg_400.bin'
    # model_path = '../data/WikiEntityModel_400_neg10_iter5.seq'
    out_path = '../data/word2sem_.csv'
    n_entities = 10
    min_relation_count = 3
    shuffle = True
    generate_relation_vectors = True

    extract_relations(n_entities, min_relation_count, model_path, out_path, shuffle, generate_relation_vectors)

    print '{0:.1f}'.format(time.time() - start_time), 'seconds'

if __name__ == '__main__':
    main()
