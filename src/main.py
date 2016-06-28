#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import cPickle
from threading import Thread

import examples
import random
import time
import numpy as np
from os import path
from scipy import spatial
from gensim.models import Word2Vec
from dbpedia import get_relations_from_base_entity, DBPEDIA_PREFIX
from math import isnan
from multiprocessing.pool import ThreadPool



def write_csv(relations_statistics, csv_path):
    with open(csv_path, 'wb') as f:
        writer = csv.writer(f, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        header = ['Relation', 'Count', 'Average cosine sim.', 'Std. Dev. cosine sim.']
        writer.writerow(header)
        for (relation, count, avg_cos, std_cos) in relations_statistics:
            writer.writerow((relation, count, avg_cos, std_cos))


def extract_relations(model_path, n_entities, min_relation_count, out_path, shuffle, dump_vectors):
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
                avg_cos, std_cos = np.mean(cosine_distances), np.std(cosine_distances)
                if not isnan(avg_cos) and not isnan(std_cos):
                    count = len(vectors)
                    relations_statistics.append((relation, count, avg_cos, std_cos))
                    if dump_vectors:
                        mean_relation_vectors[relation] = (np.mean(vectors, axis=0), count, avg_cos, std_cos)

    relations_statistics.sort(key=lambda x: x[2], reverse=True)
    write_csv(relations_statistics, out_path)

    if dump_vectors:
        f = open(out_path + '.vectors.pkl', 'wb')
        cPickle.dump(mean_relation_vectors, f)


def evaluate(model_path, n_entities, vectors_dump, shuffle, topn):

    print 'Loading relation vectors dump'
    f = open(vectors_dump, 'rb')
    w2v_relations = cPickle.load(f)
    print 'Found', len(w2v_relations), 'relations in vectors dump'
    # dict { relation : (vector, count, avg_cos, std_cos) }


    print 'Loading model...'
    model = Word2Vec.load_word2vec_format(model_path, binary=True)
    print 'Finished loading model'



    if n_entities > 0:
        if shuffle:
            base_entities = random.sample(model.vocab.keys(), n_entities)
        else:
            base_entities = model.vocab.keys()[:n_entities]
    else:
        base_entities = model.vocab.keys()

    tp_total, fp_total, fn_total = 0, 0, 0

    for i, base_entity in enumerate(base_entities):
        print i, base_entity

        base_entity_dbpedia_relations = set()
        base_entity_word2vec_relations = set()


        # extract relations according to dbpedia (ground truth)
        print 'extracting dbpedia relations'
        for (relation, related_entity) in get_relations_from_base_entity(base_entity):
            related_entity = unicode(related_entity).encode('utf8')
            related_entity_without_prefix = related_entity[len(DBPEDIA_PREFIX):]
            if related_entity_without_prefix in model:
                print 'found dbpedia relation', relation, related_entity_without_prefix
                base_entity_dbpedia_relations.add((relation, related_entity_without_prefix))

        # extract relations according to word2vec similarity
        print 'extracting word2vec relations'

        for dbpedia_relation in base_entity_dbpedia_relations:

            if dbpedia_relation[0] in w2v_relations:
                (vector, count, avg_cos, std_cos) = w2v_relations[dbpedia_relation[0]]
                #print 'vector', vector
                #print 'base_entity', model[base_entity]
                #print 'base_entity+vector', model[base_entity]+vector
                #relation_applied_to_base_entity = model[base_entity]+vector
                candidate_related_entities = model.most_similar(positive=[model[base_entity], vector], negative=[], topn=topn)
                #print 'vector', vector
                #print 'base_entity', model[base_entity]
                #print 'base_entity+vector', relation_applied_to_base_entity
                #print 'candidate related_entities', candidate_related_entities

                for candidate in candidate_related_entities:
                    (related_entity, sim) = candidate
                    if related_entity != base_entity and sim > 0.9 :
                        #print 'found w2v relation', relation_key, related_entity
                        base_entity_word2vec_relations.add((dbpedia_relation[0], related_entity))

        tp = len(base_entity_dbpedia_relations & base_entity_word2vec_relations)
        fp = len(base_entity_word2vec_relations - base_entity_dbpedia_relations)
        fn = len(base_entity_dbpedia_relations - base_entity_word2vec_relations)

        print base_entity, 'tp=', tp, 'fp=', fp, 'fn=', fn
        print 'true positives:', base_entity_dbpedia_relations & base_entity_word2vec_relations
        #recall = float(tp) / (tp + fp)
        #precision = float(tp) / (tp + fn)

        #print base_entity, 'precision = ', precision, 'recall = ', recall

        tp_total += tp
        fp_total += fp
        fn_total += fn

    #recall_total = float(tp_total) / (tp_total + fp_total)
    #precision_total = float(tp_total) / (tp_total + fn_total)

    #print 'total:', 'precision = ', precision_total, 'recall = ', recall_total


def main():
    start_time = time.time()
    # examples.word2vec_example()
    # examples.sparql_example()
    # examples.dbpedia_example()

    model_path = '/home/garchery/Embeddings/dbpedia_Cats_model_sg_400.bin'
    # model_path = '../data/dbpedia_noCats_model_sg_400.bin'
    # model_path = '../data/WikiEntityModel_400_neg10_iter5.seq'
    out_path = '/home/garchery/word2sem/data/dbpedia_Cats_model_sg_400_10000_min3.csv'
    n_entities = 10000
    min_relation_count = 3
    shuffle = True
    dump_vectors = True

    #extract_relations(model_path, n_entities, min_relation_count, out_path, shuffle, dump_vectors)

    evaluate(model_path, 5, '/home/garchery/word2sem/data/dbpedia_Cats_model_sg_400_10000_min3.csv.vectors.pkl', True, 5)

    print '{0:.1f}'.format(time.time() - start_time), 'seconds'

if __name__ == '__main__':
    main()
