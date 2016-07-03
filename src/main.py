#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import cPickle
from threading import Thread

import examples
import random
import time
import numpy as np
import operator
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
            relation = unicode(relation).encode('utf8')
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

    n_relations = len(relation_vectors)
    for r, relation in enumerate(relation_vectors):
        print r, '/', n_relations, relation
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

    print 'Sorting relations'
    relations_statistics.sort(key=lambda x: x[2], reverse=True)
    print 'Writing to csv'
    write_csv(relations_statistics, out_path)

    if dump_vectors:
        print 'Writing vectors dump'
        f = open(out_path + '.vectors.pkl', 'wb')
        cPickle.dump(mean_relation_vectors, f)


def evaluate(model_path, n_entities, n_relations, vectors_dump, shuffle, topn):

    print 'Loading relation vectors dump'
    f = open(vectors_dump, 'rb')
    w2v_relations = cPickle.load(f)
    print 'Found', len(w2v_relations), 'relations in vectors dump'
    # dict { relation : (vector, count, avg_cos, std_cos) }

    print 'Loading model...'
    model = Word2Vec.load_word2vec_format(model_path, binary=True)
    print 'Finished loading model'

    #relation_types = set(sorted(w2v_relations.keys(), key=lambda x: w2v_relations[x][2], reverse=True)[:n_relations])  # use n_relations best relations 
    #print 'Looking for relation types:'
    #for rt in relation_types:
        #print rt, '(avg cos sim =', w2v_relations[rt][2], ')'


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
        
        relation_types = set()

        # extract relations according to dbpedia (ground truth)
        #print 'extracting dbpedia relations'
        for (relation, related_entity) in get_relations_from_base_entity(base_entity):
            related_entity = unicode(related_entity).encode('utf8')
            related_entity_without_prefix = related_entity[len(DBPEDIA_PREFIX):]
            if related_entity_without_prefix in model:
                #print 'found dbpedia relation', relation, related_entity_without_prefix
                base_entity_dbpedia_relations.add((relation, related_entity_without_prefix))
                relation_types.add(relation)
        # extract relations according to word2vec similarity
        # print 'extracting word2vec relations'

        for relation_type in relation_types:
            if relation_type in w2v_relations:
                (vector, count, avg_cos, std_cos) = w2v_relations[relation_type]
                #print 'vector', vector
                #print 'base_entity', model[base_entity]
                #print 'base_entity+vector', model[base_entity]+vector
                #relation_applied_to_base_entity = model[base_entity]+vector
                candidate_related_entities = model.most_similar(positive=[model[base_entity], vector], negative=[], topn=topn)
                #print 'vector', vector
                #print 'base_entity', model[base_entity]
                #print 'base_entity+vector', relation_applied_to_base_entity
                #print 'candidate related_entities', candidate_related_entities

                for (related_entity, sim) in candidate_related_entities:
                    if related_entity != base_entity and sim > avg_cos - std_cos:
                        #print 'found w2v relation', relation_key, related_entity
                        base_entity_word2vec_relations.add((relation_type, related_entity))

        tp = len(base_entity_dbpedia_relations & base_entity_word2vec_relations)
        fp = len(base_entity_word2vec_relations - base_entity_dbpedia_relations)
        fn = len(base_entity_dbpedia_relations - base_entity_word2vec_relations)

        print base_entity, 'tp=', tp, 'fp=', fp, 'fn=', fn
        # print 'true positives:', base_entity_dbpedia_relations & base_entity_word2vec_relations
        # print 'false negatives:', base_entity_dbpedia_relations - base_entity_word2vec_relations
        #recall = float(tp) / (tp + fn)
        #precision = float(tp) / (tp + fp)

        #print base_entity, 'precision = ', precision, 'recall = ', recall

        tp_total += tp
        fp_total += fp
        fn_total += fn

    precision_total = float(tp_total) / (tp_total + fp_total)
    recall_total = float(tp_total) / (tp_total + fn_total)
    f1_total = 2 * precision_total * recall_total / (precision_total + recall_total)

    print 'Overall evaluation:'
    print 'precision =', round(precision_total, 3)
    print 'recall =', round(recall_total, 3)
    print 'F1 =', round(f1_total,3)


def main():
    start_time = time.time()
    # examples.word2vec_example()
    # examples.sparql_example()
    # examples.dbpedia_example()

    model_path = '/home/garchery/Embeddings/dbpedia_Cats_model_sg_400.bin'
    # model_path = '/home/garchery/Embeddings/dbpedia_noCats_model_sg_400.bin'
    # model_path = '/home/garchery/Embeddings/WikiEntityModel_400_neg10_iter5.seq'
    out_path = '/home/garchery/word2sem/data/wiki_50000_min3.csv'
    n_entities = 50000
    min_relation_count = 3
    shuffle = True
    dump_vectors = True

    #extract_relations(model_path, n_entities, min_relation_count, out_path, shuffle, dump_vectors)

    n_entities = 500
    n_relations = 20 # -1 for all possible relation types
    word2vec_similar_topn = 2
    relation_vectors_dump = '/home/garchery/word2sem/data/dbpedia_cats_50000_min3.csv.vectors.pkl'
    evaluate(model_path, n_entities, n_relations, relation_vectors_dump, True, word2vec_similar_topn)

    print '{0:.1f}'.format(time.time() - start_time), 'seconds'

if __name__ == '__main__':
    main()
