import sparql
from gensim.models.word2vec import LineSentence, Word2Vec
from os.path import join, dirname


def word2vec_example():
    file = join(dirname(__file__), '../data', 'raw_sentences.txt')
    with open(file, 'r') as corpus:
        lines = []
        for line in corpus:
            lines.append(line)

    sentences = LineSentence(join(dirname(__file__), '../data', 'enwik8_clean.txt'))
    model = Word2Vec(sentences, size=400, window=8, min_count=5, workers=4, sg=1)
    # for word in model.vocab:
    #     print word

    print model.most_similar('france')
    print model.most_similar(positive=['woman', 'king'], negative=['man'])

def sparql_example():
    q = ('SELECT DISTINCT ?station, ?orbits WHERE { '
    '?station a <http://dbpedia.org/ontology/SpaceStation> . '
    '?station <http://dbpedia.org/property/orbits> ?orbits . '
    'FILTER(?orbits > 50000) } ORDER BY DESC(?orbits)')
    result = sparql.query('http://dbpedia.org/sparql', q)
    for row in result:
        print 'row:', row
        values = sparql.unpack_row(row)
        print values[0], "-", values[1], "orbits"