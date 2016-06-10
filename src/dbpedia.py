import sparql


SPARQL_ENDPOINT = 'http://dbpedia-live.openlinksw.com/sparql/'
#'http://dbpedia.org/sparql'
# 'http://lod.sztaki.hu/sparql'
DBPEDIA_PREFIX = 'http://dbpedia.org/resource/'


def get_related_entities(base_entity):

    query = ('SELECT ?relation, ?object WHERE { ' + '<' + DBPEDIA_PREFIX + base_entity + '> ?relation ?object . }')
    results = []
    for row in sparql.query(SPARQL_ENDPOINT, query):
        results.append(sparql.unpack_row(row))

    return results
