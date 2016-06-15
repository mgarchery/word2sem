import sparql


SPARQL_ENDPOINT = 'http://dbpedia-live.openlinksw.com/sparql/'
#'http://dbpedia.org/sparql'
# 'http://lod.sztaki.hu/sparql'
DBPEDIA_PREFIX = 'http://dbpedia.org/resource/'


def get_related_entities(base_entity):
    results = []
    try:
        query = ('SELECT ?relation, ?object WHERE { ' + '<' + DBPEDIA_PREFIX + base_entity + '> ?relation ?object . }')

        for row in sparql.query(SPARQL_ENDPOINT, query):
            results.append(sparql.unpack_row(row))
    except Exception as e:
        print 'Error while getting relations for entity', base_entity
        print e

    return results
