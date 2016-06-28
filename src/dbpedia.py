import sparql


#SPARQL_ENDPOINT = 'http://dbpedia-live.openlinksw.com/sparql/'
SPARQL_ENDPOINT = 'http://dbpedia.org/sparql'
# 'http://lod.sztaki.hu/sparql'
DBPEDIA_PREFIX = 'http://dbpedia.org/resource/'


def get_relations_from_base_entity(base_entity):
    results = []
    try:
        query = ('SELECT ?relation, ?object WHERE { ' + '<' + DBPEDIA_PREFIX + base_entity + '> ?relation ?object . }')
        for row in sparql.query(SPARQL_ENDPOINT, query):
            results.append(sparql.unpack_row(row))
    except Exception as e:
        print 'Error while getting relations from entity', base_entity
        print str(e)

    return results


def get_relations_to_base_entity(base_entity):
    results = []
    try:
        query = ('SELECT ?subject, ?relation WHERE { ?subject ?relation <' + DBPEDIA_PREFIX + base_entity + '> . }')
        for row in sparql.query(SPARQL_ENDPOINT, query):
            results.append(sparql.unpack_row(row))
    except Exception as e:
        print 'Error while getting relations to entity', base_entity
        print str(e)

    return results
