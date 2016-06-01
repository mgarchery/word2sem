import sparql


SPARQL_ENDPOINT = 'http://dbpedia.org/sparql'


def get_related_entities(base_entity):
    query = ('SELECT ?relation, ?object WHERE { '
         +'<' + base_entity + '> ?relation ?object . }')

    results = []
    for row in sparql.query(SPARQL_ENDPOINT, query):
        results.append(sparql.unpack_row(row))

    return results