import re

from rdflib import Graph, Literal, URIRef, RDFS, RDF, OWL, SKOS
from rdflib.query import Result

from ..structs import Concept, Relation

namespaces = """
    prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    prefix owl: <http://www.w3.org/2002/07/owl#>
    prefix ns1: <http://www.w3.org/2004/02/skos/core#>
    prefix ns2: <http://www.w3.org/2006/time#>
"""


def graph_from(path: str) -> Graph:
    g = Graph()
    g.parse(path)
    return g


def get_classes(graph: Graph, kind: str) -> list[tuple[str, str]]:
    return {
        'owl': _owl_get_classes,
        'skos': _skos_get_classes,
        'rdf': _rdf_get_classes
    }[kind.lower()](graph)


def get_hierarchical_relations(graph: Graph, kind: str) -> list[tuple[Concept, Concept]]:
    return {
        'owl': _owl_get_hierarchical_relations,
        'skos': _skos_get_hierarchical_relations
    }[kind.lower()](graph)


def _define_concept(graph: Graph, concept: Concept):
    identifier = concept.identifier
    label = Literal(concept.representation)

    this = URIRef(identifier)
    graph.add((this, RDFS.label, label))
    graph.add((this, RDF.type, OWL.Class))

    return this


def _add_relation(graph: Graph, concept_a: str, concept_b: str):
    graph.add(
        (URIRef(concept_a), RDFS.subClassOf, URIRef(concept_b))
    )


def create_ontology(relations: list[tuple[Concept, Concept]]):
    g = Graph()
    unique_concepts = dict()
    for a, b in relations:
        unique_concepts[a.identifier] = a
        unique_concepts[b.identifier] = b


    for concept in unique_concepts.values():
        _define_concept(g, concept)

    for a, b in relations:
        _add_relation(g, a.identifier, b.identifier)

    print(f'UNIQUE CONCEPTS: {len(unique_concepts)}')
    return g


##
# OWL
##

def _owl_get_classes(graph: Graph) -> list[tuple[str, str]]:
    results = graph.query(
        f"""
            {namespaces}

            SELECT distinct ?id ?label WHERE {{
                ?id rdfs:label ?label .
                ?id a owl:Class
            }}
        """
    )
    return [(result.id.toPython(), result.label.value) for result in results]


def _owl_get_hierarchical_relations(graph: Graph) -> list[tuple[Concept, Concept]]:
    results = graph.query(
        f"""
                {namespaces}

                SELECT distinct ?c1 ?label1 ?c2 ?label2 WHERE {{
                    ?c1 rdfs:label ?label1 .
                    ?c1 rdfs:subClassOf ?c2 .
                    ?c2 rdfs:label ?label2 .
                    ?c1 a owl:Class
                }}
            """
    )
    return [(Concept(result.c1.toPython(), result.label1.value),
             Concept(result.c2.toPython(), result.label2.value)) for result in results]


##
# SKOS
#

def _skos_get_classes(graph: Graph) -> list[tuple[str, str]]:
    results = graph.query(
        f"""
            {namespaces}

            SELECT distinct ?id ?label WHERE {{
                ?id skos:prefLabel ?label
            }}
        """
    )
    return [(result.id.value, result.label.value) for result in results]


def _skos_get_hierarchical_relations(graph: Graph) -> list[tuple[Concept, Concept]]:
    results = graph.query(
        f"""
            {namespaces}

            SELECT distinct ?label1 ?label2 WHERE {{
                ?c1 skos:prefLabel ?label1 .
                ?c1 skos:broader ?c2 .
                ?c2 skos:prefLabel ?label2
            }}
        """
    )
    return [(Concept(result.c1.toPython(), result.label1.value),
             Concept(result.c2.toPython(), result.label2.value)) for result in results]


##
# RDF
#

def _rdf_get_classes(graph: Graph) -> list[str]:
    results = graph.query(
        f"""
            {namespaces}

            SELECT distinct ?id ?label WHERE {{
                ?id rdfs:label ?label
            }}
        """
    )
    return [result.label.value for result in results]


##
# GENERAL
#

def _parse_concepts(results: Result):
    return [Concept(identifier=result.id.toPython(),
                    representation=result.label.value)
            for result in results]


def get_concept(graph: Graph, identifier: str):
    results = graph.query(
        f"""
                {namespaces}

                SELECT distinct ?id ?label ?comment ?definition ?hasTime WHERE {{
                    BIND(<{identifier}> AS ?id)
                    ?id rdfs:label ?label .
                }}
            """
    )
    concepts = _parse_concepts(results)
    return concepts[0] if concepts else None


def get_concepts(graph: Graph):
    results = graph.query(
        f"""
            {namespaces}

            SELECT distinct ?id ?label WHERE {{
                ?id rdfs:label ?label .
            }}
        """
    )
    return _parse_concepts(results), []


def get_root_concepts(graph: Graph):
    results = graph.query(
        f"""
            {namespaces}

            SELECT distinct ?id ?label ?comment ?definition ?hasTime WHERE {{
                ?id rdfs:label ?label .
                ?o3 rdfs:subClassOf ?id
                FILTER NOT EXISTS {{ 
                    ?id rdfs:subClassOf ?o2 
                }}
            }}
        """
    )
    return _parse_concepts(results), []


def get_subclasses_of_concept(graph: Graph, identifier: str):
    results = graph.query(
        f"""
            {namespaces}

            SELECT distinct ?id ?label WHERE {{
                ?id rdfs:label ?label .
                ?id rdfs:subClassOf <{identifier}>
            }}
        """
    )
    concepts = _parse_concepts(results)
    relations = [Relation(type='http://www.w3.org/2000/01/rdf-schema#subClassOf',
                          domain=result.id.toPython(),
                          range=identifier)
                 for result in results]
    return concepts, relations


def get_superclasses_of_concept(graph: Graph, identifier: str):
    results = graph.query(
        f"""
                {namespaces}

                SELECT distinct ?id ?label ?comment ?definition ?hasTime WHERE {{
                    ?id rdfs:label ?label .
                    <{identifier}> rdfs:subClassOf ?id
                }}
            """
    )
    concepts = _parse_concepts(results)
    relations = [Relation(type='http://www.w3.org/2000/01/rdf-schema#subClassOf',
                          domain=identifier,
                          range=result.id.toPython())
                 for result in results]
    return concepts, relations


def get_ancestors_of_concept(graph: Graph, identifier: str):
    results = graph.query(
        f"""
                {namespaces}

                SELECT distinct ?id ?label WHERE {{
                    ?id rdfs:label ?label .
                    <{identifier}> rdfs:subClassOf+ ?id
                }}
            """
    )
    concepts = _parse_concepts(results)
    return concepts, []


def get_relations_of_concept(graph: Graph, identifier: str):
    """
    This returns a list of triples of relations that are not subClassOf between the given identifier and other concepts.
    """
    results = graph.query(
        f"""
                {namespaces}

                SELECT distinct ?id ?label ?comment ?definition ?hasTime ?relation WHERE {{
                    <{identifier}> ?relation ?id .
                    ?id rdfs:label ?label .
                    FILTER NOT EXISTS {{ <{identifier}> rdfs:subClassOf ?id }}
                }}
            """
    )
    relations = [Relation(type=result.relation.toPython(),
                          domain=identifier,
                          range=result.id.toPython())
                 for result in results]
    return _parse_concepts(results), relations


def get_relations(graph: Graph) -> list[tuple[Concept, Concept]]:
    results = graph.query(
        f"""
                {namespaces}

                SELECT distinct ?c1 ?label1 ?c2 ?label2 WHERE {{
                    ?c1 rdfs:label ?label1 .
                    ?c1 rdfs:subClassOf ?c2 .
                    ?c2 rdfs:label ?label2 .
                }}
            """
    )
    return [(Concept(result.c1.toPython(), result.label1.value),
             Concept(result.c2.toPython(), result.label2.value)) for result in results]
