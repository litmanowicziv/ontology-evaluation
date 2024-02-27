import json
from collections import defaultdict

from rdflib import Graph

from .ontology import sparql
from .nlp.term_processor import process


def process_ontology(path: str, kind='OWL'):
    """
    This function takes an ontology file path as input and loads it via rdflib.
    It then fetches the classes (concepts) and normalizes the textual representation of the concept (the label).
    This normalized form is used as a key to the vocabulary that is structured and the value is a set of identifiers.

    This results in a vocabulary such that all concepts with the same textual representation are stored together.

    <normalized term> -> { set of concept identifiers }

    :param path: The path to the ontology file.
    :param kind: The ontology format (e.g. OWL, SKOS, RDF)
    :return: The ontology as a graph object and the vocabulary.
    """
    g = sparql.graph_from(path)
    classes = sparql.get_classes(g, kind)
    ontology_vocabulary = defaultdict(set)
    for identifier, term in classes:
        normalized = process(term)
        ontology_vocabulary[normalized].add(identifier)

    return g, ontology_vocabulary


def load_domain(file_path: str):
    with open(file_path) as f:
        concepts = json.loads(f.read()).get('concepts')

    domain_lemma_vocabulary = defaultdict(lambda: defaultdict(set))
    # { 'term' : { 'variations': {}, 'semantic_types': {} } }

    for concept in concepts:
        concept_name = concept.get('concept_name')
        variations = concept.get('variations')
        semantic_types = concept.get('pos')
        normalized_name = process(concept_name)
        domain_lemma_vocabulary[normalized_name]['variations'].update(variations)
        domain_lemma_vocabulary[normalized_name]['semantic_types'].update(semantic_types)

    return concepts, domain_lemma_vocabulary


def get_sub_ontology(ontology_graph: Graph,
                     ontology_vocabulary: dict[str, set[str]],
                     common_concepts: set[str]) -> Graph:
    relevant_identifiers = set()
    for concept_name in common_concepts:
        identifiers = ontology_vocabulary[concept_name]
        relevant_identifiers.update(identifiers)

    relevant_relations = []

    concept_pool = set(relevant_identifiers)
    for identifier in relevant_identifiers:
        concept_ancestors, _ = sparql.get_ancestors_of_concept(ontology_graph, identifier)
        concept_pool.update([concept.identifier for concept in concept_ancestors])

    print(len(concept_pool))
    all_relations_tuples = sparql.get_hierarchical_relations(ontology_graph, 'OWL')

    for concept_a, concept_b in all_relations_tuples:
        if ((concept_a.identifier in concept_pool and concept_b.identifier in concept_pool)
                or (concept_b.identifier in relevant_identifiers)):
            relevant_relations.append((concept_a, concept_b))

    # build sub ontology from overlapping concepts
    sub_ontology_graph = sparql.create_ontology(relevant_relations)
    return sub_ontology_graph
