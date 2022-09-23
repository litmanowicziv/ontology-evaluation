import json
from collections import defaultdict

import pandas as pd
from rdflib import Graph

from evaluation import ontology
from evaluation.ontology import sparql, walk
from evaluation.dependencies import Dependency, _BASE_PATH
from evaluation.nlp.term_processor import process
from evaluation import metrics

dependencies = Dependency()


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
    g = ontology.sparql.graph_from(path)
    classes = ontology.sparql.get_classes(g, kind)
    ontology_vocabulary = defaultdict(set)
    for identifier, term in classes:
        normalized = process(term)
        ontology_vocabulary[normalized].add(identifier)

    return g, ontology_vocabulary


def load_domain():
    with open(f'{_BASE_PATH}/domain/concepts.json') as f:
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
        concept_ancestors, _ = ontology.sparql.get_ancestors_of_concept(ontology_graph, identifier)
        concept_pool.update([concept.identifier for concept in concept_ancestors])

    print(len(concept_pool))
    all_relations_tuples = ontology.sparql.get_hierarchical_relations(ontology_graph, 'OWL')

    for concept_a, concept_b in all_relations_tuples:
        if (
                (concept_a.identifier in concept_pool and concept_b.identifier in concept_pool)
                or
                (concept_b.identifier in relevant_identifiers)
        ):
            relevant_relations.append((concept_a, concept_b))

    # build sub ontology from overlapping concepts
    sub_ontology_graph = ontology.sparql.create_ontology(relevant_relations)
    return sub_ontology_graph


def main():
    ontology_file = 'envo.owl'
    print('Reading ontology...')
    ontology_graph, ontology_vocabulary = process_ontology(f'../artifacts/ontologies/{ontology_file}')
    domain_concepts, domain_vocabulary = load_domain()

    # determine overlapping concepts
    common_concepts = set(domain_vocabulary.keys()).intersection(set(ontology_vocabulary.keys()))
    print(common_concepts)
    print(len(common_concepts))
    sub_ontology_graph = get_sub_ontology(ontology_graph, ontology_vocabulary, common_concepts)
    sub_ontology_graph.serialize(format="pretty-xml", destination=f'../{ontology_file}.ontology.xml')

    # calculate percentage
    analyze_completeness(common_concepts, sub_ontology_graph, domain_vocabulary)

    # evaluate sub ontology
    analyze_correctness(ontology_file, sub_ontology_graph)


def analyze_completeness(common_concepts, sub_ontology, domain_vocabulary):
    concepts, _ = ontology.sparql.get_concepts(sub_ontology)

    print('Ontology Relevance: ', len(common_concepts) / len(concepts))
    print('Domain Coverage: ', len(common_concepts) / len(domain_vocabulary))


def analyze_correctness(name: str, sub_ontology_graph: Graph):
    concept_families = ontology.walk.get_concept_families(sub_ontology_graph)

    bert = dependencies.bert()

    results = []
    for family in concept_families:
        css = metrics.children_similarity_score(bert, family)
        pss = metrics.parent_similarity_score(bert, family)
        psd = metrics.parent_difference_agreement(bert, family)
        print(family.parent.representation, len(family.children), css, pss, psd)
        results.append({
            'family_root': family.parent.identifier,
            'family_size': len(family.children) + 1,
            'children_similarity_score': css,
            'parent_similarity_score': pss,
            'parent_difference_agreement': psd
        })

    df = pd.DataFrame(results)
    df.to_csv(f'../{name}.evaluation.csv')

    relevant_entries = df[df['family_size'] > 2]
    print('children_similarity_score mean:', relevant_entries['children_similarity_score'].mean())
    print('parent_similarity_score mean:', relevant_entries['parent_similarity_score'].mean())
    print('parent_difference_agreement mean:', relevant_entries['parent_difference_agreement'].mean())


if __name__ == '__main__':
    main()
