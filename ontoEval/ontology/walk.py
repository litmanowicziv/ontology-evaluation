from collections import defaultdict

from rdflib import Graph

from .sparql import get_superclasses_of_concept, get_subclasses_of_concept, get_relations
from ..structs import Concept, Relation, ConceptFamily


def get_superclass_concepts_and_relations(g: Graph, root: Concept,
                                          collected_concepts: list[Concept],
                                          collected_relations: list[Relation]):
    """
    Collects all ancestors recursively
    """
    super_classes, super_relations = get_superclasses_of_concept(g, root.identifier)

    for sup in super_classes:
        if sup in collected_concepts:
            continue
        get_superclass_concepts_and_relations(g, root=sup,
                                              collected_concepts=collected_concepts,
                                              collected_relations=collected_relations)
    collected_concepts.extend(cls for cls in super_classes if cls not in collected_concepts)
    collected_relations.extend(rel for rel in super_relations if rel not in collected_relations)


def get_concepts_and_relations(g: Graph, root: Concept,
                               degrees: int,
                               collected_concepts: list[Concept],
                               collected_relations: list[Relation]):
    """
    Collects all descendants recursively - based on number of degrees.
    """
    if degrees == 0:
        return

    sub_classes, sub_relations = get_subclasses_of_concept(g, root.identifier)

    for sub in sub_classes:
        get_concepts_and_relations(g, root=sub,
                                   degrees=degrees - 1,
                                   collected_concepts=collected_concepts,
                                   collected_relations=collected_relations)

    collected_concepts.extend(cls for cls in sub_classes if cls not in collected_concepts)
    collected_relations.extend(rel for rel in sub_relations if rel not in collected_relations)


def get_concept_families(g: Graph) -> list[ConceptFamily]:
    families = defaultdict(set)
    for c1, c2 in get_relations(g):
        families[c2].add(c1)

    return [ConceptFamily(entry, concepts) for entry, concepts in families.items()]
