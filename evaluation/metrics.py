from typing import Optional

from evaluation.nlp.bert import BERT
from evaluation.structs import Concept, ConceptFamily


def children_similarity_score(bert: BERT, concept_family: ConceptFamily) -> Optional[float]:
    """
    Given a set of concepts, compute the sibling similarity score.
    """
    if len(concept_family.children) < 2:
        return None

    concept_to_vec = {
        concept.identifier: bert.get_hidden_state_vector(concept.representation)
        for concept in concept_family.children
    }

    concept_means = {}
    for identifier, vector in concept_to_vec.items():
        scores = {bert.get_tensor_similarity(vector, v) for k, v in concept_to_vec.items() if k != identifier}
        mean = sum(scores) / len(scores)
        concept_means[identifier] = mean

    mean_of_means = sum(concept_means.values()) / len(concept_means)
    return float(mean_of_means)


def parent_similarity_score(bert: BERT, concept_family: ConceptFamily):
    """
    Given a concept family (parent and direct children) calculate the mean similarity between all children and the parent.
    """

    if len(concept_family.children) < 2:
        return None

    parent_vec = bert.get_hidden_state_vector(concept_family.parent.representation)
    vectors = {
        bert.get_hidden_state_vector(concept.representation)
        for concept in concept_family.children
    }

    scores = {bert.get_tensor_similarity(parent_vec, vec) for vec in vectors}
    mean = sum(scores) / len(scores)

    return float(mean)


def parent_difference_agreement(bert: BERT, concept_family: ConceptFamily):
    """
    Given a concept family (parent and direct children), calculate the similarity difference agreement between
    the child concepts and the parent.

    This formula can be represented as (1 - std of scores)
    where each score is the cosine similarity of the encoding between each child and the parent.

    A high number (close to 1) indicates there is a total agreement on the similarities across children.
    This can be interpreted as all children are different from the parent in a similar fashion.
    A low number (indicating high variance) meaning the child-parent similarities vary.
    """

    if len(concept_family.children) < 2:
        return None

    parent_vec = bert.get_hidden_state_vector(concept_family.parent.representation)
    vectors = {
        bert.get_hidden_state_vector(concept.representation)
        for concept in concept_family.children
    }

    scores = {bert.get_tensor_similarity(parent_vec, vec) for vec in vectors}
    mean = sum(scores) / len(scores)

    std = (sum([(score - mean) ** 2 for score in scores]) / (len(scores) - 1)) ** 0.5
    return float(1 - std)
