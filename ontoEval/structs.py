from dataclasses import dataclass

import torch
from rdflib import URIRef

from .nlp.bert import BERT


@dataclass(unsafe_hash=True)
class Concept:
    identifier: str
    representation: str


@dataclass(unsafe_hash=True)
class Relation:
    type: str
    domain: str  # from
    range: str  # to

    def get_as_tuple(self):
        return URIRef(self.domain), URIRef(self.type), URIRef(self.range)


@dataclass()
class ConceptFamily:
    parent: Concept
    children: set[Concept]


class ModelKnowledge:
    concept_to_vec: dict[str, torch.tensor]
    concept_to_type: dict[str, set[str]]

    _model: BERT
