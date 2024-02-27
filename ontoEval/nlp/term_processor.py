from functools import cache

import spacy

spacy_model_name = "en_core_web_sm"
if not spacy.util.is_package(spacy_model_name):
    spacy.cli.download(spacy_model_name)

nlp = spacy.load(spacy_model_name)


@cache
def _lemmatize(word: str) -> str:
    """
    Given a single word, the function returns the lemmatized version of it.
    """
    doc = nlp(word)
    return ''.join([token.lemma_.lower() for token in doc])


@cache
def process(term: str) -> str:
    """
    Given a word or phrase (multi-word), the function applies lemmatization on each of the words in the term.
    The function assumes the words are separated with a single space.
    """
    lemmatized_words = []
    words = term.split(' ')
    for word in words:
        lemmatized_words.append(_lemmatize(word))

    return ' '.join(lemmatized_words)
