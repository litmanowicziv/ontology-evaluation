from pathlib import Path

from dependency_injector.containers import Container
from dependency_injector import providers

from evaluation.nlp.bert import BERT, get_pretrained_model, get_tokenizer

_BASE_PATH = Path(__file__).resolve().parent
_MODEL_PATH = f'{_BASE_PATH}/model'


class Dependency(Container):
    _hf_bert_tokenizer = providers.Singleton(get_tokenizer, model_dir=_MODEL_PATH)
    _hf_bert_model = providers.Singleton(get_pretrained_model, model_dir=_MODEL_PATH)
    bert = providers.Singleton(BERT, model=_hf_bert_model, tokenizer=_hf_bert_tokenizer)
