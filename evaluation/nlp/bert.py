from collections import defaultdict
from dataclasses import dataclass
from transformers import pipeline

import torch
from torch import tensor, Tensor
from transformers import BertForMaskedLM, RobertaTokenizer, PreTrainedTokenizer


def get_tokenizer(model_dir: str):
    return RobertaTokenizer.from_pretrained(f'{model_dir}/tokenizer/', max_len=512)


def get_pretrained_model(model_dir: str):
    return BertForMaskedLM.from_pretrained(f'{model_dir}/',
                                           return_dict=True,
                                           output_hidden_states=True,
                                           output_attentions=True)


def get_attention_nth_layer_mth_head_kth_token(
        attention_outputs, n, m, k, average_heads=False, logger=None
):
    """
    Function to compute attention weights by:
    i)   Take the attention weights from the nth multi-head attention layer assigned to kth token
    ii)  Take the mth attention head
    """
    if average_heads is True and m is not None:
        logger.warning(
            "Argument passed for param @m will be ignored because of head averaging."
        )

    # Get the attention weights outputted by the nth layer
    attention_outputs_concatenated = torch.cat(
        attention_outputs, dim=0
    )  # (K, N, P, P) (12, 12, P, P)
    attention_outputs = attention_outputs_concatenated.data[
                        n, :, :, :
                        ]  # (N, P, P) (12, P, P)

    # Get the attention weights assigned to kth token
    attention_outputs = attention_outputs[:, k, :]  # (N, P) (12, P)

    # Compute the average attention weights across all attention heads
    if average_heads:
        attention_outputs = torch.sum(attention_outputs, dim=0)  # (P)
        num_attention_heads = attention_outputs_concatenated.shape[1]
        attention_outputs /= num_attention_heads
    # Get the attention weights of mth head
    else:
        attention_outputs = attention_outputs[m, :]  # (P)

    return attention_outputs


def get_normalized_attention(
        attention_outputs,
        method="last_layer_heads_average",
        normalization_method="normal",
        token=0,
        logger=None,
):
    """
    Function to get the normalized version of the attention output
    """
    assert method in (
        "first_layer_heads_average",
        "last_layer_heads_average",
        "nth_layer_heads_average",
        "nth_layer_mth_head",
        "custom",
    )
    assert normalization_method in ("normal", "min-max")

    attention_weights = None
    if method == "first_layer_heads_average":
        attention_weights = get_attention_nth_layer_mth_head_kth_token(
            attention_outputs=attention_outputs,
            n=0,
            m=None,
            k=token,
            average_heads=True,
            logger=logger,
        )
    elif method == "last_layer_heads_average":
        attention_weights = get_attention_nth_layer_mth_head_kth_token(
            attention_outputs=attention_outputs,
            n=-1,
            m=None,
            k=token,
            average_heads=True,
            logger=logger,
        )
    elif method == "nth_layer_heads_average":
        n = 5
        attention_weights = get_attention_nth_layer_mth_head_kth_token(
            attention_outputs=attention_outputs,
            n=n,
            m=None,
            k=token,
            average_heads=True,
            logger=logger,
        )
    elif method == "nth_layer_mth_head":
        n = -1
        m = -1
        attention_weights = get_attention_nth_layer_mth_head_kth_token(
            attention_outputs=attention_outputs,
            n=n,
            m=m,
            k=0,
            average_heads=False,
            logger=logger,
        )
    elif method == "custom":
        n = -1
        m = -1
        k = 0
        attention_weights = get_attention_nth_layer_mth_head_kth_token(
            attention_outputs=attention_outputs,
            n=n,
            m=m,
            k=k,
            average_heads=False,
            logger=logger,
        )

    # Remove the beginning [CLS] & ending [SEP] tokens for better intuition
    attention_weights = attention_weights[1:-1]

    # Apply normalization methods to attention weights
    if normalization_method == "min-max":  # Min-Max Normalization
        max_weight, min_weight = (
            attention_weights.max(),
            attention_weights.min(),
        )

        attention_weights = (attention_weights - min_weight) / (
                max_weight - min_weight
        )
    elif normalization_method == "normal":  # Z-Score Normalization
        mu, std = attention_weights.mean(), attention_weights.std()
        attention_weights = (attention_weights - mu) / std

    # Convert tensor to NumPy array
    attention_weights = attention_weights.data

    return attention_weights


@dataclass
class BERT:
    model: BertForMaskedLM
    tokenizer: PreTrainedTokenizer

    def __post_init__(self):
        self._pipeline = pipeline("fill-mask", model=self.model, tokenizer=self.tokenizer)

    def get_attention_map(self, text: str) -> dict[str, dict[str, float]]:
        self.model.eval()
        token_map = {
            x: self.tokenizer.encode(x, add_special_tokens=False, add_prefix_space=i > 0) for i, x in
            enumerate(text.split())
        }
        token_id_to_text = {}
        for word, tokens in token_map.items():
            for token in tokens:
                token_id_to_text[token] = word

        encoded_inputs = self.tokenizer(text)
        tokens_tensor = torch.tensor([encoded_inputs['input_ids']])
        segments_tensors = torch.tensor([encoded_inputs['attention_mask']])

        # Run the text through BERT, and collect all of the hidden states produced
        with torch.no_grad():
            outputs = self.model(tokens_tensor, segments_tensors)
            attention = outputs.attentions

        result = dict()
        sentence_tokens = encoded_inputs['input_ids'][1:-1]

        # for each token in the sentence compute the attention of all the other tokens.
        for i, token_id in enumerate(sentence_tokens):
            normalized_attention = get_normalized_attention(attention,
                                                            method='first_layer_heads_average',
                                                            normalization_method='min-max',
                                                            token=i)
            current_token = token_id_to_text.get(token_id)
            result[current_token] = {
                token_id_to_text.get(tkid): float(attention)
                for tkid, attention in zip(sentence_tokens, normalized_attention)
            }

        return result

    def get_attention_average(self, text: str):
        result = self.get_attention_map(text)
        scores = defaultdict(list)
        for entry in result.values():
            for token, value in entry.items():
                scores[token].append(value)

        return {
            k: sum(v) / len(v)
            for k, v in scores.items()
        }

    def get_hidden_state_vector(self, text: str, layers: list[int] = None) -> Tensor:
        if layers is None:
            layers = [-4, -3, -2, -1]

        self.model.eval()
        encoded_inputs = self.tokenizer(text)

        tokens_tensor = torch.tensor([encoded_inputs['input_ids']])
        segments_tensors = torch.tensor([encoded_inputs['attention_mask']])

        # Run the text through BERT, and collect all of the hidden states produced

        with torch.no_grad():
            outputs = self.model(tokens_tensor, segments_tensors)
            states = outputs.hidden_states

        output = torch.stack([states[i] for i in layers]).sum(0).squeeze()
        return output.mean(dim=0)

    def get_tensor_similarity(self, v1: Tensor, v2: Tensor):
        cos = torch.nn.CosineSimilarity(dim=0)
        return cos(v1, v2)

    def fill_mask(self, text: str):
        return self._pipeline(text)