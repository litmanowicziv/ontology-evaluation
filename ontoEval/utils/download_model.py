from google.cloud import storage
from pathlib import Path


def download(target_dir='./../model'):
    Path(f"{target_dir}/tokenizer/").mkdir(parents=True, exist_ok=True)

    client = storage.Client()
    input_bucket = client.get_bucket('oceanic-ner-model')

    input_bucket.blob('bert/pytorch_model.bin').download_to_filename(f'{target_dir}/pytorch_model.bin')
    input_bucket.blob('bert/training_args.bin').download_to_filename(f'{target_dir}/training_args.bin')
    input_bucket.blob('bert/config.json').download_to_filename(f'{target_dir}/config.json')

    input_bucket.blob('bert/tokenizer/merges.txt').download_to_filename(f'{target_dir}/tokenizer/merges.txt')
    input_bucket.blob('bert/tokenizer/tokenizer.json').download_to_filename(f'{target_dir}/tokenizer/tokenizer.json')
    input_bucket.blob('bert/tokenizer/vocab.json').download_to_filename(f'{target_dir}/tokenizer/vocab.json')


if __name__ == '__main__':
    download()
