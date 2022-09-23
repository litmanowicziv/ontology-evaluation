# github-ontology-evaluation

### Requirements
Install Python 3.9

Install dependencies
```shell
pip install -r requirements.txt
```

### Running
Run `main.py`

Change the ontology being evaluated by updating the file name:
```python
def main():
    ontology_file = 'envo.owl'
```

### BERT Pre-training
The code that was used to pre-train the model is available under `bert-pre-training`.
The text that was ujsed is available under `artifacts/text`. It is the combination of all papers in a single text file.