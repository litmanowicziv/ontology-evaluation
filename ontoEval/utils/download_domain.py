import json
from pathlib import Path

from google.cloud import bigquery

SQL = """
SELECT
  lower(name) as concept_name,
  ARRAY_AGG(DISTINCT name) as variations,
  COUNT(DISTINCT paper_object_key) AS occurrences_in_papers,
  ARRAY_AGG(DISTINCT pos) as pos,
  COUNT(1) as occurrences
FROM (
  SELECT
    REPLACE(identifier, CONCAT('.', items[OFFSET(2)],
                               '.', items[OFFSET(1)],
                               '.', items[OFFSET(0)]), ''
    ) AS paper_object_key,
    name,
    pos
  FROM (
    SELECT
      identifier,
      array_reverse(SPLIT(identifier, '.')) AS items,
      name,
      pos
    FROM (
      SELECT
        identifier,
        name,
        pos
      FROM
        `oceanbase-188613.ontology_data_mining.concepts` ) ) )
where length(name) > 2
GROUP BY
  concept_name
HAVING
  occurrences_in_papers > 10
ORDER BY occurrences_in_papers DESC
"""


def download(bigquery_client=bigquery.Client(), target_dir='./../domain'):
    query_job = bigquery_client.query(SQL)
    result_set = list()
    for row in query_job.result():
        result_set.append(dict(row))

    Path(f"{target_dir}/").mkdir(parents=True, exist_ok=True)

    with open(f'{target_dir}/concepts.json', 'w') as f:
        f.write(json.dumps({'concepts': result_set}))


if __name__ == '__main__':
    download()
