
## Mandatory:

Use KB_entity_type.ipynb to obtain the entity type sets and the meta proximity graph without entity types aligned.


## Optional:

Use `python prompy_gen.py` to generate the prompt for LLMs to align the types between two KGs.

Obtain the aligned type pairs from any LLM (e.g., Claude) and store the aligned type pairs in `matched_types.txt`.

Replace the types in the meta proximity graph with the aligned types using `python match_type.py`.
