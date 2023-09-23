
## Create the meta proximity graph without entity types aligned:

1. Use KB_entity_type.ipynb to obtain the entity type sets and the meta proximity graph without entity types aligned.


## Match the entity types using LLMs:

1. Use `python prompt_gen.py` to generate the prompt for LLMs to align the types between two KGs.

2. Obtain the aligned type pairs from any LLM (e.g., Claude) and store the aligned type pairs in `matched_types.txt`.

3. Replace the types in the meta proximity graph with the aligned types using `python match_type.py`.
