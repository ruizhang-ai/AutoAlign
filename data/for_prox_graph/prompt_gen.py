
with open(f"typeset1.txt", "r") as f:
    typeset1 = f.read().splitlines()
with open(f"typeset2.txt", "r") as f:
    typeset2 = f.read().splitlines()


promt_template = f"Now you are an expert in linguistics and knowledge graphs. I will give you two sets of words, indicating the entity types from two knwoledge graphs. You need to identify all the word pairs from the two sets tat are synonyms. For example, if the first set has the word ‘people’ and the second set has the word 'person', you need to identify the two words being synonyms and return me the pair (people, person). Now the following are the two sets: Set 1: {typeset1} Set 2: {typeset2} Please return all the pairs that are synonyms from the two sets reagarding entity types. Do not output the pairs if they are exactly the same. Remember you only need to return the pairs, each pair in one line. Each pair contain two types, one from Set 1 and another from Set 2, in the format (type1, type2)."

with open(f"prompt.txt", "w") as f:
    f.write(promt_template)

print(f"Please use the prompt in prompt.txt to generate the Aligned data")

