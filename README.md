# AutoAlign: Fully Automatic and Effective Knowledge Graph Alignment enabled by Large Language Models


This is our implementation for the [paper](https://arxiv.org/pdf/2307.11772.pdf):

AutoAlign: Fully Automatic and Effective Knowledge Graph Alignment enabled by Large Language Models


## Description

The task of entity alignment between knowledge graphs (KGs) aims to identify every pair of entities from two different KGs
that represent the same entity. Many machine learning-based methods have been proposed for this task. However, to our best
knowledge, existing methods all require manually crafted seed alignments, which are expensive to obtain. In this paper, we propose
the first fully automatic alignment method named AutoAlign, which does not require any manually crafted seed alignments. Specifically,
for predicate embeddings, AutoAlign constructs a predicate-proximity-graph with the help of large language models to automatically
capture the similarity between predicates across two KGs. For entity embeddings, AutoAlign first computes the entity embeddings of
each KG independently using TransE, and then shifts the two KGs’ entity embeddings into the same vector space by computing the
similarity between entities based on their attributes. Thus, both predicate alignment and entity alignment can be done without manually
crafted seed alignments. AutoAlign is not only fully automatic, but also highly effective. Experiments using real-world KGs show that
AutoAlign improves the performance of entity alignment significantly compared to state-of-the-art methods.


<p align="center">
  <img src="/img/overall-framework.pdf", alt="Model Structure" width="800">
  <p align="center"><em>Overview of our proposed AutoAlign method for entity alignment.</em></p>
</p>

## What are in this Repository
This repository contains the following main contents:

```
/
├── code/                         
|   |   ├── AutoAlign.py      --> (The main code of AutoAlign)
├── data/                   
|   ├── DW-NB/                --> (The DW-NB dataset)
|   ├── DY-NB/                --> (The DY-NB dataset)
|   ├── for_prox-graph/       --> (Necessary code for generating proximity graphs)
├── img/                      --> (The images for README (not used for the code))   
```

## Run our code

```
python AutoAlign.py
```

## Cite our paper

Please credit our work by citing the following paper:

```
@article{zhang2023autoalign,
  title={AutoAlign: Fully Automatic and Effective Knowledge Graph Alignment enabled by Large Language Models},
  author={Zhang, Rui and Su, Yixin and Trisedya, Bayu Distiawan and Zhao, Xiaoyan and Yang, Min and Cheng, Hong and Qi, Jianzhong},
  journal={arXiv preprint arXiv:2307.11772},
  year={2023}
}
```
