# RAIDAR: geneRative AI Detection viA Rewriting (ICLR 2024)

<p align="center">
  <p align="center" margin-bottom="0px">
    <a href="http://www.cs.columbia.edu/~mcz/"><strong>Chengzhi Mao*</strong></a>
    ·
    <a href="http://www.cs.columbia.edu/~vondrick/"><strong>Carl Vondrick</strong></a>
    ·
        <a href="http://www.wanghao.in"><strong>Hao Wang</strong></a>
        .
    <a href="http://www.cs.columbia.edu/~junfeng/"><strong>Junfeng Yang</strong></a>
    ·
    <p align="center" margin-top="0px"><a href="https://arxiv.org/abs/2401.12970">https://arxiv.org/abs/2401.12970</a></p>
</p>

We find that large language models (LLMs) are more likely to modify human-written text than AI-generated text when tasked with rewriting. This tendency arises because LLMs often perceive AI-generated text as high-quality, leading to fewer modifications. We introduce a method to detect AI-generated content by prompting LLMs to rewrite text and calculating the editing distance of the output. We dubbed our geneRative AI Detection viA Rewriting method Raidar.  Raidar  significantly improves the F1 detection scores of existing AI content detection models -- both academic and commercial -- across various domains, including News, creative writing, student essays, code, Yelp reviews, and arXiv papers, with gains of up to 29 points. Operating solely on word symbols without high-dimensional features, our method is compatible with black box LLMs, and is inherently robust on new content. Our results illustrate the unique imprint of machine-generated text through the lens of the machines themselves.



# Experiment 

## Yelp

Original data is: `yelp_huma.json`.

1. Generate Yelp GPT data: `main_fakeyelp_creator.py`, will obtain data `yelp_GPT_concise.json`.

Dataset: `yelp_huma.json`, `yelp_GPT_concise.json`

2. Our Detection algorithm: 

Step 1: Run LLM rewrite. `main_yelp_gpt_rewrite.py`, which will obtain `rewrite_yelp_human_inv.json` and `rewrite_yelp_GPT_inv.json`. 
If want to use llama, then run `main_yelp_llama_rewrite.py`

Step 2: Train a classifier/threshold on the edit distance features. `detect_yelp_inv.py`


### Other Variants
For equivariance, `main_yelp_gpt_equi_rewrite.py` for rewrite. Data saved in `equi_data`

For equivariance, Data saved in `uncertainty_data`

For detection on text from different models, see `data_A_rewrite_yelp_generated_from_B`

For evade detection, see `evade`

## Code

Dataset: `code_GPT-v2.json`, `code_human-v2.json`

## Arxiv


Dataset: `arxiv_GPT_concise.json`, `arXiv_human.json`


Note, the OpenAI key in the project is expired, you need to put in your own.
