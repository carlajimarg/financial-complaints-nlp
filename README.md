# Financial complaint classification and response drafting

This project looks at how consumer financial complaints can be sorted by product from the text customers submit. I compare TF-IDF classifiers with a fine-tuned DeBERTa-v3-base model, then use the selected classifier to route complaints and prepare response drafts for human review.

The data comes from [liri-uzh/cfpb-complaints-mini](https://huggingface.co/datasets/liri-uzh/cfpb-complaints-mini), a subset of complaints from the [Consumer Financial Protection Bureau](https://www.consumerfinance.gov/data-research/consumer-complaints/). The task includes 7 categories: bank accounts, credit cards, credit reporting, debt collection, money transfers, mortgages and student loans.

## Classification

The dataset used has 6,469 training complaints and 1,635 test complaints. After a pretreatment, I use 5,400 complaints for training and 953 for validation.

I trained 4 TF-IDF configurations before fine-tuning DeBERTa-v3-base to see how results change. The models were selected using validation macro-F1.

| Model | Validation macro-F1 |
| --- | ---: |
| TF-IDF + logistic regression | 0.8360 |
| Larger word TF-IDF + logistic regression | 0.8400 |
| Word TF-IDF + linear SVM | 0.8465 |
| Character TF-IDF + linear SVM | 0.8380 |
| DeBERTa-v3-base | 0.8578 |

DeBERTa was the highest-scoring model on validation. On the held-out test set, it reached **0.8517 macro-F1** and **0.852 accuracy**. It performed best on student loans (0.96 F1) and mortgages (0.94 F1). Bank accounts (0.76 F1) and money transfers (0.79 F1) were harder to distinguish.

## Review and response drafting

I use the maximum softmax score to decide which category predictions should be checked manually. The threshold is set at the 20th percentile of validation scores (0.989). On the test set, 79.4% of complaints passed this check, with 92.3% routing accuracy among those cases.

For complaints that pass the check, the notebook:

1. Filters a set of 49 CFPB-linked guidance summaries to the predicted product category.
2. Uses `all-MiniLM-L6-v2` embeddings and cosine similarity to retrieve two relevant summaries.
3. Sends the complaint and those summaries to `Phi-3.5-mini-instruct` to draft a response.

## Running the notebook

This notebook was created in Google Colab using GPU to run the results.

## Limitations

The dataset is relatively small and more balanced than real complaints. DeBERTa truncates narratives at 384 tokens, which may leave out details in long complaints. The set of guidance summaries is also small. 

This is a notebook experiment, not a system ready to handle customer complaints without supervision.
