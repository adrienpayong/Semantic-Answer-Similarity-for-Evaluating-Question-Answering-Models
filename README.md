# Review of the paper Semantic Answer Similarity for Evaluating Question Answering Models
## Introduction
The evaluation of question answering (QA) models relies on human-annotated datasets of question- answer pairs. Given a question, the ground-truth answer is compared to the answer predicted by a model with regard to different similarity met- rics. Currently, the most prominent metrics for the evaluation of QA models are exact match (EM), F1-score, and top-n-accuracy. All these three met- rics rely on string-based comparison. Even a prediction that differs from the ground truth in only one character in the answer string is evaluated as completely wrong. 

To mitigate this problem and have a continuous score ranging between 0 and 1, the F1-score can be used. In this case, precision is calculated based on the relative number of tokens in the predicti that are also in the ground-truth answer and recall is calculated based on the relative number of tokens in the ground-truth answer that are also in the prediction. As an F1-score is not as simple to interpret as accuracy, there is a third common metric for the evaluation of QA models. Top-n-accuracy evaluates the first n model predictions as a group and considers the predictions correct if there is any positional overlap between the ground-truth answer and one of the first n model predictions — otherwise, they are considered incorrect.
## Issue and Solution
If a dataset contains multi-way annotations, there can be multiple different ground-truth answers for the same question. The maximum similarity score
of a prediction over all ground-truth answers is used in that case, which works with all the metrics above. However, a problem is that sometimes only one correct answer is annotated when in fact there are multiple correct answers in a document. 

If the multiple correct answers are semantically but not lexically the same, existing metrics require all correct answers within a document to be annotated and cannot be used reliably otherwise. Figure 1 gives an
example of a context, a question, multiple groundtruth answers, a prediction and different similarity scores. the existing metrics cannot capture the seman-
tic similarity of the prediction and the ground-truth answers but are limited to lexical similarity. Given the shortcomings of the existing metrics,
a novel metric for QA is needed and we address this challenge by presenting SAS, a cross-encoder- based semantic answer similarity metric.
## Approach 
The researchers consider four different approaches to estimate the semantic similarity of pairs of answers: 
- bi-encoder approach
- cross-encoder approach
- vanilla version of BERTScore
- trained version of BERTScore

**Bi-Encoder**: The bi-encoder approach is based on the sentence transformers architecture (Reimers and Gurevych, 2019), which is a siamese neural network architecture comprising two language models that encode the two text inputs and cosine similarity to calculate a similarity score of the two encoded texts. The model that use is based on xlm-roberta-base, where the training has been continued on an unreleased multi-lingual paraphrase dataset. The resulting model, called paraphrase-xlm-r-multilingual-v1, has then been fine-tuned on the English-language STS benchmark
dataset (Cer et al., 2017) and a machine-translated German-language version3 of the same data. The final model is called T-Systems-onsite/cross-en-de-
roberta-sentence-transformer and is available on the huggingface model hub.

**SAS**: Our new approach called SAS differs from the bi-encoder in that it does not calculate separate embeddings for the input texts. Instead, we use a cross-encoder architecture, where the two texts are concatenated with a special separator token in between. The underlying language model is called cross-encoder/stsb-roberta-large and has been trained on the STS benchmark dataset (Ceret al., 2017).

**BERTScore vanilla or trained**: The BERTScore vanilla approach uses the task-agnostic, pre-trained language models bert-base-uncased for the English-
language datasets and deepset/gelectra-base for the German-language dataset. In line with the approach by Zhang et al. (2020), the researchers use the language models to generate contextual embeddings, match the embeddings of the tokens in the ground- truth answer and in the prediction and take the maximum cosine similarity of the matched tokens as the similarity score of the two answers. 

## Experiments
To evaluate the ability of the different approaches to estimate semantic answer similarity, the researchers measure their correlation with human judgment of similarity on three datasets. 
### Datasets
The evaluation uses subsets of three existing datasets: SQuAD, GermanQuAD, and NQ-open. We process and hand-annotate the datasets as de- scribed in the following so that each of the processed subsets contains pairs of answers and a class label that indicates their semantic similarity. are three similarity classes: dissimilar answers, approximately similar answers, and equivalent answers, which are all described in Table1.

**SQuAD**: the researchers annotate the semantic similarity of pairs of answers in a subset of the English-language SQuAD test dataset (Rajpurkar et al., 2018). They consider a subset where 566 pairs of ground-truth answers have an F1-score of 0 (no lexical overlap of the answers) and 376 pairs have an F1-score larger than 0 (some lexical overlap of the answers). The resulting dataset comprises 942 pairs of answers each with a majority vote indicating either dissimilar answers, approximately similar answers, or equivalent answers.

**GermanQuAD**: To show that the presented approaches also work on non-English datasets, they consider the German-language GermanQuAD dataset (Möller et al., 2021) which contains a three-way annotated test set, which means there are three correct answers given for each question.
**NQ-open**: The original Natural Questions dataset (NQ) (Kwiatkowski et al., 2019) was meant for reading comprehension but Lee et al. (2019) adapted the dataset for open-domain QA and it has been released under the name NQ-open. We use the test dataset of NQ-open as it contains not only questions and ground-truth answers but also model predictions and annotations how similar these predictions are to the ground-truth answer. 
## Results
- Table 2 lists the correlation between different au tomated evaluation metrics and human judgment using Spearman’s rho and Kendall’s tau-b rank correlation coefficients on labeled subsets of SQuAD, GermanQuAD, and NQ-open datasets.
- The traditional metrics ROUGE-L and METEOR have very weak correlation with human judgement if there is no lexical overlap between the pair of an-
swers, in which case the F1-score and BLEU are 0.
- If there is some lexical overlap, the correlation is stronger for all these metrics but BLEU lags far behind the others. METEOR is outperformed by ROUGE-L and F1-score, which achieve almost equal correlation. 
- METEOR is outperformed by ROUGE-L and F1-score, which achieve almost equal correlation.
- All four semantic answer similarity approaches outperform the traditional metrics and among them, the cross-encoder model is consistently achieving the strongest correlation with human judgment except for slightly underperforming the trained BERTScore metric with regard to τ on English-language pairs of answers with no lexical overlap.
- This result shows that semantic similarity metrics are needed in addition to lexical-based metrics for automated evaluation of QA models. The former correlate much better with human judgment and thus, are a better estimation of a model’s performance in real-world applications.

**Embedding Extraction for BERTScore**:
- BERT-Score can be used with different language models to generate contextual embeddings of text inputs. While the embeddings are typically extracted from the last layer of the model, they can be extracted from any of its layers and related work has shown that for some tasks the last layer is not the best (Liu
et al., 2019). 
