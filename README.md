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
