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
The researchers consider four different approaches to estimate the semantic similarity of pairs of answers: a biencoder approach, a cross-encoder approach, a vanilla version of BERTScore, and a trained ver-sion of BERTScore. 
