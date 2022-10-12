# SleepQA: A Health Coaching Dataset on Sleep for Extractive Question Answering


We release SleepQA, a dataset created from 7,005 passages comprising 4,250 training examples with single annotations and 750 examples with 5-way annotations. We fine-tuned different domain-specific BERT models on our dataset and perform extensive automatic and human evaluation of the resulting end-to-end QA pipeline. Comparisons of our pipeline with baseline show improvements in domain-specific natural language processing on real-world questions.

This code is based on the following paper: Iva Bojic, Qi Chwen Ong, Megh Thakkar, Esha Kamran, Irving Yu Le Shua, Rei Ern Jaime Pang, Jessica Chen, Vaaruni Nayak, Shafiq Joty, Josip Car. **SleepQA: A Health Coaching Dataset on Sleep for Extractive Question Answering.**

# Dataset collection

We collected our dataset in three phases: 1) *passage curation*, 2) *passage-question-answer triplets*, and 3) *inter-annotator agreement annotations*. Additionally, we collected 4) *real-world questions*, which we use for extrinsic evaluation of our QA system. 


## Passage curation

We download more than 1,000 articles from two web pages[^1][^2] to obtain a high-quality, evidence-based, and medically reviewed sleep health information. Subsequently, we reorganize all passages and divide the content into passages with lengths of 100 to 150 words. Our pre-processing work produced 7,005 clean passages covering a wide range of topics related to sleep health.

## Passage-question-answer triplets

Questions and respective answers were generated manually for a total of 5,000 randomly chosen passages. For each label, the question formed must start with one of the following words: who, what, where, when, why or how (i.e., factoid question). The question ends with a question mark. A text span from the passage is selected as answer.

## Inter-annotator agreement annotations

We randomly sample 150 labels from each annotator and assign them to four other annotators. With a passage and question shown, they repeat the annotation process independently without knowing each other's input of answers. Each of them identifies answers for 600 questions formed by their counterparts, yielding 750 passages with one question and five answers. In this way, each of the questions in the set of 750 labels has a 5-way annotation.

## Real-world questions

In addition to collecting labels, we also collect 1,000 real-world questions related to sleep in natural language. Annotators were neither presented with the text corpus nor given any specific instruction to form a certain type of questions, such as factoid questions. By collecting real-world questions, in addition to performing intrinsic evaluation of our system using test labels (i.e., using passage-question-answer triplets), we are also able to do extrinsic evaluation.


# Dataset analysis

## Inter-annotator agreement

To calculate inter-annotator agreement, we treat the answers that are provided by the initial annotator as ground truth answers and keep the answers from four other annotators as human predictions. We take the maximum for both EM and F1 scores over all predictions (i.e., for four comparisons) and average them over all questions. The resulting score in this subset is **0.85** for the EM and **0.91** for F1 score. These numbers are above 0.8 ensuring reasonable quality of annotations[^4].

## Average number of words

We compared the general characteristics of our dataset (e.g., average number of words in passages, questions and answers) with six datasets [^3] containing extractive and short abstractive answers.

| Dataset      | Avg.             | \# of | words |     |Word| frequency | of    |1st    | question|(%) || 
| :----------: | :--------------: | :---: | :---: |:--: |:--:|:--:| :--: | :---: | :---: | :---: | :---: |
|              | P                | Q     | A     | Why |How |What| When | Where | Who   | Which | OTHER |
| **SleepQA**  | **120**          | **10**|**10** |**6**|**17**|**68**|**5**|**1**| **3**| **0** | **0** |
| SQuAD (2\.0) | 117              | 10    | 3     | 1   | 9  | 45 | 6    | 4     | 10    | 4     | 18    |
| MS MARCO v2  | 56               | 6     | 14    | 2   | 17 | 35 | 3    | 4     | 3     | 2     | 35    |
| TriviaQA     | 2895             | 14    | 2     | <1  | 4  | 33 | 2    | 2     | 17    | 42    | <1    |
| CoQA         | 271              | 6     | 3     | 2   | 5  | 27 | 2    | 5     | 15    | 1     | 43    |
| HotpotQA     | 917              | 18    | 2     | <1  | 3  | 37 | 3    | 2     | 14    | 29    | 13    |
| NarrativeQA  | 656              | 10    | 5     | 10  | 11 | 38 | 2    | 8     | 23    | 2     | 7     |

Average number of words per passages and questions in SleepQA dataset are compared to the one in SQuAD (2.0) dataset. However, the average number of words in answers is three times as much.

## Question and answer entailment

Question A entails question B if every answer to B is also exactly or partially correct answer to A. Similarly, answer A and answer B can be considered to be entailed if both are able to answer the same question. Since in our label collection process, we used different passages for formulating each question, it was expected that both question and answer entailment would be rather low due to different choices of words and the unlikeliness of identical phrasing appearing in different passages. However, there is a greater-than-expected occurrence of identical answers. This can be attributed to two factors: the large proportion of questions that call for numerical answers such as 7, as well as the fact that answers are text spans as opposed to full sentences, leading to less variation.

| Entailment type | Occurrence |
| :-------------: | :--------: |
| Question        |     222    |
| Answer          |     149    |

## Question answer similarities

In order to compare similarities between a given question and a given answer in a pair in our dataset against the other similar datasets, we downloaded five datasets using download script provided by authors. From each training set, we then randomly selected 1,000 question-answer pairs and for each question we detected the full sentence where the answer came from. In that sense, for each dataset separately we built a subset of labels where answers were full sentences, rather than text spans. Finally, for each pair in a particular dataset, we calculated F1 score separately and then averaged them over all 1,000 pairs. 

| Dataset name | F1 score |
| :----------: | :------: |
| **SleepQA**  | **0.17** |
|  SQUAD 1.1   |   0.09   |
| TriviaQA     |   0.07   |
|  CuratedTrec | 0.05     |
|  NQ          | 0.04     |
| WebQuestions | 0.02     |

Detected similarities between a question and an answer in a question-answer pair in our dataset were higher than those from other datasets. This could potentially be a result of labeling process during which annotators were encouraged to first find a potential answer from the passage and then formulate a question based on the chosen answer. This resulted in using similar phrases in the posed questions from the corresponding passages. Although, we do note that perhaps some of the overlap could be reduced by giving reminders to rephrase questions, this problem cannot be completely solved using just annotators’ efforts. In future work, we will investigate whether using back translation for data augmentation could solve this problem. The main idea behind using back translation for data augmentation is that the training examples are machine-translated from a source to a pivot language and back, thus obtaining paraphrases. 



[^1]: [Sleep foundation webpage](https://www.sleepfoundation.org)
[^2]: [The sleep doctor webpage](https://thesleepdoctor.com)
[^3]: [ELI5: Long form question answering](https://arxiv.org/abs/1907.09190)
[^4]: [Inter-coder agreement for computational linguistics](https://direct.mit.edu/coli/article/34/4/555/1999/Inter-Coder-Agreement-for-Computational)
[^5]: [Five different datasets](https://github.com/facebookresearch/DPR/blob/main/dpr/data/download_data.py)
