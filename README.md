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

We compared the general characteristics of our dataset (e.g. average number of words in passages, questions and answers) with six datasets [^3] containing extractive and short abstractive answers.

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

[^1]: [Sleep foundation webpage](https://www.sleepfoundation.org)
[^2]: [The sleep doctor webpage](https://thesleepdoctor.com)
[^3]: [ELI5: Long form question answering](https://arxiv.org/abs/1907.09190)
[^4]: [Inter-coder agreement for computational linguistics](https://direct.mit.edu/coli/article/34/4/555/1999/Inter-Coder-Agreement-for-Computational)
