import random, json, pprint
import numpy as np
import pandas as pd
from collections import defaultdict
from nltk.tokenize import word_tokenize


def json_csv(input_file, output_file):
    
    with open(output_file, "w", encoding = "utf-8") as fout:
        with open(input_file, "r", encoding = "utf-8") as fin: 
            objs = json.load(fin)
            for obj in objs:
                q = obj['question']
                a = obj['predictions'][0]['prediction']['text']
                p = obj['predictions'][0]['prediction']['passage']
                
                fout.write("{}\t{}\t{}\n". format(q, p, a))


def randomize_answers(pipeline1, pipeline2, output_file):
    
    pipeline1 = pd.read_csv(pipeline1, delimiter = "\t", 
                              names = ['q_p1', 'p_p1', 'a_p1'], header = None, encoding = 'utf-8')
    
    pipeline2 = pd.read_csv(pipeline2, delimiter = "\t", 
                                  names = ['q_p2', 'p_p2', 'a_p2'], header = None, encoding = 'utf-8')
     
    df_merged = pd.merge(pipeline1, pipeline2, left_index = True, right_index = True)
       
    for qn in df_merged.index:
        row = df_merged.iloc[qn]   
        
        if(row['q_p1'] != row['q_p2']):
            print("Questions are not the same!")
            
        choice = random.choice([True, False])
        
        if(choice == True):
            df_merged.loc[qn, 'answer_1'] = row['a_p1']
            df_merged.loc[qn, 'answer_2'] = row['a_p2']
            df_merged.loc[qn, 'par_1'] = row['p_p1']
            df_merged.loc[qn, 'par_2'] = row['p_p2']
        else:
            df_merged.loc[qn, 'answer_1'] = row['a_p2']
            df_merged.loc[qn, 'answer_2'] = row['a_p1']   
            df_merged.loc[qn, 'par_1'] = row['p_p2']
            df_merged.loc[qn, 'par_2'] = row['p_p1']
            
            
    df_merged = df_merged[['q_p1', 'a_p1', 'a_p2', 'par_1', 'par_2', 'answer_1', 'answer_2']]
    df_merged.to_csv(output_file, index = False)
    
    
def untangle_answers(input_file, output_file):
    
    scores_span, scores_par = defaultdict(int), defaultdict(int)

    # q_p1,a_p1,a_p2,par_1,par_2,answer_1,answer_2,score_a,score_par
    filled_file = pd.read_csv(input_file, encoding = 'utf-8')
     
    fact_q = ["who", "what", "where", "when", "why", "how"] 
    
    for qn in filled_file.index:
        row = filled_file.iloc[qn]  
        
        if(row['a_p1'] != row['answer_1']):
            if(row['score_a'] == 1):
                filled_file.loc[qn, 'score_answer'] = 2
            elif(row['score_a'] == 2):
                 filled_file.loc[qn, 'score_answer'] = 1
            else:
                filled_file.loc[qn, 'score_answer'] = row['score_a']
                
            if(row['score_par'] == 1):
                filled_file.loc[qn, 'score_paragraph'] = 2
            elif(row['score_par'] == 2):
                 filled_file.loc[qn, 'score_paragraph'] = 1
            else:
                filled_file.loc[qn, 'score_paragraph'] = row['score_par']
        else:
            filled_file.loc[qn, 'score_answer'] = row['score_a']
            filled_file.loc[qn, 'score_paragraph'] = row['score_par']
            
        words = word_tokenize(row['q_p1'])
            
        if(words[0] not in fact_q):
            scores_span[filled_file.loc[qn, 'score_answer']] += 1
            scores_par[filled_file.loc[qn, 'score_paragraph']] += 1
                
    print("Human evaluation scores for span answers (all): \n{}".format(filled_file['score_answer'].value_counts()))
    print("Human evaluation scores for span answers + explanations (all): \n{}".format(filled_file['score_paragraph'].value_counts())) 
    
    print("Human evaluation scores for span answers (not factual):")
    print("\n".join("{}\t{}".format(k, v) for k, v in scores_span.items()))
    
    print("Human evaluation scores for span answers + explanations (not factual):")
    print("\n".join("{}\t{}".format(k, v) for k, v in scores_par.items()))

    filled_file = filled_file[['q_p1', 'par_1', 'par_2', 'a_p1', 'a_p2', 'score_answer', 'score_paragraph']]
    
    filled_file.to_csv(output_file, index = False)


def convert(int_number):
    
    bin_number = np.zeros(4, int)
    
    bin_number[int_number - 1] = 1
    
    return bin_number


def get_ac1(preds_mat):
    if preds_mat.shape[0] == 1:
        return 0
    pi_q = np.sum(preds_mat, axis = 1) / preds_mat.shape[1]
    pey = (1 / (preds_mat.shape[0] - 1)) * np.sum(pi_q * (1 - pi_q))
    pa = (np.sum(preds_mat, axis = 1) * (np.sum(preds_mat, axis = 1)-1)) / (preds_mat.shape[1] * (preds_mat.shape[1] - 1))
    pa = np.sum(pa)
    ac1 = (pa - pey) / (1 - pey)
    ac1 *= 100
    return ac1


def calculate_gwet_AC1(input_file):
    
    agreement_file = pd.read_csv(input_file, encoding = 'utf-8')
    
    agreement_spans = agreement_file[['score_a_1', 'score_a_2', 'score_a_3', 'score_a_4', 'score_a_5']]
    agreement_exp = agreement_file[['score_p_1', 'score_p_2', 'score_p_3', 'score_p_4', 'score_p_5']]
    
    results = []
    for qn in agreement_spans.index:
        row = agreement_spans.iloc[qn]  

        preds_mat = np.stack(list(map(lambda n: convert(n), row)),  axis = -1)
        
        results.append(get_ac1(preds_mat))
        
    print("Gwet's AC1 score for span answers is: {:.2f}".format(sum(results)/len(results)))
    
    results = []
    for qn in agreement_exp.index:
        row = agreement_exp.iloc[qn]  

        preds_mat = np.stack(list(map(lambda n: convert(n), row)),  axis = -1)
        
        results.append(get_ac1(preds_mat))
        
    print("Gwet's AC1 score for span answers + explanations is: {:.2f}".format(sum(results)/len(results)))
            
            
      

        