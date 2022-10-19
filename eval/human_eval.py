import random, json
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
    
    dic_scores = defaultdict(int)

    # q_p1,a_p1,a_p2,par_1,par_2,answer_1,answer_2,score_a,score_par
    filled_file = pd.read_csv(input_file, encoding = 'utf-8')
     
    fact_q = ["who", "what", "where", "when", "why", "how"] 
    
    print(filled_file.shape) 
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
            dic_scores[filled_file.loc[qn, 'score_answer']] += 1
                
    print(filled_file['score_answer'].value_counts())   
    print(filled_file['score_paragraph'].value_counts()) 
    
    print(dic_scores)

    filled_file = filled_file[['q_p1', 'par_1', 'par_2', 'a_p1', 'a_p2', 'score_answer', 'score_paragraph']]
    
    filled_file.to_csv(output_file, index = False)

