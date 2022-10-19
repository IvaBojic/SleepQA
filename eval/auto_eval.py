import sys 
sys.path.append("../utils")

import csv, time, json, os
from pyserini.search.lucene import LuceneSearcher
from transformers import pipeline, BertForQuestionAnswering, AutoTokenizer
       
from f1_score import calculate_f1
     

def berts_top1(input_folder):

    for file in os.listdir(input_folder):      
        top1_res = 0
        
        with open (input_folder + file, "r", encoding = "utf-8") as fin:
            objs = json.load(fin)
            for obj in objs:
                flag = obj['ctxs'][0]['has_answer']
                if(flag):
                    top1_res += 1
                    
        print("recall@1 for {}: {:.2f}".format(file.split("_")[0], top1_res/len(objs)))     
        
       
def read_file(file_name):
    
    q_a = []
    
    with open(file_name, "r", encoding = "utf-8") as fin:
        for line in csv.reader(fin, delimiter = "\t"):
            
            q = line[0]
            a = line[1].replace('["', "").replace('"]', "").replace('"', '') 
            
            q_a.append([q, a])

    return q_a


def lucene_topk(input_file, index_file, top_k):

    q_a = read_file(input_file)
    
    topk_res = [0] * top_k
    start_time = time.time()
    
    # initialize sparce searcher
    simple_searcher = LuceneSearcher(index_file)
    
    run_time = int(time.time() - start_time)
    print("Initialization: {} seconds.".format(run_time))
    start_time = time.time()
    
    for q, a in q_a:

        hits = simple_searcher.search(q, top_k) 
        
        for i, hit in enumerate(hits):
            doc_id = hit.docid
            context = json.loads(simple_searcher.doc(doc_id).raw())['contents']              
            
            if(a in context):
                for j in range(i, top_k):
                    topk_res[j] += 1
                break
    
    for j in range(top_k):
        topk_res[j] /= len(q_a)
        
    print("recall@1: {}".format(topk_res[0]))  
    print("recall@k: {}".format(topk_res))          
    
    run_time = int(time.time() - start_time)
    print("Answering time: {} seconds.".format(run_time))
    

def read_json(file_name):
    
    f1_go, em_go = [], []

    with open(file_name, "r", encoding = "utf-8") as fin: 
        objs = json.load(fin)
        for obj in objs:
            gold = obj['gold_answers'][0]
            answer = obj['predictions'][0]['prediction']['text']
    
            macro_f1 = calculate_f1(gold, answer)   
            f1_go.append(macro_f1)
           
            if(macro_f1 == 1):
                em_go.append(1)
            else:
                em_go.append(0)
                
    return f1_go, em_go


def berts_em(input_folder):
    
    for file in os.listdir(input_folder):      

        f1_go, em_go = read_json(input_folder + file)
               
        print("{} --> em: {:.2f}, f1: {:.2f}".format(file.split("_")[0], 
                                                     sum(em_go)/len(em_go), sum(f1_go)/len(f1_go)))

    
def oracle_squad2(oracle_json):
  
    start_time = time.time()
    
    model_name = 'deepset/bert-base-uncased-squad2'
    model = BertForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)    
    nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)
    
    run_time = int(time.time() - start_time)
    print("Initialization: {} seconds.".format(run_time))
    start_time = time.time()
    
    f1_go, em_go = [], []
    
    with open(oracle_json, "r", encoding = "utf-8") as fin: 
        objs = json.load(fin)
        for obj in objs:
            q = obj['question']
            answer = obj['answers'][0]
            text = obj['ctxs'][0]['text']
            
            if(answer not in text):
                print(answer)
                continue
                    
            answer_object = nlp({'question': q, 'context': text})['answer']
            
            macro_f1 = calculate_f1(answer_object, answer)   
            f1_go.append(macro_f1)
           
            if(macro_f1 == 1):
                em_go.append(1)
            else:
                em_go.append(0)
                
    print("em: {:.2f}, f1: {:.2f}".format(sum(em_go)/len(em_go), sum(f1_go)/len(f1_go)))
    
    run_time = int(time.time() - start_time)
    print("Answering time: {} seconds.".format(run_time))
  
      
def pipeline1(input_file):
    
    f1_go, em_go = read_json(input_file)
           
    print("{} --> em: {:.2f}, f1: {:.2f}".format(input_file.split("_")[0].split("/")[-1:][0], 
                                                  sum(em_go)/len(em_go), sum(f1_go)/len(f1_go)))    
 
       
def pipeline2(input_file, index_file, flag_open, output_file):
    
    q_a = read_file(input_file)
    
    start_time = time.time()
    
    simple_searcher = LuceneSearcher(index_file)

    model_name = 'deepset/bert-base-uncased-squad2'
    model = BertForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)    
    nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)
    
    run_time = int(time.time() - start_time)
    print("Initialization: {} seconds.".format(run_time))
    start_time = time.time()
    
    f1_go, em_go = [], []
    q_a_p = {}
    
    for q, a in q_a:

        hits = simple_searcher.search(q)            
        doc_id = hits[0].docid
           
        context = json.loads(simple_searcher.doc(doc_id).raw())['contents']

        answer_object = nlp({'question': q, 'context': context})['answer']

        if (not flag_open):
            macro_f1 = calculate_f1(answer_object, a)   
            f1_go.append(macro_f1)
        
            if(macro_f1 == 1):
                em_go.append(1)
            else:
                em_go.append(0)
        else:
            q_a_p[q] = (context, answer_object)
    
    if (not flag_open):
        print("pipeline2 --> em: {:.2f}, f1: {:.2f}".format(sum(em_go)/len(em_go), sum(f1_go)/len(f1_go)))
    else:
        with open(output_file, "w", encoding = "utf-8") as fout: 
            for q in q_a_p.keys():
                fout.write("{}\t{}\t{}\n". format(q, q_a_p[q][0], q_a_p[q][1]))

      
    run_time = int(time.time() - start_time)
    print("Answering time: {} seconds.".format(run_time))        
       
    

    
    
    
    
    
    
    