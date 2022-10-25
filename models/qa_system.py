import sys 
sys.path.append("../utils")
sys.path.append("../DPR-main/")

from f1_score import calculate_f1

import csv, pickle, torch, time
from transformers import (
    DPRContextEncoder, DPRContextEncoderTokenizer, 
    DPRQuestionEncoder, DPRQuestionEncoderTokenizer, 
    DPRReader, DPRReaderTokenizer
    )

from dpr.indexer.faiss_indexers import DenseFlatIndexer


def iterate_encoded_files(file):
    
    print("Reading file {}".format(file))
    with open(file, "rb") as reader:
        doc_vectors = pickle.load(reader)
        for doc in doc_vectors:
            doc = list(doc)                
            yield doc


class LocalFaissRetriever():

    def __init__(self): 
        self.index = DenseFlatIndexer()
        self.index.init_index(768)
        
    def index_encoded_data(self, file, buffer_size):
        buffer = []
        for item in iterate_encoded_files(file):
            buffer.append(item)
            if 0 < buffer_size == len(buffer):
                self.index.index_data(buffer)
                buffer = []
        self.index.index_data(buffer)
        print("Data indexing completed.")

    def get_top_docs(self, query_vectors, top_docs):
        time0 = time.time()
        results = self.index.search_knn(query_vectors, top_docs)
        print("index search time: {} sec.".format(time.time() - time0))
        self.index = None
        return results


def validate_retriever(questions, retrieved_passages):
    
    top = 0
    qa_dic = {}
      
    # read questions
    with open(questions, "r", encoding = "utf-8") as fin:
         reader = csv.reader(fin, delimiter = "\t")
         for row in reader:
             question = row[0]
             answer = row[1].strip('"["').strip('"]"')
             qa_dic[question] = answer
          
    # read in passages from our pipeline    
    with open(retrieved_passages, "r", encoding = "utf-8") as fin:
       reader = csv.reader(fin, delimiter = "\t")
       for row in reader:
           question = row[0]
           text = row[2]        

           # check if the retrieved paragraph from our pipeline is contains answer
           if(text.find(qa_dic[question]) != -1):
                top += 1
              
    print("Top k documents hits: {}".format(top))


def validate_reader(questions, span_answers):
    
    f1, em = [], []
    qa_dic = {}
    
    with open(questions, "r", encoding = "utf-8") as fin:
         reader = csv.reader(fin, delimiter = "\t")
         for row in reader:
             question = row[0]
             answer = row[1].strip('"["').strip('"]"')
             qa_dic[question] = answer.replace(".", "")
             
    with open(span_answers, "r", encoding = "utf-8") as fin:
       reader = csv.reader(fin, delimiter = "\t")
       for row in reader:
           question = row[0]
           
           answer = row[1].strip().replace(".", "").replace(" %", "%")
           
           macro_f1 = calculate_f1(qa_dic[question], answer)   
           f1.append(macro_f1)
           
           if(macro_f1 == 1):
               em.append(1)
           else:
               em.append(0)

 
    print("em: {:.2f}, f1: {:.2f}".format(sum(em)/len(em), sum(f1)/len(f1)))
    
    
############################################## CONTEXT #############################################################
# facebook/dpr-ctx_encoder-single-nq-base
def generate_dense_encodings(text_corpus, ctx_encoder, out_file):
    
    total = 0
    results = []
    
    tokenizer = DPRContextEncoderTokenizer.from_pretrained(ctx_encoder)
    encoder = DPRContextEncoder.from_pretrained(ctx_encoder)

    with open(text_corpus, encoding = "utf-8") as fin:
         reader = csv.reader(fin, delimiter="\t")
         for row in reader:
             sample_id = str(row[0])
             passage = row[1].strip('"')    
             title = row[2]
             
             tokens = tokenizer(title + "[SEP]" + passage, return_tensors="pt", max_length = 256, 
                                  padding='max_length', truncation = True)["input_ids"]
                    
             tokens[0][255] = 102            # add 102 in the end of padding  
             embeddings = encoder(tokens).pooler_output       
             results.extend([(sample_id, embeddings[0,:].detach().numpy())])
             
             total += 1
             if(total % 10 == 0):
                 print("Encoded {} passages.".format(total))                 

    with open(out_file, mode = "wb") as f:
        pickle.dump(results, f)

    print("Total passages processed {}. Written to {}".format(len(results), out_file))


############################################## QUESTION #############################################################
# facebook/dpr-question_encoder-single-nq-base
def dense_retriever(questions, question_encoder, text_corpus, corpus_embedded, retrieved_passages):
    
    index_buffer_sz = 50000  
    par_dic = {}
    
    with open(text_corpus, "r", encoding = "utf-8") as fin:
         reader = csv.reader(fin, delimiter="\t")
         for row in reader:
             sample_id = str(row[0])
             passage = row[1].strip('"')    
             title = row[2]
             
             par_dic[sample_id] = (title, passage)
    
    tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(question_encoder)
    model = DPRQuestionEncoder.from_pretrained(question_encoder)   

    questions_embedded = []
    questions_list = []
    
    total = 0
    with open(retrieved_passages, "w", encoding = "utf-8") as fout:
        with open(questions, encoding = "utf-8") as fin:
             reader = csv.reader(fin, delimiter="\t")
             for row in reader:
                 question = row[0].strip('"')
                 questions_list.append(question)
                 
                 tokens = tokenizer(question, return_tensors="pt", max_length = 256, 
                                      padding='max_length', truncation = True)["input_ids"]          
                 tokens[0][255] = 102            # add 102 in the end of padding          
                 embeddings = model(tokens).pooler_output
                 
                 questions_embedded.append(embeddings[0,:])                
                     
                 total += 1
                 if(total % 10 == 0):
                     print("Encoded {} questions.".format(total))                 
                                  
                 if(total % 100 == 0):
    
                     retriever = LocalFaissRetriever()
                     retriever.index_encoded_data(corpus_embedded, index_buffer_sz)
                 
                     questions_embedded = torch.stack(questions_embedded)     
                     top_results_and_scores = retriever.get_top_docs(questions_embedded.detach().numpy(), 1)
                                                
                     for i in range(len(top_results_and_scores)):
                        par_id = top_results_and_scores[i][0][0]
                            
                        results = par_dic[par_id]
                        title = results[0]
                        passage = results[1]
                            
                        fout.write("{}\t{}\t{}\n".format(questions_list[i], title, passage))
                        
                     questions_embedded, questions_list = [], []
                 

# ############################################### READER ##############################################################
# facebook/dpr-reader-single-nq-base
def extractive_reader(retrieved_passages, reader, span_answers):
    
    tokenizer = DPRReaderTokenizer.from_pretrained(reader)
    model = DPRReader.from_pretrained(reader)
    
    total = 0
    with open(span_answers, "w", encoding = "utf-8") as fout:
        with open(retrieved_passages, "r", encoding = "utf-8") as fin:
             reader = csv.reader(fin, delimiter="\t")
                  
             for row in reader:
                 question = row[0].strip('"')
                 title = row[1].strip('"')
                 text = row[2].strip('"')
                 encoded_inputs = tokenizer(question, title, text, return_tensors = "pt", 
                        max_length = 300, padding = 'max_length', truncation = True)

                 outputs = model(**encoded_inputs)
                 predicted_spans = tokenizer.decode_best_spans(encoded_inputs, outputs, max_answer_length = 20, num_spans = 1, num_spans_per_passage = 1)
             
                 fout.write("{}\t{}\n".format(question, predicted_spans[0].text))

                 total += 1
                 if(total % 10 == 0):
                     print("Extracted spans for {} questions.".format(total)) 
             

if __name__ == "__main__":

    text_corpus = "../data/training/sleep-corpus.tsv"
    questions = "../data/training/sleep-test.csv"
    
    ctx_encoder = "pytorch/ctx_encoder/"
    question_encoder = "pytorch/question_encoder/"
    reader = "pytorch/reader/"
    
    corpus_embedded = "processed/sleep-corpus_e29"
    retrieved_passages = "processed/sleep_test_e29.csv"
    span_answers = "processed/pipeline1_label_1.250.csv"
    
    #generate_dense_encodings(text_corpus, ctx_encoder, corpus_embedded)
    
    #dense_retriever(questions, question_encoder, text_corpus, corpus_embedded, retrieved_passages)

    #validate_retriever(questions, retrieved_passages)
 
    extractive_reader(retrieved_passages, reader, span_answers)
    
    validate_reader(questions, span_answers)
