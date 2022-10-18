from indexes import prepare_json, create_indexes
from auto_eval import *


if __name__ == "__main__":
    
    text_corpus = "../data/training/sleep-corpus.tsv"
    # needs absolute path!
    json_folder = "/4TB/guest1/github/SleepQA/data/bm25_json/"
    # needs absolute path!
    index_folder = "/4TB/guest1/github/SleepQA/data/bm25_indexes/"
    
    sleep_test = "../data/training/sleep-test.csv"
    oracle_json = "../data/oracle_json/sleep-test.json"
    
    retrieval_folder = "../data/processed/retrieval/"
    reader_oracle_folder = "../data/processed/reader/oracle/"

    pipeline1_file = "../data/processed/reader/pipeline1_label_1.250.json"
  
    # create json
    #prepare_json(text_corpus, json_folder)
    
    # create sparse indexes
    #create_indexes(json_folder, index_folder)
    
    # calculate recall@1 for different bert retrieval models
    berts_top1(retrieval_folder)
    
    # retrive k (100) most relevant passagaes for each question
    lucene_topk(sleep_test, index_folder, 100)
    
    # calculate EM/F1 for different bert readers on oracle paragraphs
    berts_em(reader_oracle_folder)
    
    # run bert-uncased-squad2 reader on oracle paragraphs
    oracle_squad2(oracle_json)
    
    # calculate EM/F1 for the best pipeline on test labels
    pipeline1(pipeline1_file)
    
    # run BM25 + bert-uncase-squad2 pipeline on test labels
    pipeline2(sleep_test, index_folder, False)
    

    
    