from indexes import prepare_json, create_indexes
from auto_eval import *
from human_eval import *


if __name__ == "__main__":
    
    text_corpus = "../data/training/sleep-corpus.tsv"
    # needs absolute path!
    json_folder = "/4TB/guest1/github/SleepQA/data/bm25_json/"
    # needs absolute path!
    index_folder = "/4TB/guest1/github/SleepQA/data/bm25_indexes/"
    
    sleep_test = "../data/training/sleep-test.csv"
    oracle_json = "../data/training/oracle/sleep-test.json"
    sleep_open = "../data/training/open_questions.csv"
    
    retrieval_folder = "../data/processed/retrieval/"
    reader_oracle_folder = "../data/processed/reader/oracle/"

    pipeline1_label = "../data/processed/pipeline1_label_1.250.json"
    pipeline1_open_j = "../data/processed/pipeline1_open_1.250.json"
    pipeline1_open_c = "../data/processed/pipeline1_open.csv" 
    pipeline2_file = "../data/processed/pipeline2_open.csv"
    compare_file = "../data/processed/p1_p2_compare.csv"
    score_file = "../data/processed/p1_p2_compare_scored.csv"
    human_eval = "../data/processed/p1_p2_final.csv"
    
    consensus_file = "../data/agreement/model_agreement.csv"
  
    # create json
    prepare_json(text_corpus, json_folder)
    
    # create sparse indexes
    create_indexes(json_folder, index_folder)
    
    # calculate recall@1 for different bert retrieval models
    berts_top1(retrieval_folder)
    
    # retrive k (100) most relevant passagaes for each question
    lucene_topk(sleep_test, index_folder, 100)
    
    # calculate EM/F1 for different bert readers on oracle paragraphs
    berts_em(reader_oracle_folder)
    
    # run bert-uncased-squad2 reader on oracle paragraphs
    oracle_squad2(oracle_json)
    
    # calculate EM/F1 for the best pipeline on test labels
    pipeline1(pipeline1_label)
    
    # run BM25 + bert-uncase-squad2 pipeline on test labels
    pipeline2(sleep_test, index_folder, False, None)
    
    # transcribe json to csv for pipeline 1
    json_csv(pipeline1_open_j, pipeline1_open_c)    
    
    # run BM25 + bert-uncase-squad2 pipeline on open-ended questions
    pipeline2(sleep_open, index_folder, True, pipeline2_file)    
    
    # randomizes answers from two pipelines and prepare for human evaluation
    ##randomize_answers(pipeline1_open_c, pipeline2_file, compare_file)
    
    #untangle answers from two pipelines
    untangle_answers(score_file, human_eval)

    # calculates Gwet_AC1 for consensus
    calculate_gwet_AC1(consensus_file)

    
    