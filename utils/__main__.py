from dataset_analysis import avg_no_words, calculate_entailment, calculate_qa_sim
from inter_agreement import labels_agreement


if __name__ == "__main__":

    agreement_file = "../data/agreement/labels_agreement.csv"
    
    file_list = ["../data/training/sleep-train.csv", 
                "../data/training/sleep-dev.csv", 
                "../data/training/sleep-test.csv"]
    
    text_corpus = "../data/training/sleep-corpus.tsv"
    
    test_json = "../data/training/sleep-train.json"
  
    # calculate em/f1 for inter-annotators agreement on labels
    labels_agreement(agreement_file)
    
    # calculate avg. number of words in paragraphs/questions/answers
    avg_no_words(file_list, text_corpus)
    
    # calculate question/answer entailment
    calculate_entailment(file_list)
    
    # calculate similarities between questions and answers
    calculate_qa_sim(test_json)
