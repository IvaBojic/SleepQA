import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from nltk.tokenize import word_tokenize


def avg_no_words(filelist, text_corpus):
      
    no_words_p, no_words_q, no_words_a = [], [], []
    first_word = defaultdict(int)
    
    for file in filelist:
        with open(file, "r", encoding = "utf-8") as fin:
            for line in csv.reader(fin, delimiter = "\t"):
                
                question = line[0]
                answer = line[1].replace('["', "").replace('"]', "").replace('"', '') 
                
                words_q = word_tokenize(question)
                words_a = word_tokenize(answer)
                
                no_words_q.append(len(words_q))
                no_words_a.append(len(words_a))
    
                first_word[words_q[0]] += 1
   
    with open(text_corpus, "r", encoding = "utf-8") as fin:
        for line in csv.reader(fin, delimiter = "\t"):

            paragraph = line[1]         
            words_p = word_tokenize(paragraph)          
            no_words_p.append(len(words_p))
   
    print("Average number of words in paragraphs: {:.1f}".format(sum(no_words_p)/len(no_words_p)))   
    print("Average number of words in questions: {:.1f}".format(sum(no_words_q)/len(no_words_q)))   
    print("Average number of words in answers: {:.1f}".format(sum(no_words_a)/len(no_words_a)))  
    
    for key in first_word.keys():
        print("{}: {:.1f}%".format(key, first_word[key]*100/len(no_words_q)))  
        
    return no_words_p, no_words_q, no_words_a
        
        
def draw_hist(no_words_p, no_words_q, no_words_a):
    
    plt.figure(figsize=(7, 7))  
    plt.hist(no_words_q, alpha = 0.5, bins = np.arange(25)-0.5, edgecolor = 'black', label="questions")
    plt.hist(no_words_a, alpha = 0.5, bins = np.arange(25)-0.5, edgecolor = 'black', label="answers")
    plt.xlabel('Number of words', fontsize = 18)
    plt.ylabel('Quantity', fontsize = 18)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.xlim(xmin = 0, xmax = 30)
    plt.ylim(ymin = 0, ymax = 750)
    plt.legend(loc='upper right')
    plt.savefig("no-words-qa.png", transparent = True, bbox_inches = 'tight')
    plt.close()
    
    plt.figure(figsize=(7, 7))  
    plt.hist(no_words_p, bins = 50)
    plt.xlabel('Number of words', fontsize = 18)
    plt.ylabel('Quantity', fontsize = 18)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.xlim(xmin = 50, xmax = 200)
    plt.ylim(ymin = 0, ymax = 1200)
    plt.savefig("no-words-par.png", transparent = True, bbox_inches = 'tight')
    plt.close()


def main():

    filelist = ["../data/dpr_training/sleep-train.csv", 
                "../data/dpr_training/sleep-dev.csv", 
                "../data/dpr_training/sleep-test.csv"]
    
    text_corpus = "../data/dpr_training/sleep-corpus.tsv"
    
    # calculate avg. number of words in paragraphs/questions/answers
    no_words_p, no_words_q, no_words_a = avg_no_words(filelist, text_corpus)
    # draw histograms
    draw_hist(no_words_p, no_words_q, no_words_a)
    

if __name__ == "__main__":
    main()