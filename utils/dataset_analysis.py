import csv, json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from nltk.tokenize import word_tokenize, sent_tokenize
from iteration_utilities import duplicates, unique_everseen

from f1_score import calculate_f1


def read_files(file_list):
    
    q_a = []
    
    for file in file_list:
        with open(file, "r", encoding = "utf-8") as fin:
            for line in csv.reader(fin, delimiter = "\t"):
                
                q = line[0]
                a = line[1].replace('["', "").replace('"]', "").replace('"', '') 
                
                q_a.append([q, a])
    
    return q_a


def avg_no_words(file_list, text_corpus):
      
    no_words_p, no_words_q, no_words_a = [], [], []
    first_word, second_word = defaultdict(int), defaultdict(int)
    
    q_a = read_files(file_list)
    
    for q, a in q_a:
                
        words_q = word_tokenize(q)
        words_a = word_tokenize(a)
        
        no_words_q.append(len(words_q))
        no_words_a.append(len(words_a))

        first_word[words_q[0]] += 1
        second_word[words_q[1]] += 1
   
    with open(text_corpus, "r", encoding = "utf-8") as fin:
        for line in csv.reader(fin, delimiter = "\t"):

            p = line[1]         
            words_p = word_tokenize(p)          
            no_words_p.append(len(words_p))
   
    print("Average number of words in paragraphs: {:.1f}".format(sum(no_words_p)/len(no_words_p)))   
    print("Average number of words in questions: {:.1f}".format(sum(no_words_q)/len(no_words_q)))   
    print("Average number of words in answers: {:.1f}".format(sum(no_words_a)/len(no_words_a)))  
    
    for key in first_word.keys():
        print("{}: {:.1f}%".format(key, first_word[key]*100/len(no_words_q)))  
    
    # draw histograms of par/q/a lenghts    
    draw_hist(no_words_p, no_words_q, no_words_a)
    draw_pie(second_word)  
    
        
def draw_hist(no_words_p, no_words_q, no_words_a):
    
    plt.figure(figsize=(7, 7))  
    plt.hist(no_words_q, alpha = 0.5, bins = np.arange(25), edgecolor = 'black', label="questions")
    plt.hist(no_words_a, alpha = 0.5, bins = np.arange(25), edgecolor = 'black', label="answers")
    plt.xlabel('Number of words', fontsize = 18)
    plt.ylabel('Number of questions/answers', fontsize = 18)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.xlim(xmin = 0, xmax = 25)
    plt.ylim(ymin = 0, ymax = 750)
    plt.legend(loc='upper right')
    plt.savefig("no-words-qa.pdf", transparent = True, bbox_inches = 'tight')
    plt.close()
    
    plt.figure(figsize=(7, 7))  
    plt.hist(no_words_p, bins = 25, alpha = 0.5, edgecolor = 'black', color='green')
    plt.xlabel('Number of words', fontsize = 18)
    plt.ylabel('Number of paragraphs', fontsize = 18)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.xlim(xmin = 50, xmax = 220)
    plt.ylim(ymin = 0, ymax = 2100)
    plt.savefig("no-words-par.pdf", transparent = True, bbox_inches = 'tight')
    plt.close()


def draw_pie(second_word):
 
    total, others = 0, 0
    values, labels = [], []
    for key in dict(sorted(second_word.items(), reverse = True, key = lambda item: item[1])):
         # first five most popular words
         if(total < 5):
             values.append(second_word[key])
             labels.append(key)
         else:
             others += second_word[key]
         total += 1
    
    values.append(others)       
    labels.append("others")
    
    colors = sns.color_palette("husl", 6)
    plt.figure(figsize=(9, 9))  
    plt.pie(values, labels = labels, colors = colors, autopct = '%.0f%%', textprops = {'fontsize': 18})
    plt.savefig("second-word-qns.pdf", transparent = True, bbox_inches = 'tight')
    plt.close()
    

def calculate_entailment(file_list):
  
    ent_a, ent_q = defaultdict(set), defaultdict(set)
    
    q_a = read_files(file_list) 
    dup_q = list(unique_everseen(duplicates([item[0] for item in q_a])))
    dup_a = list(unique_everseen(duplicates([item[1] for item in q_a])))
  
    for q, a in q_a:
         if(a in dup_a):
             ent_q[a].add(q)
         if(q in dup_q):
             ent_a[q].add(a)
    
    print("Number of question entailment: {}".format(sum([len(list(ent_q[x])) for x in ent_q if isinstance(list(ent_q[x]), list)])))
    print("Number of answer entailment: {}".format(sum([len(list(ent_a[x])) for x in ent_a if isinstance(list(ent_a[x]), list)])))
   

def calculate_qa_sim(test_json):
    
    counter = 0
    f1 = []
    
    with open(test_json, "r", encoding = "utf-8") as fin: 
        objs = json.load(fin)
        for obj in objs:
            q = obj['question']
            a = obj['answers'][0]
            
            try:
                p = obj['positive_ctxs'][0]['text']
                
                if(p.count(a) == 1):
                    counter += 1
                   
                    if(counter == 1000):
                        break
                    
                    sentences = sent_tokenize(p)
                    
                    for sentence in sentences:
                        if(sentence.find(a) != -1):
                            
                            macro_f1 = calculate_f1(q, sentence)   
                            f1.append(macro_f1)
            except:
                pass
                  
    print("Similarities between questions and answers (f1): {:.2f}".format(sum(f1)/len(f1)))    