import csv
from collections import defaultdict

from f1_score import prep, calculate_f1


def labels_agreement(agreement_file):
    
    f1, em = [], []
   
    a = defaultdict(list)

    # main one first
    with open(agreement_file, "r", encoding = "utf-8") as fin:
        for line in csv.reader(fin):
            # no#answer#is_main
            if(line[2] == "TRUE"): 
                a[int(line[0])].append(prep(line[1]))
              
    # others after
    with open(agreement_file, "r", encoding = "utf-8") as fin:
        for line in csv.reader(fin):
            if(line[2] == "FALSE" and int(line[0]) in a.keys()):
                a[int(line[0])].append(prep(line[1]))
   
    for key in a:
       
        if(len(a[key]) != 5):
            print(a[key])
           
        max_f1 = -1
        for i in range(1, 5):
            macro_f1 = calculate_f1(a[key][0], a[key][i]) 
            if(macro_f1 > max_f1):
                max_f1 = macro_f1
           
        f1.append(max_f1)       
        
        if(max_f1 == 1):
            em.append(1)
        else:
            em.append(0)
   
    print("em: {:.2f}".format(sum(em)/len(em)))
    print("f1: {:.2f}".format(sum(f1)/len(f1)))
    
