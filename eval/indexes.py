import json, csv
from jnius import autoclass


# Step 1 - convert everything to jsonl
# Step 2 - build indexes 


def prepare_json(input_file, output_folder):
    
    counter = 0

    with open (output_folder + "pyserini_db.jsonl", "w", newline='\n', encoding='utf-8') as fout:
       with open(input_file, "r", encoding = "utf-8") as fin: 
          for line in csv.reader(fin, delimiter = "\t"):
                counter += 1
                text = line[1]
                title = line[2]
                output_dict = {
                    'id': title + str(counter), 
                    'contents': text
                    }             
                fout.write(json.dumps(output_dict) + '\n')
                        
    print(counter)
    

def create_indexes(input_folder, output_folder):
    
    args = ['-collection', 'JsonCollection', 
            '-generator', 'DefaultLuceneDocumentGenerator', 
            '-threads', '1', 
            '-input', input_folder, 
            '-index', output_folder, 
            '-storePositions', 
            '-storeDocvectors', 
            '-storeRaw']
    
    JIndexCollection = autoclass('io.anserini.index.IndexCollection')
    JIndexCollection.main(args)

