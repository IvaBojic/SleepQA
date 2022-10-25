from utils import *
import sys
sys.path.append(DPR_LIB_PATH)
from abc import ABC
import json
import logging
import os
import pickle
import torch
from transformers import (
    DPRContextEncoder, DPRContextEncoderTokenizer, 
    DPRQuestionEncoder, DPRQuestionEncoderTokenizer, 
    DPRReader, DPRReaderTokenizer
)
from dpr.indexer.faiss_indexers import DenseFlatIndexer
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class LocalFaissRetriever():
    
    def __init__(self):
        self.index = DenseFlatIndexer()
        self.index.init_index(768)
        self.par_dic = {}
    
        with open(TEXT_CORPUS, "r", encoding = "utf-8") as fin:
            reader = csv.reader(fin, delimiter="\t")
            for row in reader:
                sample_id = str(row[0])
                passage = row[1].strip('"')    
                title = row[2]
                self.par_dic[sample_id] = (title, passage)
        
    def iterate_encoded_files(self, file):
        with open(file, "rb") as reader:
            doc_vectors = pickle.load(reader)
            for doc in doc_vectors:
                doc = list(doc)                
                yield doc
        
    def index_encoded_data(self, file, buffer_size):
        buffer = []
        for item in self.iterate_encoded_files(file):
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


class EndToEndQA(BaseHandler):
    """
    Transformers text classifier handler class. This handler takes a text (string) and
    as input and returns the classification text based on the serialized transformers checkpoint.
    """

    def initialize(self, ctx):
        self.manifest = ctx.manifest

        properties = ctx.system_properties
        model_dir_retriever = MODEL_DIR_RETRIEVER
        model_dir_reader = MODEL_DIR_READER
        corpus_embedded = CORPUS_EMBEDDED
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Read model serialize/pt file
        self.retrieval_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(model_dir_retriever)
        self.retrieval_model = DPRQuestionEncoder.from_pretrained(model_dir_retriever)
            
        self.retriever = LocalFaissRetriever()
        self.retriever.index_encoded_data(corpus_embedded, buffer_size=50000)

        self.retrieval_model.to(self.device)
        self.retrieval_model.eval()

        self.reader_tokenizer = DPRReaderTokenizer.from_pretrained(model_dir_reader)
        self.reader_model = DPRReader.from_pretrained(model_dir_reader)
        self.reader_model.to(self.device)
    
        self.initialized = True

    def preprocess(self, data):
        """ Very basic preprocessing code - only tokenizes. 
            Extend with your own preprocessing steps as needed.
        """
        text = data[0].get("data")
        if text is None:
            text = data[0].get("question")
        question = text.decode('utf-8')
        logger.info("Received question: '%s'", question)
        return question
    

    def docs_retrieval(self, question, topk=1):
        """ 
            Find the top-k documents for a given question.
            Args:
                question (str): question to be answered
            Returns:
                question (str): question to be answered
                titles (list): list of titles of the top-k documents
                passages (list): list of passages of the top-k documents
        """
        tokens = self.retrieval_tokenizer(question,
                        return_tensors="pt",
                        max_length = 256, 
                        padding='max_length', 
                        truncation = True)["input_ids"] 
        tokens[0][255] = 102# add 102 in the end of padding        
        embeddings = self.reader_model(tokens).pooler_output
        q_vec = embeddings.detach().cpu().numpy()
        top_results_and_scores = self.retriever.get_top_docs(q_vec.detach().numpy(), topk=topk)
        for i in range(len(top_results_and_scores)):
            par_id = top_results_and_scores[i][0][0]
            title, passage = self.retriever.par_dic[par_id]
        return question, title, passage

    def answering(self, question, title, passage):
        """
            Answering the question based on the retrieved documents.
            Args:
                question (str): question to be answered
                titles (list): list of titles of the top-k documents
                passages (list): list of passages of the top-k documents
            Returns:
                answer (str): answer to the question
                explaination (str): explaination of the answer - sentences contained answer
        """
        question = question.strip('"')
        title = title.strip('"')
        text = text.strip('"')
        encoded_inputs = self.reader_tokenizer(
            question, title, text, 
            return_tensors = "pt", 
            max_length = 300, 
            padding = 'max_length', truncation = True
        )
        outputs = model(**encoded_inputs)
        predicted_spans = tokenizer.decode_best_spans(encoded_inputs, outputs, max_answer_length = 20, num_spans = 1, num_spans_per_passage = 1)    
        return predicted_spans[0].text

    def inference(self, question):
        """
            Pipeline for inference.
            Args:
                question (str): question to be answered
            Returns:
                answer (str): answer to the question
        """
        prediction = self.model(
            inputs['input_ids'].to(self.device), 
            token_type_ids=inputs['token_type_ids'].to(self.device)
        )[0].argmax().item()
        logger.info("Model predicted: '%s'", prediction)

        if self.mapping:
            prediction = self.mapping[str(prediction)]

        return [prediction]

    def postprocess(self, inference_output):
        return inference_output

    def handle(self, data, context):
        if not self.initialized:
            self.initialize(context)

        if data is None:
            return None

        # Do inference
        question = self.preprocess(data)
        question, title, passage = self.docs_retrieval(question)
        answer = self.answering(question, title, passage)
        return answer
    
if __name__=="__main__":
    class Temp:
        system_properties = {"gpu_id": 0}
    EndToEndQA().initialize(Temp())