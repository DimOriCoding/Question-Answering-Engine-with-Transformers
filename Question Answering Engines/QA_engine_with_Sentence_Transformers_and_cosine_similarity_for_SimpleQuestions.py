#!/usr/bin/env python
# coding: utf-8

# **I import some useful libraries in python.**

# In[ ]:


get_ipython().system('pip install unidecode')
import pandas as pd
import numpy as np
import json
import csv
import difflib
from unidecode import unidecode
import random
import torch
import torch.backends.cudnn
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
from google.colab import drive
import torch.nn as nn
from transformers import BertModel
from transformers import BertTokenizerFast
import sys


# **I mount google drive because all the files (for example .csv) that i used for the implementation of my thesis are saved on my google drive.**

# In[ ]:


drive.mount('/content/drive')


# **Check of the device that it is used for the implementation of the notebooks of this thesis, that is determined from google colab.**

# In[ ]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device available for running: ")
print(device)


# **Reading of the created datasets (for both the SimpleQuestions dataset that includes all training, validation and test questions and the SimpleQuestions subset that includes only answerable over wikidata questions) as .csv file.  With the same way the corresponding test file that contains the questions with their relation id, entity id, answer id and answer label and is used for testing the QA engines is read as .csv file.**

# In[ ]:


train_dataset = pd.read_csv("drive/MyDrive/train_dataset.csv", sep = ',')
validation_dataset = pd.read_csv("drive/MyDrive/valid_dataset.csv", sep = ',')
test_dataset = pd.read_csv("drive/MyDrive/test_dataset.csv", sep = ',')
print(train_dataset)
print(validation_dataset)
print(test_dataset)


# In[ ]:


train_answerable_dataset = pd.read_csv("drive/MyDrive/train_answerable_questions_dataset.csv", sep = ',')
validation_answerable_dataset = pd.read_csv("drive/MyDrive/valid_answerable_questions_dataset.csv", sep = ',')
test_answerable_dataset = pd.read_csv("drive/MyDrive/test_answerable_dataset.csv", sep = ',')
print(train_answerable_dataset)
print(validation_answerable_dataset)
print(test_answerable_dataset)


# In[ ]:


testing_questions_list = pd.read_csv("drive/MyDrive/Questions_testing_with_answers.txt", sep = ',', names=['Question','entity_id', 'relation_id','answer_id', 'answer_label'])
print(testing_questions_list)


# **Check if the test file with the questions that is used for testing the QA engines, the SimpleQuestions training, validation and test dataset have at least one line with null value. This issues both for SimpleQuestions dataset that includes only answerable over wikidata questions and for the corresponding dataset that includes all the questions.**

# In[ ]:


train_dataset.dropna(inplace = True)
validation_dataset.dropna(inplace = True)
test_dataset.dropna(inplace = True)
print(train_dataset)
print(validation_dataset)
print(test_dataset)


# In[ ]:


train_answerable_dataset.dropna(inplace = True)
validation_answerable_dataset.dropna(inplace = True)
test_answerable_dataset.dropna(inplace = True)
print(train_answerable_dataset)
print(validation_answerable_dataset)
print(test_answerable_dataset)


# In[ ]:


testing_questions_list.dropna(inplace = True)
print(testing_questions_list)


# **Creation of the corresponding lists containing the questions, the entity labels, the relation ids and the entity ids of the corresponding SimpleQuestions datasets (training, validation, test), either this dataset includes answerable and answerable questions over wikidata questions or only answerable questions over wikidata.**

# In[ ]:


train_entity_labels = train_dataset['entity_label'].to_list()
validation_entity_labels = validation_dataset['entity_label'].to_list()
test_entity_labels = test_dataset['entity_label'].to_list()
train_entityids = train_dataset['entity_id'].to_list()
validation_entityids = validation_dataset['entity_id'].to_list()
test_entityids = test_dataset['entity_id'].to_list()
train_relationids = train_dataset['relation_id'].to_list()
validation_relationids = validation_dataset['relation_id'].to_list()
test_relationids = test_dataset['relation_id'].to_list()
train_questions = train_dataset['Question'].to_list()
validation_questions = validation_dataset['Question'].to_list()
test_questions = test_dataset['Question'].to_list()
print(len(train_entity_labels), len(train_entityids), len(train_relationids), len(train_questions))


# In[ ]:


train_answerable_entity_labels = train_answerable_dataset['entity_label'].to_list()
validation_answerable_entity_labels = validation_answerable_dataset['entity_label'].to_list()
test_answerable_entity_labels = test_answerable_dataset['entity_label'].to_list()
train_answerable_entityids = train_answerable_dataset['entity_id'].to_list()
validation_answerable_entityids = validation_answerable_dataset['entity_id'].to_list()
test_answerable_entityids = test_answerable_dataset['entity_id'].to_list()
train_answerable_relationids = train_answerable_dataset['relation_id'].to_list()
validation_answerable_relationids = validation_answerable_dataset['relation_id'].to_list()
test_answerable_relationids = test_answerable_dataset['relation_id'].to_list()
train_answerable_questions = train_answerable_dataset['Question'].to_list()
validation_answerable_questions = validation_answerable_dataset['Question'].to_list()
test_answerable_questions = test_answerable_dataset['Question'].to_list()
print(len(train_answerable_entity_labels), len(train_answerable_entityids), len(train_answerable_relationids), len(train_answerable_questions))


# **Creation of the corresponding lists containing the questions, the entity labels, the relation ids and the entity ids of the corresponding test .txt file that includes the questions that are used in order to test QA engine.**

# In[ ]:


testing_questions = testing_questions_list['Question'].to_list()
testing_questions_entity_ids = testing_questions_list['entity_id'].to_list()
testing_questions_relation_ids = testing_questions_list['relation_id'].to_list()
testing_questions_answer_ids = testing_questions_list['answer_id'].to_list()
testing_questions_answer_labels = testing_questions_list['answer_label'].to_list()
print(len(testing_questions),len(testing_questions_relation_ids),len(testing_questions_entity_ids),len(testing_questions_answer_ids),len(testing_questions_answer_labels))


# **The preprocessing for each question is made by creating the function preprocessing, which takes as argument the text (in this case the question) and performs accent removal, as well as punctuation mark removal (period, question mark, exclamation mark), s removal with apostrophe('s) and comma(,) removal.Finally this function, which is shown below, returns the original text with the above mentioned processing.**

# In[ ]:


def Preprocessing(text):
    text = unidecode(text.replace("?", "").replace("'s", "").replace(".", "").replace("!", "").replace(",", ""))
    return text


# **The maximum question length that is observed in SimpleQuestions training, validation and test dataset is found below.**

# In[ ]:


#Below the max questions length is found for all questions.
quests = train_questions + validation_questions + test_questions
print(len(quests), len(train_questions), len(validation_questions))
# Printing concatenated list
q_lengths = []
for s in range(len(quests)): #padding
    q_lengths.append(len(quests[s].split()))
print(max(q_lengths))#Max question length


# In[ ]:


#Below the max questions length is found for all questions.
#quests = pd.concat([train_dataset["Question"], validation_dataset["Question"]], ignore_index = True)
quests_answerable = train_answerable_questions + validation_answerable_questions + test_answerable_questions
print(len(quests_answerable), len(train_answerable_questions), len(validation_answerable_questions))
# Printing concatenated list
q_lengths_answerable = []
for s in range(len(quests_answerable)): #padding
    q_lengths_answerable.append(len(quests_answerable[s].split()))
print(max(q_lengths_answerable))#Max question length


# **Installation of the library of Sentence Transformers and loading of their models.**

# In[ ]:


get_ipython().system('pip install sentence-transformers')
from sentence_transformers import SentenceTransformer
similarity_model = SentenceTransformer("all-MiniLM-L6-v2")


# **Creation of the relation vocabulary that includes all the unique entity ids of the SimpleQuestions training and validation dataset through function relation_vocabulary. Also the index of each unique relation id of the relation vocabulary in this vocabulary is found through the function find_relations_index.**

# In[ ]:


#Create relation vocabulary for each question of train and validation dataset
def relation_vocabulary(relation_column_dataset):
  relation_vocabulary = []
   #In order to have the unique relation ids it is important to check if an relation id is already in the relation_vocabulary
  for relation in range(len(relation_column_dataset)):
    g = 0
    for i in range(len(relation_vocabulary)):
      if relation_column_dataset[relation] != relation_vocabulary[i]: #relation is not found yet
        g += 1
      else:
       continue
    if g == len(relation_vocabulary) and relation_column_dataset[relation] not in relation_vocabulary:
      relation_vocabulary.append(relation_column_dataset[relation]) #The relation id is not inside the relation vocabulary
      #which has some relations already or the relation vocaulary has no relations
  return relation_vocabulary

def find_relation_index(relation_vocabulary,relation_ids_column):
  relations_count = []
  for relation in range(len(relation_ids_column)):
        relations_count.append(relation_vocabulary.index(relation_ids_column[relation]))

  return relations_count
relation_ids = train_relationids + validation_relationids
relation_answerable_ids = train_answerable_relationids + validation_answerable_relationids
rel_vocab = relation_vocabulary(relation_ids)
rel_answerable_vocab = relation_vocabulary(relation_answerable_ids)
print(len(rel_vocab), rel_vocab)
print(len(rel_answerable_vocab), rel_answerable_vocab)


# **Custom function for the cosine similarity metric between two encodings.**

# In[ ]:


# Custom function for cosine similarity
def cosine_similarity(encoding1, encoding2):
  return np.dot(encoding1, encoding2) / (np.linalg.norm(encoding1) * np.linalg.norm(encoding2)) if np.linalg.norm(encoding1) * np.linalg.norm(encoding2) != 0 else 0


# **Function that contributes to the preservation of the reproducibility.**

# In[ ]:


#To preserve reproducibility
SEED = 42
def set_seed(SEED):
 random.seed(SEED)# Set python seed for custom operators.
 torch.manual_seed(SEED)
 np.random.seed(SEED)
 torch.cuda.manual_seed_all(SEED)
 torch.backends.cudnn.benchmark = False
 torch.backends.cudnn.deterministic = True


# ****Construction and initialization of the neural network for the prediction of the relation for each question.****

# In[ ]:


class BertRelationsClassifier(nn.Module):
  def __init__(self, activation, dropout_prob, rel_vocab_len):
    super(BertRelationsClassifier, self).__init__()
    activations = {
        "ELU" : nn.ELU,
        "LeakyReLU"    : nn.LeakyReLU,
        "Softplus"     : nn.Softplus,
        "Tanhshrink"     : nn.Tanhshrink
 }
    self.bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
    self.rel_vocab_len = rel_vocab_len
    self.device = device
    self.fc = nn.Linear(768, self.rel_vocab_len , device = self.device)
    self.activation_type = activation
    self.activation = activations[self.activation_type]()
    self.dropout_probability = dropout_prob
    self.dropout = nn.Dropout(self.dropout_probability)
    self.softmax = nn.Softmax(dim=1)

  def forward(self, input_ids, attention_mask):
        out = self.bert_model(input_ids, attention_mask)
        out = self.fc(out[1])#CLS token-pooled output
        out = self.activation(out)
        out = self.dropout(out)
        out = self.softmax(out)
        return out


# ****Construction and initialization of the neural network for the prediction of the entity span start and entity span end indexes for each question.****

# In[ ]:


class BertEntitySpanClassifier(nn.Module):
  def __init__(self, activation, dropout_prob, max_quest_len):
    super(BertEntitySpanClassifier, self).__init__()
    self.bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
# Whether the model returns all hidden-states.
    activations = {
        "ELU" : nn.ELU,
        "LeakyReLU"    : nn.LeakyReLU,
        "Softplus"     : nn.Softplus,
        "Tanhshrink"     : nn.Tanhshrink
 }
    self.activation_type = activation
    self.activation = activations[self.activation_type]()
    self.max_length = max_quest_len
    self.dropout_probability = dropout_prob
    self.start_head = nn.Sequential(
          nn.Linear(768, self.max_length),
          self.activation,
          nn.Dropout(p=self.dropout_probability),
          nn.Flatten(),
          nn.Softmax(dim=1)
    )

    self.end_head = nn.Sequential(
          nn.Linear(768, self.max_length),
          self.activation,
          nn.Dropout(p=self.dropout_probability),
          nn.Flatten(),
          nn.Softmax(dim=1)
    )

  def forward(self, input_ids, att_mask):
    out = self.bert_model(input_ids, att_mask)
    out = out[0]
    out_start = self.start_head(out)
    out_end = self.end_head(out)
    return out_start , out_end


# **Loading of the Transformers BERT tokenizer BertTokenizerFast.**

# In[ ]:


tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')


# **The SPARQLWrapper library is downloaded, which contributes to the generation of SPARQL queries, in order to find the entity labels.**

# In[ ]:


get_ipython().system('pip install sparqlwrapper')
from SPARQLWrapper import SPARQLWrapper, JSON, CSV
sparql = SPARQLWrapper("https://query.wikidata.org/sparql", agent= "WDQS-example Python/%s.%s" % (sys.version_info[0], sys.version_info[1]))


# **QUESTION ANSEWRING ENGINE**

# **I create the get\_entity\_id function which involves creating a SPARQL query that, given an entity label, finds and extracts its corresponding entity id.**

# In[ ]:


def get_entity_id(entity_label):
    sparql = SPARQLWrapper('https://query.wikidata.org/sparql', agent = "WDQS-example Python/%s.%s" % (sys.version_info[0], sys.version_info[1]))
    sparql.setQuery("""
    SELECT distinct ?item ?itemLabel ?itemDescription
    WHERE
    {
     ?item ?label "%s"@en.
     ?article schema:about ?item .
     ?article schema:inLanguage "en" .
     ?article schema:isPartOf <https://en.wikipedia.org/>.
     SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
    }"""  % (entity_label.title()))
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    entityids = []
    for result in results["results"]["bindings"]:
        entityids.append(result["item"]["value"].split(sep='/')[-1])
    return entityids


# **Creation of two dictionaries, where the first one named Endict will have as keys the entity labels of the SimpleQuestions training and validation dataset and as values the corresponding entity ids of the SimpleQuestions training and validation dataset and the other one named quests will have as keys the entity labels of the SimpleQuestions training and validation dataset and as values the corresponding encoding vectors from the similarity model of Sentence Transformer for each entity label of the SimpleQuestions training and validation dataset.**

# In[ ]:


# Create a dictionary with entity labels as keys and entity_ids as values
entity_ids = train_entityids + validation_entityids + test_entityids
entity_labels = train_entity_labels + validation_entity_labels + test_entity_labels
Endict = {}
for h in range(len(entity_ids)):
  label = entity_labels[h]
  id = entity_ids[h]
  Endict[label] = id
entities = list(Endict.keys())
quests = {}
# Create a dictionary with Sentence transfomer embeddings for each entity label.
for entity_label, id in Endict.items():
    quest = similarity_model.encode(entity_label)
    quests[entity_label] = quest


# **I create the function question\_encoding that takes as arguments the question and the corresponding tokenizer, in this case BertTokenizerFast.
# Finally the function question\_encoding encodes the given question through the command tokenizer.encode\_plus(). Also this function calls the Preprocessing function in order to make the preprocessing of each question (described in the previous cell with text). The ids and masks resulting from the corresponding encoding are returned.**
# 
# **Creation of the function find\_closest\_match that takes as arguments the entity label, the dictionary Endict with keys the entity labels and values the entity ids, a list of all keys of Endict, and if Sentence Transformer library is used the dictionary entity\_quests with keys the entity labels and values the encodings obtained for each entity label through the pretrained similarity model of Sentence Transformer.**
# 
# **Through this function the entity id for the entity of a question is found either through spacy library or cosine similarity metric or jaccard similarity metric between the entity label of a user question and the list of Endict keys, i.e the entity labels of the SimpleQuestions training and validation dataset (or the corresponding encodings if the Sentence Transformer similarity model is used) and the more similar entity label is returned, if the entity id for the entity of a query cannot be found through the get\_entity\_id function. In this case the entity id of the entity that has the best cosine similarity score with the entity for which we are looking for the entity id is returned, if available**

# In[ ]:


def question_encoding(question,tokenizer):
    question = Preprocessing(question)
    ids = []
    mask = []
    # Encode the question
    encoding = tokenizer.encode_plus(
        text = question,
        return_attention_mask=True,
        add_special_tokens=False
    )

    ids.append(encoding['input_ids'])
    mask.append(encoding['attention_mask'])

    return torch.tensor(ids), torch.tensor(mask)



def find_closest_match(entity_label, Endict, entities, quests, similarity_model):
      ent = similarity_model.encode(entity_label)
      similarity_scores = {}
      # Find similarity scores between entity and each entity in dictionary
      for ent_label, Ent in quests.items():
          similarity_scores[ent_label] = cosine_similarity(ent,Ent)
      Q = dict(sorted(similarity_scores.items(), key = lambda x: x[1], reverse = True))
      EntitiesList = [key for key in Q]
      if EntitiesList != []:
        entityid = Endict[EntitiesList[0]]
        print("Cosine similarity metric is used!")
        return entityid
      else:
        return None


# **Creation of functions to predict entity span start index, entity span end index and relation index. The function for predicting the entity span start index and entity span end index takes as arguments the ids and the mask resulting from the encoding of the question via the question\_encoding function, the aforementioned dictionaries Endict and entity\_quests, the pretrained model of the Sentence Transformer similarity model, the entity\_model\_path, i.e. the path where the best parameters for the best predictive model of entity span start index and entity span end index are stored, the activation function for the best predictive model of entity start index and entity end index, the dropout probability, dropout\_prob and the SEED number which helps to have the same and reproducible answers. The predictions for entity span start index and entity span end index, are returned by this function.**
# 
# **Also in the case that for a QA engine i do not use the Sentence Transformers similarity model, i.e for the QA engine include either only spacy library or jaccard similarity metric, the function entity\_prediction does not have the dictionary quests and the Sentence Transformers similarity model as arguments.**
# 

# In[ ]:


def entity_prediction(ids, mask, Endict, entities, tokenizer, quests, similarity_model, max_question_length, activation, entity_model_path, dropout_prob, SEED):
    #Loading the best model for entity span prediction
    set_seed(SEED)
    entity_model = BertEntitySpanClassifier(activation, dropout_prob, max_question_length).to(device)
    entity_model.load_state_dict(torch.load(entity_model_path))
    entity_model.eval()
    y_start, y_end = entity_model(ids, mask)
    y_pred_start = torch.argmax(y_start, dim=1)
    y_pred_end = torch.argmax(y_end, dim=1)
    entity = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(ids[0][y_pred_start.item():y_pred_end.item()+1]))
    print(entity)
    if entity in Endict:
        return Endict[entity]
    else:
        entityid = get_entity_id(entity)

        if entityid != []:
            return entityid[0]
    # Fallback to the most similar from the dictionary
    return find_closest_match(entity, Endict, entities, quests, similarity_model)


# **Correspondingly the function for the prediction of relation index, takes as arguments the ids and mask resulting from the encoding of the question through the question\_encoding function, the length of the relation vocabulary, the relation\_model\_path, i.e. the path where the best parameters for the best prediction model of relation index are stored, the activation function of the best prediction model of relation index prediction,the dropout probability, dropout\_prob and the SEED number. The prediction for the relation index gives through the relation vocabulary which relation id is in the question. This relation index is returned.**

# In[ ]:


def relation_prediction(ids, mask, rel_vocab, rel_vocab_len, activation_relation, dropout_prob, relation_model_path, SEED):
    set_seed(SEED)
    relation_model = BertRelationsClassifier(activation_relation, dropout_prob, rel_vocab_len).to(device)
    relation_model.load_state_dict(torch.load(relation_model_path))
    relation_model.eval()
    rel = relation_model(ids, mask).to(device)
    rel_pred = torch.argmax(rel, dim=1)
    relation = rel_vocab[rel_pred]
    return relation


# **To extract the final answer or answers for each question, i create the function named answer which takes as arguments the question itself, the dictionaries Endict, the dictionary entity\_quests if the Sentence Transformer similarity model is used, the tokenizer BertTokenizerFast, the pretrained similarity model of the Sentence Transformer if the cosine similarity metric is used, the entity\_model\_path, i.e. the path where the best parameters for the each one of the models created of entity start index and entity end index are stored, the relation vocabulary, the relation\_model\_path, i.e. the path where the best parameters for the best prediction model of the relation index are stored, the activation function of the best prediction model  of the relation index, the activation functions of the best prediction models of the entity start index and entity end index  and the dropout probability, dropout\_prob.**
# 
# **Also the answer function calls the aforementioned function question\_encoding in order to encode the question and get the ids and the mask that are needed for the functions entity\_prediction and relation\_prediction, which are also called from the function answer. This function returns a boolean value which shows if a question has an totally correct answer and also a boolean value which shows if a question has answer(s) either correct or incorrect or both of them, regardless if these answers include the expected answer of a question.  Additionally the answers of this question, with the limitation of maximum 15 answers, are printed.**

# In[ ]:


def get_answer(entityid, relation):
    sparql = SPARQLWrapper('https://query.wikidata.org/sparql', agent = "WDQS-example Python/%s.%s" % (sys.version_info[0], sys.version_info[1]))
    if entityid is not None:
     if (relation[0] == 'P'):
        query = """
        SELECT ?item ?itemLabel
        WHERE
        {
          wd:""" + entityid + """ wdt:""" + relation + """ ?item.
          SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
        }"""
     else:
        relation = relation.replace('R', 'P')
        query = """
        SELECT ?item ?itemLabel
        WHERE
        {
          ?item wdt:""" + relation + """ wd:""" + entityid + """.
          SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
        }"""
     sparql.setQuery(query)
     sparql.setReturnFormat(JSON)
     results = sparql.query().convert()
     return results["results"]["bindings"]


# In[ ]:


def answer(question, Endict, entities, rel_vocab, rel_vocab_len, tokenizer, quests, similarity_model, max_question_length, activation_relation, activation_entity, relation_model_path, entity_model_path, dropout_prob, SEED, real_answer, real_entity_id, real_relation_id):
    #Boolean variables that denotes if a question is answered
    question_answered = 0
    question_correct_answered = 0
    #Encode question
    ids, mask = question_encoding(question, tokenizer)
    ids = ids.to(device)
    mask = mask.to(device)
    # Loading the best model for relation prediction and predict relation id
    relation = relation_prediction(ids, mask, rel_vocab, rel_vocab_len, activation_relation, dropout_prob, relation_model_path, SEED)
    print(relation)
    # Loading the best model for entity span prediction and predict entity id
    entity = entity_prediction(ids, mask, Endict, entities, tokenizer, quests, similarity_model, max_question_length, activation_entity, entity_model_path, dropout_prob, SEED)
    print(entity)
    real_ans = similarity_model.encode(real_answer)
    if entity == None:
        print("Sorry, no answer available")
    else:
        answers = get_answer(entity, relation)
        if answers == []:
          print("Sorry, no answer available")
        else:
          model_answers = []
          for iter, answer in enumerate(answers):
             if iter == 15:
                break
             model_answers.append(answer["itemLabel"]["value"])
          answer_similarity_scores = {}
          for iter, answer in enumerate(model_answers):
              ans = similarity_model.encode(answer)
              answer_similarity_scores[answer] = cosine_similarity(real_ans,ans)
          Answers_sorted = dict(sorted(answer_similarity_scores.items(), key = lambda x: x[1], reverse = True))
          best_similarity_scores = list(Answers_sorted.values())
          best_answers_sorted = list(Answers_sorted.keys())
          # Print the answers occured from best model parameters
          # and the similarity scores of each answer with the expected answer
          for sorted_answer in range(len(best_answers_sorted)):
            print('answer: ', best_answers_sorted[sorted_answer],',','similarity score with the expected answer: {:.2f}'.format(best_similarity_scores[sorted_answer]))
          if best_similarity_scores != []:
            min_similarity_answer_score = min(best_similarity_scores)
            max_similarity_answer_score = max(best_similarity_scores)
          if entity != real_entity_id and relation != real_relation_id:
             print("The question is not answered correctly!")
          elif entity == real_entity_id and relation == real_relation_id:
            if (max_similarity_answer_score > 0.8 and len(best_answers_sorted) == 1) or (max_similarity_answer_score > 0.8 and min_similarity_answer_score > 0.05 and len(best_answers_sorted) > 1):
             question_correct_answered = 1
             print("The question is answered correctly!")
            else:
             question_answered = 1
             print("The question is answered with either only correct answers or only incorrect answers or with both of them,\nwhich are different and either some or all of them have cosine similarity score with the expected answer significantly lower than 0.8,\nregardless if the expected answer is one of the possible correct answers or not!")
          else:
            if (max_similarity_answer_score > 0.8 and len(best_answers_sorted) == 1) or (max_similarity_answer_score > 0.8 and min_similarity_answer_score > 0.05 and len(best_answers_sorted) > 1):
             question_correct_answered = 1
             print("The question is answered correctly, although either entity id or relation id is not predicted correctly!")
            else:
             question_answered = 1
             print("The question is answered with either only correct answers or only incorrect answers or with both of them,\nwhich are different and either some or all of them have cosine similarity score with the expected answer significantly lower than 0.8,\nregardless if the expected answer is one of the possible correct answers or not!")
    return question_correct_answered, question_answered


# **Question Answering engine for all the questions (answerable, non-answerable over wikidata), with the best papameters for each methodology.**

# In[ ]:


#Question Answering engine
activation_relation_best_parameters_path = 'drive/MyDrive/Relation_indexes_models_best_parameters/All_Questions/ELU/relation_prediction_span_encoded_ignoring_questions_ELU.pt'
activation_entity_best_parameters_path =  "drive/MyDrive/Entity_span_indexes_models_best_parameters/All_Questions/ELU/entity_span_encoded_prediction_ignoring_questions_ELU.pt"
print('This is the question answering engine. Type exit to quit')
dropout_prob = 0.3
count_test_questions_total = 0
count_test_questions_correct_answered = 0
count_test_questions_answered = 0
while(1):
  question = input()
  if question == "exit":
    break;
  else:
    count_test_questions_total += 1
  #Find the real answer, relation id and entity id of the input question from the corresponding .txt file.
  for i in range(len(testing_questions)):
    if testing_questions[i] == question:
      real_question_answer_label = testing_questions_answer_labels[i]
      real_question_entity_id = testing_questions_entity_ids[i]
      real_question_relation_id = testing_questions_relation_ids[i]
      break
  question_correct_answered, question_answered = answer(question, Endict, entities, rel_vocab, len(rel_vocab), tokenizer, quests, similarity_model, max(q_lengths), "ELU", "ELU", activation_relation_best_parameters_path, activation_entity_best_parameters_path, dropout_prob, SEED, real_question_answer_label, real_question_entity_id, real_question_relation_id)
  if question_correct_answered == 1:
    count_test_questions_correct_answered += 1
  if question_answered == 1:
    count_test_questions_answered += 1
print(count_test_questions_correct_answered)
print(count_test_questions_answered)
print(count_test_questions_total)
print("The percentage of the test questions that are answered correct is: {:.2f}%".format((count_test_questions_correct_answered/count_test_questions_total)*100))
print("The percentage of the test questions that are answered either with either only correct or only incorrect answers or both of them,\nregardless if the expected answer is one of these correct answer(s) or not: {:.2f}%".format((count_test_questions_answered/count_test_questions_total)*100))


# In[ ]:


#Question Answering engine
activation_relation_best_parameters_path = 'drive/MyDrive/Relation_indexes_models_best_parameters/All_Questions/ELU/relation_prediction_span_encoded_not_ignoring_questions_ELU.pt'
activation_entity_best_parameters_path =  "drive/MyDrive/Entity_span_indexes_models_best_parameters/All_Questions/Softplus/entity_span_encoded_prediction_not_ignoring_questions_Softplus.pt"
print('This is the question answering engine. Type exit to quit')
dropout_prob = 0.3
count_test_questions_total = 0
count_test_questions_correct_answered = 0
count_test_questions_answered = 0
while(1):
  question = input()
  if question == "exit":
    break;
  else:
    count_test_questions_total += 1
  #Find the real answer, relation id and entity id of the input question from the corresponding .txt file.
  for i in range(len(testing_questions)):
    if testing_questions[i] == question:
      real_question_answer_label = testing_questions_answer_labels[i]
      real_question_entity_id = testing_questions_entity_ids[i]
      real_question_relation_id = testing_questions_relation_ids[i]
      break
  question_correct_answered, question_answered = answer(question, Endict, entities, rel_vocab, len(rel_vocab), tokenizer, quests, similarity_model, max(q_lengths), "ELU", "Softplus", activation_relation_best_parameters_path, activation_entity_best_parameters_path, dropout_prob, SEED, real_question_answer_label, real_question_entity_id, real_question_relation_id)
  if question_correct_answered == 1:
    count_test_questions_correct_answered += 1
  if question_answered == 1:
    count_test_questions_answered += 1
print(count_test_questions_correct_answered)
print(count_test_questions_answered)
print(count_test_questions_total)
print("The percentage of the test questions that are answered correct is: {:.2f}%".format((count_test_questions_correct_answered/count_test_questions_total)*100))
print("The percentage of the test questions that are answered either with either only correct or only incorrect answers or both of them,\nregardless if the expected answer is one of these correct answer(s) or not: {:.2f}%".format((count_test_questions_answered/count_test_questions_total)*100))


# In[ ]:


#Question Answering engine
activation_relation_best_parameters_path = 'drive/MyDrive/Relation_indexes_models_best_parameters/All_Questions/ELU/relation_prediction_cosine_similarity_ELU.pt'
activation_entity_best_parameters_path =  "drive/MyDrive/Entity_span_indexes_models_best_parameters/All_Questions/ELU/entity_span_prediction_cosine_similarity_ELU.pt"
print('This is the question answering engine. Type exit to quit')
dropout_prob = 0.3
count_test_questions_total = 0
count_test_questions_correct_answered = 0
count_test_questions_answered = 0
while(1):#
  question = input()
  if question == "exit":
    break;
  else:
    count_test_questions_total += 1
#Find the real answer, relation id and entity id of the input question from the corresponding .txt file.
  for i in range(len(testing_questions)):
    if testing_questions[i] == question:
      real_question_answer_label = testing_questions_answer_labels[i]
      real_question_entity_id = testing_questions_entity_ids[i]
      real_question_relation_id = testing_questions_relation_ids[i]
      break
  question_correct_answered, question_answered = answer(question, Endict, entities, rel_vocab, len(rel_vocab), tokenizer, quests, similarity_model, max(q_lengths), "ELU", "ELU", activation_relation_best_parameters_path, activation_entity_best_parameters_path, dropout_prob, SEED, real_question_answer_label, real_question_entity_id, real_question_relation_id)
  if question_correct_answered == 1:
    count_test_questions_correct_answered += 1
  if question_answered == 1:
    count_test_questions_answered += 1
print(count_test_questions_correct_answered)
print(count_test_questions_answered)
print(count_test_questions_total)
print("The percentage of the test questions that are answered correct is: {:.2f}%".format((count_test_questions_correct_answered/count_test_questions_total)*100))
print("The percentage of the test questions that are answered either with either only correct or only incorrect answers or both of them,\nregardless if the expected answer is one of these correct answer(s) or not: {:.2f}%".format((count_test_questions_answered/count_test_questions_total)*100))


# In[ ]:


#Question Answering engine
activation_relation_best_parameters_path = 'drive/MyDrive/Relation_indexes_models_best_parameters/All_Questions/ELU/relation_prediction_jaccard_similarity_ELU.pt'
activation_entity_best_parameters_path =  "drive/MyDrive/Entity_span_indexes_models_best_parameters/All_Questions/Softplus/entity_span_prediction_jaccard_similarity_Softplus.pt"
print('This is the question answering engine. Type exit to quit')
dropout_prob = 0.3
count_test_questions_total = 0
count_test_questions_correct_answered = 0
count_test_questions_answered = 0
while(1):
  question = input()
  if question == "exit":
    break;
  else:
    count_test_questions_total += 1
  #Find the real answer, relation id and entity id of the input question from the corresponding .txt file.
  for i in range(len(testing_questions)):
    if testing_questions[i] == question:
      real_question_answer_label = testing_questions_answer_labels[i]
      real_question_entity_id = testing_questions_entity_ids[i]
      real_question_relation_id = testing_questions_relation_ids[i]
      break
  question_correct_answered, question_answered = answer(question, Endict, entities, rel_vocab, len(rel_vocab), tokenizer, quests, similarity_model, max(q_lengths), "ELU", "Softplus", activation_relation_best_parameters_path, activation_entity_best_parameters_path, dropout_prob, SEED, real_question_answer_label, real_question_entity_id, real_question_relation_id)
  if question_correct_answered == 1:
    count_test_questions_correct_answered += 1
  if question_answered == 1:
    count_test_questions_answered += 1
print(count_test_questions_correct_answered)
print(count_test_questions_answered)
print(count_test_questions_total)
print("The percentage of the test questions that are answered correct is: {:.2f}%".format((count_test_questions_correct_answered/count_test_questions_total)*100))
print("The percentage of the test questions that are answered either with either only correct or only incorrect answers or both of them,\nregardless if the expected answer is one of these correct answer(s) or not: {:.2f}%".format((count_test_questions_answered/count_test_questions_total)*100))


# **Question Answering engine for the answerable questions, with the best parameters for each methodology.**

# In[ ]:


#Question Answering engine
activation_relation_best_parameters_path = 'drive/MyDrive/Relation_indexes_models_best_parameters/Answerable_Questions/ELU/relation_prediction_span_encoded_ignoring_answerable_questions_ELU.pt'
activation_entity_best_parameters_path =  "drive/MyDrive/Entity_span_indexes_models_best_parameters/Answerable_Questions/Softplus/entity_span_encoded_prediction_ignoring_answerable_questions_Softplus.pt"
print('This is the question answering engine. Type exit to quit')
dropout_prob = 0.3
count_test_questions_total = 0
count_test_questions_correct_answered = 0
count_test_questions_answered = 0
while(1):
  question = input()
  if question == "exit":
    break;
  else:
    count_test_questions_total += 1
  #Find the real answer, relation id and entity id of the input question from the corresponding .txt file.
  for i in range(len(testing_questions)):
    if testing_questions[i] == question:
      real_question_answer_label = testing_questions_answer_labels[i]
      real_question_entity_id = testing_questions_entity_ids[i]
      real_question_relation_id = testing_questions_relation_ids[i]
      break
  question_correct_answered, question_answered = answer(question, Endict, entities, rel_answerable_vocab, len(rel_answerable_vocab), tokenizer, quests, similarity_model, max(q_lengths), "ELU", "Softplus", activation_relation_best_parameters_path, activation_entity_best_parameters_path, dropout_prob, SEED, real_question_answer_label, real_question_entity_id, real_question_relation_id)
  if question_correct_answered == 1:
    count_test_questions_correct_answered += 1
  if question_answered == 1:
    count_test_questions_answered += 1
print(count_test_questions_correct_answered)
print(count_test_questions_answered)
print(count_test_questions_total)
print("The percentage of the test questions that are answered correct is: {:.2f}%".format((count_test_questions_correct_answered/count_test_questions_total)*100))
print("The percentage of the test questions that are answered either with either only correct or only incorrect answers or both of them,\nregardless if the expected answer is one of these correct answer(s) or not: {:.2f}%".format((count_test_questions_answered/count_test_questions_total)*100))


# In[ ]:


#Question Answering engine
activation_relation_best_parameters_path = 'drive/MyDrive/Relation_indexes_models_best_parameters/Answerable_Questions/ELU/relation_prediction_span_encoded_not_ignoring_answerable_questions_ELU.pt'
activation_entity_best_parameters_path =  "drive/MyDrive/Entity_span_indexes_models_best_parameters/Answerable_Questions/ELU/entity_span_encoded_prediction_not_ignoring_answerable_questions_ELU.pt"
print('This is the question answering engine. Type exit to quit')
dropout_prob = 0.3
count_test_questions_total = 0
count_test_questions_correct_answered = 0
count_test_questions_answered = 0
while(1):
  question = input()
  if question == "exit":
    break;
  else:
    count_test_questions_total += 1
  #Find the real answer, relation id and entity id of the input question from the corresponding .txt file.
  for i in range(len(testing_questions)):
    if testing_questions[i] == question:
      real_question_answer_label = testing_questions_answer_labels[i]
      real_question_entity_id = testing_questions_entity_ids[i]
      real_question_relation_id = testing_questions_relation_ids[i]
      break
  question_correct_answered, question_answered = answer(question, Endict, entities, rel_answerable_vocab, len(rel_answerable_vocab), tokenizer, quests, similarity_model, max(q_lengths), "ELU", "ELU", activation_relation_best_parameters_path, activation_entity_best_parameters_path, dropout_prob, SEED, real_question_answer_label, real_question_entity_id, real_question_relation_id)
  if question_correct_answered == 1:
    count_test_questions_correct_answered += 1
  if question_answered == 1:
    count_test_questions_answered += 1
print(count_test_questions_correct_answered)
print(count_test_questions_answered)
print(count_test_questions_total)
print("The percentage of the test questions that are answered correct is: {:.2f}%".format((count_test_questions_correct_answered/count_test_questions_total)*100))
print("The percentage of the test questions that are answered either with either only correct or only incorrect answers or both of them,\nregardless if the expected answer is one of these correct answer(s) or not: {:.2f}%".format((count_test_questions_answered/count_test_questions_total)*100))


# In[ ]:


#Question Answering engine
activation_relation_best_parameters_path = 'drive/MyDrive/Relation_indexes_models_best_parameters/Answerable_Questions/ELU/relation_prediction_cosine_similarity_answerable_questions_ELU.pt'
activation_entity_best_parameters_path =  "drive/MyDrive/Entity_span_indexes_models_best_parameters/Answerable_Questions/Softplus/entity_span_prediction_cosine_similarity_answerable_questions_Softplus.pt"
print('This is the question answering engine. Type exit to quit')
dropout_prob = 0.3
count_test_questions_total = 0
count_test_questions_correct_answered = 0
count_test_questions_answered = 0
while(1):
  question = input()
  if question == "exit":
    break;
  else:
    count_test_questions_total += 1
  #Find the real answer, relation id and entity id of the input question from the corresponding .txt file.
  for i in range(len(testing_questions)):
    if testing_questions[i] == question:
      real_question_answer_label = testing_questions_answer_labels[i]
      real_question_entity_id = testing_questions_entity_ids[i]
      real_question_relation_id = testing_questions_relation_ids[i]
      break
  question_correct_answered, question_answered = answer(question, Endict, entities, rel_answerable_vocab, len(rel_answerable_vocab), tokenizer, quests, similarity_model, max(q_lengths), "ELU", "Softplus", activation_relation_best_parameters_path, activation_entity_best_parameters_path, dropout_prob, SEED, real_question_answer_label, real_question_entity_id, real_question_relation_id)
  if question_correct_answered == 1:
    count_test_questions_correct_answered += 1
  if question_answered == 1:
    count_test_questions_answered += 1
print(count_test_questions_correct_answered)
print(count_test_questions_answered)
print(count_test_questions_total)
print("The percentage of the test questions that are answered correct is: {:.2f}%".format((count_test_questions_correct_answered/count_test_questions_total)*100))
print("The percentage of the test questions that are answered either with either only correct or only incorrect answers or both of them,\nregardless if the expected answer is one of these correct answer(s) or not: {:.2f}%".format((count_test_questions_answered/count_test_questions_total)*100))


# In[ ]:


#Question Answering engine
activation_relation_best_parameters_path = 'drive/MyDrive/Relation_indexes_models_best_parameters/Answerable_Questions/ELU/relation_prediction_jaccard_similarity_answerable_questions_ELU.pt'
activation_entity_best_parameters_path =  "drive/MyDrive/Entity_span_indexes_models_best_parameters/Answerable_Questions/Softplus/entity_span_prediction_jaccard_similarity_answerable_questions_Softplus.pt"
print('This is the question answering engine. Type exit to quit')
dropout_prob = 0.3
count_test_questions_total = 0
count_test_questions_correct_answered = 0
count_test_questions_answered = 0
while(1):
  question = input()
  if question == "exit":
    break;
  else:
    count_test_questions_total += 1
  #Find the real answer, relation id and entity id of the input question from the corresponding .txt file.
  for i in range(len(testing_questions)):
    if testing_questions[i] == question:
      real_question_answer_label = testing_questions_answer_labels[i]
      real_question_entity_id = testing_questions_entity_ids[i]
      real_question_relation_id = testing_questions_relation_ids[i]
      break
  question_correct_answered, question_answered = answer(question, Endict, entities, rel_answerable_vocab, len(rel_answerable_vocab), tokenizer, quests, similarity_model, max(q_lengths), "ELU", "Softplus", activation_relation_best_parameters_path, activation_entity_best_parameters_path, dropout_prob, SEED, real_question_answer_label, real_question_entity_id, real_question_relation_id)
  if question_correct_answered == 1:
    count_test_questions_correct_answered += 1
  if question_answered == 1:
    count_test_questions_answered += 1
print(count_test_questions_correct_answered)
print(count_test_questions_answered)
print(count_test_questions_total)
print("The percentage of the test questions that are answered correct is: {:.2f}%".format((count_test_questions_correct_answered/count_test_questions_total)*100))
print("The percentage of the test questions that are answered either with either only correct or only incorrect answers or both of them,\nregardless if the expected answer is one of these correct answer(s) or not: {:.2f}%".format((count_test_questions_answered/count_test_questions_total)*100))

