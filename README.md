# Question-Answering-Engine-with-Transformers
The aim of this thesis is the creation of a question answering engine over the Wikidata knowledge graph. 
This thesis is based on the entity span and relation prediction, which are performed on questions posed in natural English. The models for these predictions are implemented through the Bidirectional Encoder Representations from Transformers (BERT) model.

More specifically the main parts of the implementation of this thesis are:


   1.The creation of a query generator that creates a SPARQL query equivalent to the user's question.
   
   2.The conctruction of some modules for preprocessing questions and converting their content (e.g custom dataset) into token IDs and attention masks.
   
   3.The creation of the BERT-based relation and span entity prediction models through methodologies such as the creation of lists consisting of 0 and 1, the combination of Sentence Transformers library with 
   either the cosine similarity metric or the jaccard similarity metric.
   
   4.The creation of a query executor that retrieves the answer in natural language from the Wikidata endpoint.
   
   5.The development of the Question answering engines based on the previous four parts by using cosine similarity metric, jaccard similarity metric and Sentence Transformers Library.

All the above components leads to a very effective question-answering engine, because the percentage of the correctly answered questions is significantly high.

# Notebooks

All the notebooks that are used for this thesis are categorized to the following files:

[**Dataset Preprocessing**](https://github.com/DimOriCoding/Question-Answering-Engine-with-Transformers/tree/main/Dataset%20Preprocessing)

This file includes notebooks that find the entity labels through SPARQL queries for all the SimpleQuestions datasets (training, validation, test), where these entity labels are added to all the SimpleQuestions datasets. Also the notebooks of this file the preprocessing of the questions is made, which included removing accents, special characters and possessive suffixes to improve entity identification.

[**Sentence Transformers + Cosine Similarity Metric**](https://github.com/DimOriCoding/Question-Answering-Engine-with-Transformers/tree/main/Sentence%20Transformers%20%2B%20Cosine%20Similarity%20Metric%20Methodology)

This file includes notebooks that the methodology of finding entity span indexes (start, end) by using either Sentence Transformers library combined with cosine simliarity metric or the command difflib.get_close_matches() from spacy library. Also these notebooks contain the implementation of separate span entity and relation models. Both showed excellent performance, with the entity span prediction model achieving 91-94% evaluation scores (F1-score, accuracy, precision, recall) and the relation prediction model 94-96% accuracy.

[**Sentence Transformers + Jaccard Similarity Metric**](https://github.com/DimOriCoding/Question-Answering-Engine-with-Transformers/tree/main/Sentence%20Transformers%20%2B%20Jaccard%20Similarity%20Metric%20Methodology)

This file includes notebooks that the methodology of finding entity span indexes (start, end) by using either Sentence Transformers library combined with jaccard simliarity metric or the command difflib.get_close_matches() from spacy library is applied. Also these notebooks contain the implementation of separate span entity and relation models, which proved to be effective in addressing the limitations of the single model approach. Both showed excellent performance, with the entity span prediction model achieving 91-94% evaluation scores (F1-score, accuracy, precision, recall) and the relation prediction model 94-96% accuracy.


[**Entity span as an encoded list with 0 and 1**] 

Another methodology that is applied is the creation of lists constisting of 0 and 1, if the entity label is part of the wording of the question else only 0. 
These lists are assuming as the entity spans for the entity of whichever question. Especially there are notebooks that demonstrate this methodology only for the questions that their entity span is consisting of 0 and 1, training and validating either all the SimpleQuestions dataset or the corresponding SimpleQuestions dataset including only Questions that are answerable over wikidata ([**](https://github.com/DimOriCoding/Question-Answering-Engine-with-Transformers/blob/main/Thesis_Entity_span_creation_by_encoding_with_0_and_1_(for_answerable_questions)_Not_ignoring_Questions_with_no_similarity_or_equality_with_the_entity_label.ipynb). Additionally there are notebooks that showcase this methodology for all the questions regardless if their entity span consisting from 0 and 1 or only 0 and if the questions that(). Also these notebooks contain the implementation of the same separate span entity and relation models such as in methodologies with sentence Transformers similarity model and either cosine or jaccard similarity metric. Both showed excellent performance, with the entity span prediction model achieving 91-97% evaluation scores (F1-score, accuracy, precision, recall) and the relation prediction model 94-96% accuracy.

[**Question Answering Engines**](https://github.com/DimOriCoding/Question-Answering-Engine-with-Transformers/tree/main/Question%20Answering%20Engines)

The notebooks that are included in this file display the development of the Question Answering Engine modules. Firstly the entity label and also the relation id prediction is made. Secondly an SPARQL query is constructed to obtain the ID of the entity label. If the entity id is not found, i.e the query is unsuccesful, the QA engine system searches for the closest match either through Sentence Transformers similarity model or a similarity metric such as cosine and jaccard. Finally the relation id and the entity id that are occurred from the corresponding predictions are combined through another SPARQL query in order to find the final answer if this exists. Essentially in this file you can see and explore all the examples of the Question Answering Engines (Sentence Transformers library, cosine similarity metric, jaccard similarity metric).

Also in this repository there is a file named [**BSc_Thesis.pdf**](https://github.com/DimOriCoding/Question-Answering-Engine-with-Transformers/blob/main/BSc_Thesis.pdf) that describes analytically all the work that is made for this thesis as well as some very important introductory concepts for this thesis (SPARQL, Semantic Parsing, RDF data, Question answering over knowledge graphs etc).

