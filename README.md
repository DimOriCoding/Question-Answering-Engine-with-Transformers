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

This file includes notebooks that the methodology of finding entity span indexes (start, end) by using either Sentence Transformers library combined with cosine simliarity metric or the command difflib.get_close_matches() from spacy library. Also these notebooks contain the implementation of separate span entity and relation models, which proved to be effective in addressing the limitations of the single model approach. Both showed excellent performance, with the entity span prediction model achieving 91-94% evaluation scores (F1-score, accuracy, precision, recall) and the relation prediction model 94-96% accuracy.

[**Sentence Transformers + Jaccard Similarity Metric**](https://github.com/DimOriCoding/Question-Answering-Engine-with-Transformers/tree/main/Sentence%20Transformers%20%2B%20Jaccard%20Similarity%20Metric%20Methodology)

This file includes notebooks that the methodology of finding entity span indexes (start, end) by using either Sentence Transformers library combined with jaccard simliarity metric or the command difflib.get_close_matches() from spacy library. Also these notebooks contain the implementation of separate span entity and relation models, which proved to be effective in addressing the limitations of the single model approach. Both showed excellent performance, with the entity span prediction model achieving 91-94% evaluation scores (F1-score, accuracy, precision, recall) and the relation prediction model 94-96% accuracy.

