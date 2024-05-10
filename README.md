# Question-Answering-Engine-with-Transformers
The aim of this thesis is the creation of a question answering engine over the Wikidata knowledge graph. 
This thesis is based on the entity span and relation prediction, which are performed on questions posed in natural English. The models for these predictions are implemented through the Bidirectional Encoder Representations from Transformers (BERT) model.

More specifically the main parts of the implementation of this thesis are:
   The creation of a query generator that creates a SPARQL query equivalent to the user's question.
   The conctruction of some modules for preprocessing questions and converting their content (e.g custom dataset) into token IDs and attention masks.
   The creation of the BERT-based relation and span entity prediction models through methodologies such as the creation of lists consisting of 0 and 1, the combination of Sentence Transformers library with either 
   the cosine similarity metric or the jaccard similarity metric.
   The creation of a query executor that retrieves the answer in natural language from the Wikidata endpoint.
   The development of the Question answering engines based on the previous four parts by using cosine similarity metric, jaccard similarity metric and Sentence Transformers Library.

All the above components leads to a very effective question-answering engine, because the percentage of the correctly answered questions is significantly high.
