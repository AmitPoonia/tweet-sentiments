Twitter Sentiment Analysis


There are three separate scripts:

1) word2vec_sentiment,py uses a pre-trained word2vec model 

2) lsa_sentiment.py uses simple bag-of-words model with LSA

3) joint_sentiments.py combines both 1 and 2


Notes:

	- Removing stop-words didn't improve results much

	- Combining two models improved results slightly but
		nothing radical

	- The results are little skewed since no. of samples 
		for all three classes are not equal.



Additional Requirements:


The pre-trained word2vec model is available to download at below link with relevant 
literature, and once it downloaded it must be placed in 'models/'' sub-directory.

	http://www.fredericgodin.com/software/
