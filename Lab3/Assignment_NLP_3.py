#!/usr/bin/env python
# coding: utf-8

# # <center>Assignment 3 : Text Processing </center>

# ## Q1: Regular Expression
# Define a function "**tokenize**" as follows: 
#    - takes a string as an input
#    - converts the string into lower case
#    - tokenizes the lower-cased string into tokens. A token is defined as follows:
#       - a token has at least 2 characters
#       - a token must start with an alphabetic letter (i.e. a-z or A-Z), 
#       - a token can have alphabetic letters, "\-" (hyphen), "." (dot), "'" (single quote), or "\_" (underscore) in the middle
#       - a token must end with an alphabetic letter (i.e. a-z or A-Z) 
#    - removes stop words from the tokens (use English stop words list from NLTK) 
#    - returns the resulting token list as the output
#    
# 

# ## Q2: Sentimeent Analysis
# 1. First define a function "**sentiment_analysis**" as follows: 
#   - takes a string, a list of positive words, and a list of negative words as inputs. Assume the lists are read from positive-words.txt and negative-words.txt outside of this function.
#   - tokenizes the string using the tokeniz function defined above 
#   - counts positive words and negative words in the tokens using the positive/negative words lists. With a list of negation words (i.e. not, no, isn't, wasn't, aren't, weren't, don't didn't, cannot, couldn't, won't, neither, nor), the final positive/negative words are defined as follows:
#     - Positive words:
#       * a positive word not preceded by a negation word 
#       * a negative word preceded by a negation word
#     - Negative words:
#       * a negative word not preceded by a negation word
#       * a positive word preceded by a negation word
#     - determined the sentiment of the string as follows:
#        - 2: number of positive words > number of negative words
#        - 1: number of positive words <= number of negative words
#   - returns the sentiment
#     
# 2. Define a function called **performance_evaluate** to evaluate the accuracy of the sentiment analysis in (1) as follows:  
#    - takes an input file ("amazon_review_300.csv"), a list of positive words, and a list of negative words as inputs. The input file has a list of reviews in the format of (label, review).
#    - reads the input file to get a list of reviews including review text and label of each review 
#    - for each review, predicts its sentiment using the function defined in 
#    - returns the accuracy as the number of correct sentiment predictions/total reviews

# ## Q3: (Bonus) Vector Space Model
# 
# 1. Define a function **find_similar_doc** as follows: 
#     - takes two inputs: a list of documents (i.e. docs), and the index of a selected document as an integer (i.e. doc_id).
#     - uses the "tokenize" function defined in Q1 to tokenize each document 
#     - generates normalized tf_idf matrix (each row is normalized) from the tokens (hint: reference to the tf_idf function defined in Section 8.5 in lecture notes) 
#     - calculates the pairwise cosine distance of documents using the generated tf_idf matrix 
#     - for the selected doc_id, finds the index of the most similar document (but not itself) by the cosine similarity score 
#     - returns the index of the most similar document and the similarity score
# 2. Test your function with "amazon_review_300.csv" and a few reviews from this file.
#    - Check the most similar review discovered for each of the selected reviews
#    - Can you use the calculated similarity score to determine if two documents are similar?  
#    - Do you think this function can successfully find similar documents? Why does it work or not work? 
#    - If it does not work, what can you do to improve the search?
#    - Write down your analysis along with some evidence or observations you have in a pdf file and submit this pdf file along with your code.

# In[10]:


import nltk
from nltk.corpus import stopwords
import csv
from scipy.spatial import distance
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
import re


# In[11]:


def tokenize(text):
    
    tokens = []
    stop_words = set(stopwords.words('english')) 
    
    text=text.lower()
    
    text= re.sub(r'\b(.+)(\s+\1\b)+', r'\1', text)
    tokens= re.findall(r'\w+\S\w+', text)
    
    tokens = [w for w in tokens if not w in stop_words] 
    
    return(tokens) 


# In[12]:


def sentiment_analysis(text, positive_words, negative_words):
    
    negations=["not", "no", "isn't", "wasn't", "aren't",                "weren't", "don't", "didn't", "cannot",                "couldn't", "won't", "neither", "nor"]
    tokens=tokenize(text)
    
    sentiment = None
    
    # add your code
#    positive_tokens=[]
#    negative_tokens=[]
    positive = 0
    negative = 0
    for idx, token in enumerate(tokens):
        if token in positive_words:
            if idx>0:
                if tokens[idx-1] not in negations:
           #         positive_tokens.append(token)
                    positive = positive + 1
                elif tokens[idx-1] in negations:
           #         negative_tokens.append(token)
                    negative = negative + 1
            else:
           #     positive_tokens.append(token)
                positive = positive + 1
                
        if token in negative_words:
            if idx>0:
                if tokens[idx-1] not in negations:
           #         negative_tokens.append(token)
                    negative = negative + 1
                elif tokens[idx-1] in negations:
           #         positive_tokens.append(token)
                    positive = positive + 1
            else:
           #     negative_tokens.append(token)
                negative = negative + 1        
                
    if positive > negative:
        sentiment = 2
    elif positive <= negative:
        sentiment = 1    
    return sentiment


def performance_evaluate(input_file, positive_words, negative_words):
    
    accuracy = None
    count = 0
    value = 0
    
    # add your code
    with open(input_file,'r') as f:
        input_file=[line.strip() for line in f]  
    for i in range(1, len(input_file)):
        value = str(sentiment_analysis(input_file[i],positive_words, negative_words))
        if value == input_file[i][0]:
            count = count + 1
    accuracy = count/len(input_file)
    
    return accuracy


# In[ ]:





# In[13]:


if __name__ == "__main__":  
    
    # Test Q1
    text="Composed of 3 CDs and quite a few songs (I haven't an exact count),           all of which are heart-rendering and impressively remarkable.           It has everything for every listener -- from fast-paced and energetic           (Dancing the Tokage or Termina Home), to slower and more haunting (Dragon God),           to purely beautifully composed (Time's Scar),           to even some fantastic vocals (Radical Dreamers).          This is one of the best videogame soundtracks out there,           and surely Mitsuda's best ever. ^_^"

    tokens=tokenize(text)
    
    print("Q1 tokens:", tokens)
    
    # Test Q2
    
    with open("./dataset/positive-words.txt",'r') as f:
        positive_words=[line.strip() for line in f]
        
    with open("./dataset/negative-words.txt",'r') as f:
        negative_words=[line.strip() for line in f]
        
    acc=performance_evaluate("./dataset/amazon_review_300.csv",                                   positive_words, negative_words)
    print("\nQ2 accuracy: {0:.2f}".format(acc))
   

    


# In[ ]:




