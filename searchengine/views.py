from sklearn.feature_extraction.text import TfidfVectorizer
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render,redirect
from pymongo import MongoClient
from dotenv import load_dotenv
from groq import Groq
import requests
import string
import nltk
import os

client = MongoClient("mongodb://localhost:27017/")
db = client["AI-Search"]
query_collection = db["query"]

load_dotenv()

def prettify(response):
    if '\n' in response:
        return response.replace("\n", "<br>")
    return response

def model(prompt):
    client = Groq(api_key=os.getenv('GROQ_API_KEY'))
    completion = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user","content": f'{prompt}'}],
    )
    return completion.choices[0].message.content


def label_tokens_with_importance(sentence):
   
    vectorizer = TfidfVectorizer(tokenizer=nltk.word_tokenize, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([sentence])  
    feature_names = vectorizer.get_feature_names_out()

    tfidf_scores = tfidf_matrix.toarray()[0]
    token_scores = {}

    res = []
    for token in nltk.word_tokenize(sentence):
        token_lower = token.lower()
        if token_lower in string.punctuation:
            continue
        if token_lower in feature_names:
            token_scores[token] = tfidf_scores[feature_names.tolist().index(token_lower)]
            res.append((token,float(tfidf_scores[feature_names.tolist().index(token_lower)])))
        else:
            token_scores[token] = 0.0 
            

    sorted_tokens = sorted(res, key=lambda x: x[1], reverse=True)
    return sorted_tokens


@csrf_exempt
def home(request):
    if request.method == 'POST':
        prompt = request.POST.get("prompt")
        response = model(prompt=prompt)
        imgarr = []
        if 'image' in prompt:
            prompt1 = prompt.replace('image', '').strip()
            for tkn,score in label_tokens_with_importance(prompt1):
                print(tkn, score)
                img = requests.get(f'https://api.openverse.org/v1/images/?q={tkn}&page_size=10&page=1').json()
                for photo in img['results']:
                    imgarr.append(photo['url'])
        print(imgarr)
                
        query_collection.insert_one({'prompt': prompt, 'response': prettify(response), 'images':imgarr})

        return render(request, 'home.html', {
            'history': list(query_collection.find({})),
            'images':imgarr
        })
    return render(request, 'home.html', {
        'history': list(query_collection.find({})),
    })
