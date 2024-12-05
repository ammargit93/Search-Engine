from django.shortcuts import render,redirect
from django.views.decorators.csrf import csrf_exempt
from groq import Groq
from dotenv import load_dotenv
import markdown
import os
import re 

load_dotenv()


def prettify(response):
    if '\n' in response:
        return response.replace("\n", "<br><br>")
    return response

def model(prompt):
    client = Groq(api_key=os.getenv('GROQ_API_KEY'))
    completion = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user","content": f'{prompt}'}],
    )
    print(completion.choices[0].message.content)
    return completion.choices[0].message.content


@csrf_exempt
def home(request):
    if request.method == 'POST':
        prompt = request.POST.get("prompt")
        print(prompt)
        response = model(prompt=prompt)
        return render(request, 'home.html', {'ai_response':prettify(response),'prompt':prompt})
    return render(request, 'home.html')