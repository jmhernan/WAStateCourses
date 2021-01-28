# Vocablary coverage from pretrained models
# Remove non english words
# Clean up characters and non-word items
import re 
import nltk
nltk.download('words')
words = set(nltk.corpus.words.words())

import gensim
import gensim.downloader as gensim_api

# Remove non-english using NLTK

test = '''
Note by the SecretariatandRegulations to give effect to Article 102of the 
Charter of the United Nations,adopted by the General Assembly on 
14 December Traitds ou accords internationaux transmis par un 
Membrtde 'Organisation des Nations Unies et conclus avant la date.
'''

test = re.findall('[A-Z][^A-Z]*', test)
test = ' '.join(test)
test = " ".join(w for w in nltk.wordpunct_tokenize(test) if w.lower() in words or not w.isalpha())
print(test)
### Try Google model 
gl_embed = gensim_api.load("glove-wiki-gigaword-300")

test_2 = '''
Note by the SecretariatandRegulations to give effect to Article 102of the 
Charter of the United Nations,adopted by the General Assembly on 
14 December Traitds ou accords internationaux transmis par un 
Membrtde 'Organisation des Nations Unies et conclus avant la date.
'''

test_2 = re.findall('[A-Z][^A-Z]*', test_2)
test_2 = ''.join(test_2)
test_2 = " ".join(w for w in nltk.wordpunct_tokenize(test_2) if w.lower() in gl_embed.wv.vocab or not w.isalpha())

print(test)
print(test_2) # picking up some french...intresting