from nltk.corpus import wordnet as wn
import spacy
from lxml import html
import requests
from nltk.stem import WordNetLemmatizer

class DefaultBasic:
    def __init__(self, method = 'macmillan'):
        assert method in ['wordnet', 'macmillan'], f"method {method} not supported."
        self.method = method
        self.nlp = spacy.load('en_core_web_sm', disable=["parser", "ner"])
        if method == 'macmillan':
            self.lemmatizer = WordNetLemmatizer()

    def _wordnet(self, word):
        syn = wn.synsets(word)[0]
        lemmas = syn.lemmas()
        examples = syn.examples()
        if not examples: # Means there is no example sentence in wordnet
            return None
        doc = self.nlp(examples[0])
        for lemma in lemmas:
            for idx, t in enumerate(doc):
                if lemma.name() == t.lemma_:
                    index = idx
                    sent = [token.text for token in doc]
                    break
            else:
                continue
            break
        return sent, index
    
    def _macmillan(self, word):
        word = self.lemmatizer.lemmatize(word)
        headers = {'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.4 Safari/605.1.15'}
        url = f'https://www.macmillandictionary.com/dictionary/british/{word}'
        res = requests.get(url, headers = headers)
        tree = html.fromstring(res.content)
        try:
            sent = tree.find('.//div[@class="SENSE-CONTENT"]').find('.//div[@class="EXAMPLES anchor first"]').xpath('string()')
        except:
            return None
        doc = self.nlp(str(sent))
        for idx, t in enumerate(doc):
            if word == t.lemma_:
                index = idx
                sent = [token.text for token in doc]
                break
        else:
            return None
        return sent, index
    
    def __call__(self, word):
        if self.method == 'wordnet':
            return self._wordnet(word)
        elif self.method == 'macmillan':
            return self._macmillan(word)


if __name__ == '__main__':
    basicer = DefaultBasic()
    print(basicer('break'))