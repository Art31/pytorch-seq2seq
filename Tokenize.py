import spacy
import re

class tokenize(object):
    
    def __init__(self, lang):
        self.nlp = spacy.load(lang)

        """
        Script used for cleaning corpus in order to train word embeddings.
        All emails are mapped to a EMAIL token.
        All numbers are mapped to 0 token.
        All urls are mapped to URL token.
        Different quotes are standardized.
        Different hiphen are standardized.
        HTML strings are removed.
        All text between brackets are removed.
        All sentences shorter than 5 tokens were removed.
        ...
        """
        # Punctuation list
        punctuations = re.escape('!"#%\'()*+,./:;<=>?@[\\]^_`{|}~')
        # ##### #
        # Regex #
        # ##### #
        self.re_remove_brackets = re.compile(r'\{.*\}')
        self.re_remove_html = re.compile(r'<(\/|\\)?.+?>', re.UNICODE)
        self.re_transform_numbers = re.compile(r'\d', re.UNICODE)
        self.re_transform_emails = re.compile(r'[^\s]+@[^\s]+', re.UNICODE)
        self.re_transform_url = re.compile(r'(http|https)://[^\s]+', re.UNICODE)
        # Different quotes are used.
        self.re_quotes_1 = re.compile(r"(?u)(^|\W)[‘’′`']", re.UNICODE) # https://docs.python.org/2/library/re.html#re.U
        self.re_quotes_2 = re.compile(r"(?u)[‘’`′'](\W|$)", re.UNICODE)
        self.re_quotes_3 = re.compile(r'(?u)[‘’`′“”]', re.UNICODE)
        self.re_dots = re.compile(r'(?<!\.)\.\.(?!\.)', re.UNICODE)
        self.re_punctuation = re.compile(r'([,";:]){2},', re.UNICODE)
        self.re_hiphen = re.compile(r' -(?=[^\W\d_])', re.UNICODE)
        self.re_tree_dots = re.compile(u'…', re.UNICODE)
        # Differents punctuation patterns are used.
        self.re_punkts = re.compile(r'(\w+)([%s])([ %s])' %
                            (punctuations, punctuations), re.UNICODE)
        self.re_punkts_b = re.compile(r'([ %s])([%s])(\w+)' %
                                (punctuations, punctuations), re.UNICODE)
        self.re_punkts_c = re.compile(r'(\w+)([%s])$' % (punctuations), re.UNICODE)
        self.re_changehyphen = re.compile(u'–')
        self.re_doublequotes_1 = re.compile(r'(\"\“\”)')
        self.re_doublequotes_2 = re.compile(r'(\'\‘\’)')
        self.re_trim = re.compile(r' +', re.UNICODE)

            
    # OLD TOKENIZER
    # def tokenizer(self, sentence):
    #     sentence = re.sub(
    #     r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", str(sentence))
    #     sentence = re.sub(r"[ ]+", " ", sentence)
    #     sentence = re.sub(r"\!+", "!", sentence)
    #     sentence = re.sub(r"\,+", ",", sentence)
    #     sentence = re.sub(r"\?+", "?", sentence)
    #     sentence = sentence.lower()
    #     return [tok.text for tok in self.nlp.tokenizer(sentence) if tok.text != " "]

    # NEW TOKENIZER
    def tokenizer(self, sentence):
        sentence = re.sub(r"[ ]+", " ", sentence)
        sentence = sentence.lower()
        sentence = sentence.replace('\xa0', ' ')
        sentence = self.re_tree_dots.sub('...', sentence)
        sentence = re.sub('\.\.\.', '', sentence)
        sentence = self.re_remove_brackets.sub('', sentence)
        sentence = self.re_changehyphen.sub('-', sentence)
        sentence = self.re_remove_html.sub(' ', sentence)
        sentence = self.re_transform_numbers.sub('0', sentence)
        sentence = self.re_transform_url.sub('URL', sentence)
        sentence = self.re_transform_emails.sub('EMAIL', sentence)
        sentence = self.re_quotes_1.sub(r'\1"', sentence)
        sentence = self.re_quotes_2.sub(r'"\1', sentence)
        sentence = self.re_quotes_3.sub('"', sentence)
        sentence = re.sub('"', '', sentence)
        sentence = self.re_dots.sub('.', sentence)
        sentence = self.re_punctuation.sub(r'\1', sentence)
        sentence = self.re_hiphen.sub(' - ', sentence)
        sentence = self.re_punkts.sub(r'\1 \2 \3', sentence)
        sentence = self.re_punkts_b.sub(r'\1 \2 \3', sentence)
        sentence = self.re_punkts_c.sub(r'\1 \2', sentence)
        sentence = self.re_doublequotes_1.sub('\"', sentence)
        sentence = self.re_doublequotes_2.sub('\'', sentence)
        sentence = self.re_trim.sub(' ', sentence)
        # return text.strip()
        return [tok.text for tok in self.nlp.tokenizer(sentence) if tok.text != " "]