import cgi
import cgitb
print('content-type:text/html\r\n\r\n')
form=cgi.FieldStorage()
sample=str(form.getvalue("sampletext"))
#INSTALLING AND IMPORTING NLTK
import nltk

#DOWNLOADING AND IMPORTING TWITTER_SAMPLES AND CREATING VARIABLES
from nltk.corpus import twitter_samples
positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')
text = twitter_samples.strings('tweets.20150430-223406.json')

#TOKENIZING
tweet_tokens = twitter_samples.tokenized('positive_tweets.json')

#NORMALIZING VIA LEMMATIZATION
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer

#FUNCTION FOR LEMMATIZATION<DONT USE>def lemmatize_sentence(tokens):    AS THIS IS CARRIED OUT IN THE "NOISE REMOVAL" STEP
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentence = []
    for word, tag in pos_tag(tokens):
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatized_sentence.append(lemmatizer.lemmatize(word, pos))
    return lemmatized_sentence

#NOISE REMOVAL
import re, string

def remove_noise(tweet_tokens, stop_words = ()):

    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens

#GENERATING CLEAN TOKENS LIST
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')

positive_cleaned_tokens_list = []
negative_cleaned_tokens_list = []

for tokens in positive_tweet_tokens:
    positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

for tokens in negative_tweet_tokens:
    negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))



#GENERATOR FUNCTION FOR PROVIDING LIST OF WORDS IN ALL TWEET-TOKENS JOINED FOR WORD FREQUENCY DETERMINATION
def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token

all_pos_words = get_all_words(positive_cleaned_tokens_list)

#FREQUENCY DETERMINATION
from nltk import FreqDist

freq_dist_pos = FreqDist(all_pos_words)

#PREPARING DATA FOR ML ALGO:

#CONVERTING TOKENS INTO DICTIONARIES
def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)

positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)

#SPLITTING DATASET FOR TRAINING AND TESTING THE MODEL
import random

positive_dataset = [(tweet_dict, "Positive")
                     for tweet_dict in positive_tokens_for_model]

negative_dataset = [(tweet_dict, "Negative")
                     for tweet_dict in negative_tokens_for_model]

dataset = positive_dataset + negative_dataset

random.shuffle(dataset)

train_data = dataset[:7000]
test_data = dataset[7000:]


#BUILDING THE MODEL AND TESTING
from nltk import classify
from nltk import NaiveBayesClassifier
classifier = NaiveBayesClassifier.train(train_data)

print("Accuracy is:", classify.accuracy(classifier, test_data))

print(classifier.show_most_informative_features(10))

#CUSTOM INPUT



from nltk.tokenize import word_tokenize

#custom_tweet = "I ordered just once from TerribleCo, they screwed up, never used the app again."   #this is where we have to give the data from web application

custom_tokens = remove_noise(word_tokenize(sample))
fi=classifier.classify(dict([token, True] for token in custom_tokens))
print('<html>')
print('<body>')
print('<h1> THAT WAS A {} COMMENT - THANKS FOR BEING KIND! THE UNIVERSE IS GRATEFUL!</h1>'.format(fi))
print('</body>')
print('</html>')
#print(classifier.classify(dict([token, True] for token in custom_tokens)))  #this is where we will write the code to send the output back to html page
