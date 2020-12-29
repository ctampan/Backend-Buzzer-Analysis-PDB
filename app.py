from flask import Flask, request, jsonify
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk import word_tokenize
from pyspark import SparkContext
from pyspark.sql import SparkSession
from flask import g
import subprocess
import json

import nltk
import re
import numpy as np
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

HDFS_URL = "hdfs://namenode.pdb:9000"

HDFS_PATH = '/pdb/'


def normalisasi(tweet):
    normal_tw = str(tweet).lower()  # lowercase
    normal_tw = re.sub('\s+', ' ', normal_tw)  # remove extra space
    normal_tw = normal_tw.strip()  # trim depan belakang
    normal_tw = re.sub(r'[^\w\s]', ' ', normal_tw)  # buang punctuation
    # regex huruf yang berulang kaya haiiii (untuk fitur unigram)
    normal_regex = re.compile(r"(.)\1{1,}")
    # buang huruf yang berulang
    normal_tw = normal_regex.sub(r"\1\1", normal_tw)
    return normal_tw


def remove_stopwords(x):
    id_stopwords = stopwords.words('indonesian')
    en_stopwords = stopwords.words('english')
    nosw_news = []
    for t in x:
        if (t not in id_stopwords) and (t not in en_stopwords):
            nosw_news.append(t)
    return nosw_news


def generate_response(df, pred):
    data = {}

    jumlah_bot = 0
    jumlah_user = 0
    barchart = {}
    user = set()

    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    for mo in months:
        barchart[mo + ' 2020'] = {'bot': 0, 'user': 0}

    barchart['Jan 2021'] = {'bot': 0, 'user': 0}

    latest_tweet_bot = []
    latest_tweet_human = []

    for i in range(len(pred)):

        timebar = df['created_at'][i].split(' ')
        if int(timebar[-1]) != 2020 :
            continue

        user.update(df['Username'])

        if(pred[i] == 'person'):

            if len(latest_tweet_human) < 5:
                lt = {'name': df['Name'][i], 'tweet': df['Tweet Content'][i], 'profile_image_url':df['profile_image_url'][i], 'timestamp_ms':df['timestamp_ms'][i], 'created_at':df['created_at'][i], 'username':df['Username'][i]}
                latest_tweet_human.append(lt)
                latest_tweet_human = sorted(latest_tweet_human, key=lambda x: x['timestamp_ms'], reverse=True)
            else:
                if latest_tweet_human[-1]['timestamp_ms'] < df['timestamp_ms'][i]:
                    latest_tweet_human.pop(-1)
                    lt = {'name': df['Name'][i], 'tweet': df['Tweet Content'][i], 'profile_image_url':df['profile_image_url'][i], 'timestamp_ms':df['timestamp_ms'][i], 'created_at':df['created_at'][i], 'username':df['Username'][i]}
                    latest_tweet_human.append(lt)
                    latest_tweet_human = sorted(latest_tweet_human, key=lambda x: x['timestamp_ms'], reverse=True)

            jumlah_user += 1
            barchart[timebar[1] + ' ' + timebar[-1]]['user'] += 1
        else:

            if len(latest_tweet_bot) < 5:
                lt = {'name': df['Name'][i], 'tweet': df['Tweet Content'][i], 'profile_image_url':df['profile_image_url'][i], 'timestamp_ms':df['timestamp_ms'][i], 'created_at':df['created_at'][i], 'username':df['Username'][i]}
                latest_tweet_bot.append(lt)
                latest_tweet_bot = sorted(latest_tweet_bot, key=lambda x: x['timestamp_ms'], reverse=True)
            else:
                if latest_tweet_bot[-1]['timestamp_ms'] < df['timestamp_ms'][i]:
                    latest_tweet_bot.pop(-1)
                    lt = {'name': df['Name'][i], 'tweet': df['Tweet Content'][i], 'profile_image_url':df['profile_image_url'][i], 'timestamp_ms':df['timestamp_ms'][i], 'created_at':df['created_at'][i], 'username':df['Username'][i]}
                    latest_tweet_bot.append(lt)
                    latest_tweet_bot = sorted(latest_tweet_bot, key=lambda x: x['timestamp_ms'], reverse=True)

            jumlah_bot += 1
            barchart[timebar[1] + ' ' + timebar[-1]]['bot'] += 1

    data['pie_chart'] = {'jumlah_bot': jumlah_bot, 'jumlah_user': jumlah_user}
    data['summary'] = {'total_tweet': len(pred), 'total_user': len(user)}

    m_b = ['Jan 2020', 'Feb 2020', 'Mar 2020', 'Apr 2020', 'May 2020', 'Jun 2020', 'Jul 2020', 'Aug 2020', 'Sep 2020', 'Oct 2020', 'Nov 2020', 'Dec 2020', 'Jan 2021']
    barchart_list = []

    for m in m_b:
        temp = barchart
        temp['month'] = m
        barchart_list.append(temp)

    data['barchart'] = barchart_list
    data['latest_tweet_bot'] = latest_tweet_bot
    data['latest_tweet_human'] = latest_tweet_human
    response = jsonify(message=data)
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


def convert_json_to_df(tw_json_list):
    column_names = ['created_at', 'Tweet Content',
                    'Name', 'Username', 'User Bio', 'timestamp_ms', 'profile_image_url']

    df = pd.DataFrame(columns=column_names)

    for tw in tw_json_list:

        df = df.append({'created_at': tw['created_at'], 'Tweet Content': tw['text'], 'Name': tw['user']['name'],
                        'Username': tw['user']['screen_name'], 'User Bio': tw['user']['description'], 'timestamp_ms': int(tw['timestamp_ms']), 'profile_image_url':tw['user']['profile_image_url']}, ignore_index=True)

    return df


def brew_data(tw_json_list):

    tweet_data = convert_json_to_df(tw_json_list)
    tweet_data['User Bio'] = tweet_data['User Bio'].fillna('nan')

    tweet_data['User Bio'] = tweet_data['User Bio'].apply(normalisasi)
    tweet_data['User Bio'] = tweet_data['User Bio'].apply(
        lambda x: word_tokenize(x.lower()))
    tweet_data['User Bio'] = tweet_data['User Bio'].apply(
        lambda x: [token for token in x if token.isalnum()])
    tweet_data['User Bio'] = tweet_data['User Bio'].apply(remove_stopwords)
    tweet_data['User Bio'] = tweet_data['User Bio'].apply(
        lambda x: ' '.join(x))

    pkl_filename = "transform.pkl"
    with open(pkl_filename, 'rb') as file:
        transform_model = pickle.load(file)

    unigram = transform_model.transform(tweet_data['User Bio']).todense().A

    pkl_filename = "model_multinomialNB.pkl"
    with open(pkl_filename, 'rb') as file:
        predict_model = pickle.load(file)

    predict_result = predict_model.predict(unigram)

    return generate_response(tweet_data, predict_result)


@app.route('/api', methods=['GET'])
def test():
    sc = g._sc = SparkContext.getOrCreate()
    spark = SparkSession(sc)

    URI = sc._gateway.jvm.java.net.URI
    Path = sc._gateway.jvm.org.apache.hadoop.fs.Path
    FileSystem = sc._gateway.jvm.org.apache.hadoop.fs.FileSystem
    Configuration = sc._gateway.jvm.org.apache.hadoop.conf.Configuration

    fs = FileSystem.get(URI(HDFS_URL), Configuration())

    status = fs.listStatus(Path(HDFS_PATH))

    tweet_list = []

    ctr = 0

    for fileStatus in status:

        # if ctr != 7:
        #     ctr += 1
        #     continue

        tweet = sc.textFile(fileStatus.getPath().toString())

        for tw_avr in tweet.collect():
            if not ("avro.schema" in tw_avr):
                try:
                    tw_json = json.loads(tw_avr)
                    tweet_list.append(tw_json)
                except:
                    pass

        # return str(tweet.take(100)[0])

    return brew_data(tweet_list)


@app.teardown_appcontext
def teardown_sparkcontext(exception):
    sc = getattr(g, '_sc', None)
    if sc is not None:
        sc.stop()


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')