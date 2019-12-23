from janome.tokenizer import Tokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
import json

app = Flask(__name__)
CORS(app)

# コンテンツベースレコメンド
# 同じような文章の人を推薦する
# アルゴリズム：コサイン類似度


@app.route("/", methods=["POST"])
def pridect():

    # reccomend_predict = "ディープラーニング"
    reccomend_predict = request.json['reccomend']
    print(reccomend_predict)

    url = "http://localhost:8888/ew-api/public/api/accessLogHelloRenoNoteKeyWordsController"
    r = requests.get(url)
    data = json.loads(r.text)
    df = pd.DataFrame(data)

    texts = df['keywords']
    t = Tokenizer()

    def japaneseTokenize(text):
        tokens = t.tokenize(text)
        return [token.base_form for token in tokens
                if not token.part_of_speech.split(',')[0] in ['助詞', '助動詞', '記号', '接頭詞']]

    count = CountVectorizer(analyzer=japaneseTokenize)
    bag_of_words = count.fit_transform(texts)

    cosine_sim2 = cosine_similarity(bag_of_words)

    def get_index_from_hello_reno_note_id(hello_reno_note_id):
        return df[df.hello_reno_note_id == hello_reno_note_id].index.values[0]

    def get_hello_reno_note_id_from_index(index):
        return df[df.index == index]["hello_reno_note_id"].values[0]

    text_index = get_index_from_hello_reno_note_id(reccomend_predict)

    similar_texts = list(enumerate(cosine_sim2[int(text_index)]))

    sorted_similar_texts = sorted(
        similar_texts, key=lambda x: x[1], reverse=True)[1:]

    i = 0
    arr = []
    print("Top 5 texts similar to "+reccomend_predict+" are:\n")
    for element in sorted_similar_texts:
        arr.append(get_hello_reno_note_id_from_index(element[0]))
        print(get_hello_reno_note_id_from_index(element[0]))
        # print(arr)
        i = i+1
        if i > 3:
            break
    return jsonify(arr)


if __name__ == "__main__":
    app.run()
