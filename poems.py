# 对于给定的一首唐诗，分析该唐诗的作者是白居易还是王维？
import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# clean data
file_path = './tang.csv'
df = pd.read_csv(file_path)
df_wang = df[df['作者']=='王维']['内容']
df_bai = df[df['作者']=='白居易']['内容']

#set attributes and labels
poems_bai = df_bai.values.tolist()
poems_wang = df_wang.values.tolist()

# balance 
min_len_poems = min(len(poems_bai), len(poems_wang))
poems_bai = poems_bai[:min_len_poems]
poems_wang = poems_wang[:min_len_poems]

labels = [0] * min_len_poems + [1] * min_len_poems
poems = poems_bai + poems_wang

# reshape data
def reshape_poems(poems, vect = None):
    poems_cut = [" ".join(jieba.lcut(poem)) for poem in poems]
    print(poems_cut)
    if vect is None:
        vect = CountVectorizer()
        x = vect.fit_transform(poems_cut)
    else:
        x = vect.transform(poems_cut)
    x = x.toarray()
    return x, vect

x, vect = reshape_poems(poems)

# split data
x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=0.2, shuffle=True)

# Multi-layer Perceptron hidden = 100, relu
mlp = MLPClassifier(activation='tanh', hidden_layer_sizes=[100,50,25])
mlp.fit(x_train, y_train)
y_pred = mlp.predict(x_test)

# calculate accuracy
score = accuracy_score(y_test, y_pred)
score_f1 = f1_score(y_test, y_pred)
y_probs = mlp.predict_proba(x_test)[:, 1]
score_auc = roc_auc_score(y_test, y_probs)

print('------------------------------\nAccuracy: ', round(score,3), '\nF1: ', round(score_f1,3), '\nAUC: ', round(score_auc,3), '\n------------------------------')


# reshape input and predict
def predict_author(poem, vect):
    poem_cut, _ = reshape_poems([poem], vect)
    prediction = mlp.predict(poem_cut)
    return "白居易" if prediction[0] == 0 else "王维"

# loop prompt
while True:
    print("退出请输入'1'。")
    input_poem = input('请输入白居易或王维的诗词：')
    if input_poem == '1':
        break
    else:
        print('诗词的作者为：', predict_author(input_poem,vect), '\n------------------------------')

print('结束。')
