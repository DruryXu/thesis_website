from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from collections import defaultdict
import torch
import torch.utils.data as Data
import torch.nn as nn
import math
import pandas as pd

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = "mysql+pymysql://root:123456@localhost:3306/thesis"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
db = SQLAlchemy(app)

questionaire, genre_to_user, best_genres, input = [], defaultdict(list), [], []
corpus, item_to_genre, rec_result = defaultdict(list), defaultdict(list), []
names = []
cur_id, cur_user_id = ["useles"], ["useless"]

class rating(db.Model):
    __tablename__ = 'testapp_ratings'

    id = db.Column(db.String, primary_key = True)
    user_id = db.Column(db.String)
    anime_id = db.Column(db.String)
    rating = db.Column(db.String)

class animes(db.Model):
    __tablename__ = "testapp_animes"

    id = db.Column(db.String, primary_key=True)
    anime_id = db.Column(db.String)
    name = db.Column(db.String)
    genre = db.Column(db.String)
    type = db.Column(db.String)
    episodes = db.Column(db.String)
    members = db.Column(db.String)
    ratings = db.Column(db.String)
    weight = db.Column(db.String)

@app.route('/welcome')
def hello_world():
    tfidf()
    return render_template("welcome.html")

@app.route('/questionaire', methods = ["GET", "POST"])
def question():
    input.append(request.form.get("clicked"))
    while len(input) < 6:
        return render_template("questions.html", form = questionaire[len(input) - 1])

    choice = []
    for c in input[1:]:
        if c == "yes":
            choice.append("like")
        else:
            choice.append("don't like")

    return render_template("cold_start.html", genres = questionaire, feedback = choice)

@app.route('/recsys', methods = ["GET", "POST"])
def recsys():
    r_df = pd.read_csv("rating.csv").drop(["id"], axis = 1)
    a_df = pd.read_csv("anime.csv")
    print(input)
    aid = rec(r_df)
    for a in aid:
        names.extend([a_df[a_df.anime_id == a].name.tolist()[0]])

    print(names)
    return render_template("index.html", names = names)

@app.route('/single/<num>', methods = ["GET", "POST"])
def single(num):
    a_df = pd.read_csv("anime.csv")
    r_df = pd.read_csv("rating.csv")
    infos, info = a_df[a_df.name == names[int(num)]].values.tolist()[0], []
    print(infos)
    rate = request.form.get("rating")
    if rate:
        s = pd.Series({'id': max(r_df.id.tolist()) + 1, 'user_id': max(r_df.user_id.tolist()) + 1, 'anime_id': infos[1], 'rating': rate})
        r_df = r_df.append(s, ignore_index = True)
        # r_df.to_csv("rating.csv")
        return render_template('single2.html', info = infos, rate = rate)
    return render_template("single.html", info = infos)

@app.route('/update')
def Update():
    names = update()
    return render_template('index.html', names = names)


def rec(df):
    candidate, all_user = set(corpus.keys()), set(corpus.keys())
    liked_genres = []
    for idx, ans in enumerate(input[1:]):
        if ans == "yes":
            candidate &= set(genre_to_user[best_genres[idx]])
            liked_genres.append(best_genres[idx])
        else:
            candidate &= all_user - set(genre_to_user[best_genres[idx]])

    recommand = defaultdict(int)
    for user in candidate:
        for st in df[df.user_id == user].values.tolist():
            overlap = set(item_to_genre[st[1]]) & set(liked_genres)
            if overlap:
                recommand[st[1]] += (int(st[2]) - 5) * (len(overlap) / len(liked_genres))

    return [r[0] for r in sorted(recommand.items(), key=lambda x: x[1], reverse=True)]

def tfidf():
    a_df = pd.read_csv("anime.csv").fillna("")
    r_df = pd.read_csv("rating.csv").drop(["id"], axis = 1)
    clicked, genres = [], []
    for st in a_df.values.tolist():
        # print(st)
        item_to_genre[st[1]] = [a.strip() for a in st[3].split(",") if len(a) > 0]
        genres += st[3].split(",")

    num_genres = len(set(genres))
    print(num_genres)
    for st in r_df.values.tolist():
        for genre in item_to_genre[st[1]]:
            # if st.user_id not in genre_to_user[genre]:
            genre_to_user[genre].append(st[0])

        corpus[st[0]] += item_to_genre[st[1]]

    # cur_id.extend([maxid])
    # cur_user_id.extend(([maxuid]))
    word_frequency, doc_frequency, word_tf, word_idf = defaultdict(int), defaultdict(int), defaultdict(float), defaultdict(float)
    for st in corpus.values():
        for word in st:
            word_frequency[word] += 1
        for word in set(st):
            doc_frequency[word] += 1

    num_words = sum(word_frequency.values())
    for word in word_frequency:
        word_tf[word] = word_frequency[word] / num_words

    for doc in doc_frequency:
        word_idf[doc] = math.log(len(corpus) / (doc_frequency[doc] + 1))

    tf_idf = {}
    for word in word_tf:
        tf_idf[word] = word_tf[word] * word_idf[word]

    sorted_tf_idf = sorted(tf_idf.items(), key=lambda x: x[1], reverse=True)
    for i in range(5):
        questionaire.append({"genre": sorted_tf_idf[i][0]})
        best_genres.append(sorted_tf_idf[i][0])


class NCF(nn.Module):
    def __init__(self, num_users, num_items, model, factor_num=8, num_layers=3,
                 MLP_model=None, GMF_model=None, alpha=0.5, dropout=0.5):
        super(NCF, self).__init__()
        self.MLP_model = MLP_model
        self.GMF_model = GMF_model
        self.alpha = alpha
        self.dropout = dropout
        self.user_embed_GMF = nn.Embedding(num_users, factor_num)
        self.item_embed_GMF = nn.Embedding(num_items, factor_num)
        self.user_embed_MLP = nn.Embedding(num_users, factor_num * (2 ** (num_layers - 1)))
        self.item_embed_MLP = nn.Embedding(num_items, factor_num * (2 ** (num_layers - 1)))
        self.sigmoid = nn.Sigmoid()

        self.MLP = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(factor_num * (2 ** num_layers), factor_num * (2 ** (num_layers - 1))),
            nn.ReLU()
        )
        for layer in range(num_layers - 1, 0, -1):
            self.MLP.add_module('dropout' + str(num_layers - layer), nn.Dropout(p=self.dropout))
            self.MLP.add_module('linear' + str(num_layers - layer),
                                nn.Linear(factor_num * (2 ** layer), factor_num * (2 ** (layer - 1))))
            self.MLP.add_module('relu' + str(num_layers - layer), nn.ReLU())

        self.model = model
        if self.model in ['GMF', 'MLP']:
            self.NeuMF = nn.Linear(factor_num, 1)
        else:
            self.NeuMF = nn.Linear(2 * factor_num, 1)

        self.__init_weights__()

    def __init_weights__(self):
        if self.model in ['GMF', 'MLP']:
            nn.init.normal_(self.user_embed_GMF.weight, std=0.01)
            nn.init.normal_(self.item_embed_GMF.weight, std=0.01)
            nn.init.normal_(self.user_embed_MLP.weight, std=0.01)
            nn.init.normal_(self.item_embed_MLP.weight, std=0.01)

            for layer in self.MLP:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
            #                   nn.init.normal_(layer.weight, std = 0.01)

            nn.init.kaiming_uniform_(self.NeuMF.weight, a=1, nonlinearity='sigmoid')
        #             nn.init.normal_(self.NeuMF.weight, std = 0.01)

        elif self.GMF_model and self.MLP_model:
            self.user_embed_GMF.weight.data.copy_(self.GMF_model.user_embed_GMF.weight)
            self.item_embed_GMF.weight.data.copy_(self.GMF_model.item_embed_GMF.weight)
            self.user_embed_MLP.weight.data.copy_(self.MLP_model.user_embed_MLP.weight)
            self.item_embed_MLP.weight.data.copy_(self.MLP_model.item_embed_MLP.weight)

            for (m1, m2) in zip(self.MLP, self.MLP_model.MLP):
                if isinstance(m1, nn.Linear) and isinstance(m2, nn.Linear):
                    m1.weight.data.copy_(m2.weight)
                    m1.bias.data.copy_(m2.bias)

            NeuMF_weight = torch.cat(
                (self.alpha * self.GMF_model.NeuMF.weight, (1 - self.alpha) * self.MLP_model.NeuMF.weight), 1)
            NeuMF_bias = self.GMF_model.NeuMF.bias + self.MLP_model.NeuMF.bias

            self.NeuMF.weight.data.copy_(NeuMF_weight)
            self.NeuMF.bias.data.copy_(NeuMF_bias)

    def forward(self, user, item):
        if self.model is 'GMF' or 'NCF':
            user_embed_GMF = self.user_embed_GMF(user)
            item_embed_GMF = self.item_embed_GMF(item)

            #             print(user_embed_GMF.device, item_embed_GMF.decive)
            GMF_output = user_embed_GMF * item_embed_GMF

        if self.model is 'MLP' or 'NCF':
            user_embed_MLP = self.user_embed_MLP(user)
            item_embed_MLP = self.item_embed_MLP(item)

            MLP_input = torch.cat((user_embed_MLP, item_embed_MLP), 1)
            MLP_output = self.MLP(MLP_input)

        if self.model is 'NCF':
            output = self.NeuMF(torch.cat((MLP_output, GMF_output), 1))
        elif self.model is 'MLP':
            output = self.NeuMF(MLP_output)
        elif self.model is 'GMF':
            output = self.NeuMF(GMF_output)

        return self.sigmoid(output)

class NCFDataset(Data.Dataset):
    def __init__(self, data_ps, labels):
        super(NCFDataset, self).__init__()
        self.data = data_ps
        self.label = labels

    def __getitem__(self, idx):
        user = self.data[idx][0]
        item = self.data[idx][1]
        label = self.label[idx]

        return user, item, label

    def __len__(self):
#         return self.num_ng * len(self.users) + len(self.data_ps)
        return len(self.data)

def train(net, num_epochs, data_iter, lr):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     device = torch.device('cpu')
    print('train on', device)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.MSELoss()
    for epoch in range(num_epochs):
        print(epoch + 1)
        l_sum, n = 0, 0
        for user, item, label in data_iter:
            # print(user, item, label)
            user = user.to(device)
            item = item.to(device)
            label = label.to(device)
            pred = net(user, item)
            l = loss(pred.view(label.shape), label.float())

            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            l_sum += l.cpu().item()
            n += 1

        print(epoch + 1, l_sum / n)

def fill_rating(anime_id):
    temp = animes.query.filter(animes.anime_id == anime_id).all()
    return temp[0].weight

def predict(net, user, num_animes, items):
    df = pd.read_csv("anime.csv")
    all_items = torch.LongTensor([i for i in range(num_animes)]).cuda()
    all_user = torch.LongTensor([user for _ in range(num_animes)]).cuda()

    pred = net(all_user, all_items)
    _, idx = torch.topk(pred, k=10, dim=0)

    names = []
    for i in idx:
        names.append(df[df.anime_id == i].name.tolist()[0])

    return names

def update():
    df = pd.read_csv("rating.csv").drop(["id"], axis = 1)
    ratings, users, items = df.values.tolist(), df.user_id.tolist(), df.anime_id.tolist()

    for d in ratings:
        users.append(d[0])
        items.append(d[1])

    print('1')
    users, items = list(set(users)), list(set(items))
    num_users, num_animes = len(users), len(items)

    user_to_idx = {int(user): idx for idx, user in enumerate(users)}
    anime_to_idx = {int(anime): idx for idx, anime in enumerate(items)}

    user_item_dic, data, labels = defaultdict(list), [], []

    for d in ratings:
        user_item_dic[user_to_idx[d[0]]].append(anime_to_idx[d[1]])
        data.append([user_to_idx[d[0]], anime_to_idx[d[1]]])
        if int(d.rating) != -1:
            labels.append(d[2] / 10)
        else:
            labels.append(float(fill_rating(d[1])))


    # print(data)
    # print(labels)

    print('2')
    # users = [user_to_idx[i] for i in list(set(train_data[:, 0]))]
    # num_users, num_animes = 100, 100
    dataset = NCFDataset(data, labels)
    data = Data.DataLoader(dataset, batch_size=10000, shuffle=True)

    print("3")
    MLP_net = NCF(num_users, num_animes, model='MLP')
    train(MLP_net, 30, data, lr=0.0001)

    print('4')
    GMF_net = NCF(num_users, num_animes, model='GMF')
    train(GMF_net, 30, data, lr=0.0001)

    print('5')
    NCF_net = NCF(num_users, num_animes, model='NCF', GMF_model=GMF_net, MLP_model=MLP_net)
    train(NCF_net, 30, data, lr=0.0001)

    # names = predict(NCF_net, max(df.user_id.tolist()), num_animes, items)
    return predict(NCF_net, max(df.user_id.tolist()), num_animes, items)

if __name__ == '__main__':
    app.run()
