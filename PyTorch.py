# import torch
# import torch.nn as nn
# import torch.utils.data as Data
# from app import rating
# from app import animes
# from collections import defaultdict
#
# class NCF(nn.Module):
#     def __init__(self, num_users, num_items, model, factor_num=8, num_layers=3,
#                  MLP_model=None, GMF_model=None, alpha=0.5, dropout=0.5):
#         super(NCF, self).__init__()
#         self.MLP_model = MLP_model
#         self.GMF_model = GMF_model
#         self.alpha = alpha
#         self.dropout = dropout
#         self.user_embed_GMF = nn.Embedding(num_users, factor_num)
#         self.item_embed_GMF = nn.Embedding(num_items, factor_num)
#         self.user_embed_MLP = nn.Embedding(num_users, factor_num * (2 ** (num_layers - 1)))
#         self.item_embed_MLP = nn.Embedding(num_items, factor_num * (2 ** (num_layers - 1)))
#         self.sigmoid = nn.Sigmoid()
#
#         self.MLP = nn.Sequential(
#             nn.Dropout(p=self.dropout),
#             nn.Linear(factor_num * (2 ** num_layers), factor_num * (2 ** (num_layers - 1))),
#             nn.ReLU()
#         )
#         for layer in range(num_layers - 1, 0, -1):
#             self.MLP.add_module('dropout' + str(num_layers - layer), nn.Dropout(p=self.dropout))
#             self.MLP.add_module('linear' + str(num_layers - layer),
#                                 nn.Linear(factor_num * (2 ** layer), factor_num * (2 ** (layer - 1))))
#             self.MLP.add_module('relu' + str(num_layers - layer), nn.ReLU())
#
#         self.model = model
#         if self.model in ['GMF', 'MLP']:
#             self.NeuMF = nn.Linear(factor_num, 1)
#         else:
#             self.NeuMF = nn.Linear(2 * factor_num, 1)
#
#         self.__init_weights__()
#
#     def __init_weights__(self):
#         if self.model in ['GMF', 'MLP']:
#             nn.init.normal_(self.user_embed_GMF.weight, std=0.01)
#             nn.init.normal_(self.item_embed_GMF.weight, std=0.01)
#             nn.init.normal_(self.user_embed_MLP.weight, std=0.01)
#             nn.init.normal_(self.item_embed_MLP.weight, std=0.01)
#
#             for layer in self.MLP:
#                 if isinstance(layer, nn.Linear):
#                     nn.init.xavier_uniform_(layer.weight)
#             #                   nn.init.normal_(layer.weight, std = 0.01)
#
#             nn.init.kaiming_uniform_(self.NeuMF.weight, a=1, nonlinearity='sigmoid')
#         #             nn.init.normal_(self.NeuMF.weight, std = 0.01)
#
#         elif self.GMF_model and self.MLP_model:
#             self.user_embed_GMF.weight.data.copy_(self.GMF_model.user_embed_GMF.weight)
#             self.item_embed_GMF.weight.data.copy_(self.GMF_model.item_embed_GMF.weight)
#             self.user_embed_MLP.weight.data.copy_(self.MLP_model.user_embed_MLP.weight)
#             self.item_embed_MLP.weight.data.copy_(self.MLP_model.item_embed_MLP.weight)
#
#             for (m1, m2) in zip(self.MLP, self.MLP_model.MLP):
#                 if isinstance(m1, nn.Linear) and isinstance(m2, nn.Linear):
#                     m1.weight.data.copy_(m2.weight)
#                     m1.bias.data.copy_(m2.bias)
#
#             NeuMF_weight = torch.cat(
#                 (self.alpha * self.GMF_model.NeuMF.weight, (1 - self.alpha) * self.MLP_model.NeuMF.weight), 1)
#             NeuMF_bias = self.GMF_model.NeuMF.bias + self.MLP_model.NeuMF.bias
#
#             self.NeuMF.weight.data.copy_(NeuMF_weight)
#             self.NeuMF.bias.data.copy_(NeuMF_bias)
#
#     def forward(self, user, item):
#         if self.model is 'GMF' or 'NCF':
#             user_embed_GMF = self.user_embed_GMF(user)
#             item_embed_GMF = self.item_embed_GMF(item)
#
#             #             print(user_embed_GMF.device, item_embed_GMF.decive)
#             GMF_output = user_embed_GMF * item_embed_GMF
#
#         if self.model is 'MLP' or 'NCF':
#             user_embed_MLP = self.user_embed_MLP(user)
#             item_embed_MLP = self.item_embed_MLP(item)
#
#             MLP_input = torch.cat((user_embed_MLP, item_embed_MLP), 1)
#             MLP_output = self.MLP(MLP_input)
#
#         if self.model is 'NCF':
#             output = self.NeuMF(torch.cat((MLP_output, GMF_output), 1))
#         elif self.model is 'MLP':
#             output = self.NeuMF(MLP_output)
#         elif self.model is 'GMF':
#             output = self.NeuMF(GMF_output)
#
#         return self.sigmoid(output)
#
# class NCFDataset(Data.Dataset):
#     def __init__(self, data_ps, labels):
#         super(NCFDataset, self).__init__()
#         self.data = data_ps
#         self.label = labels
#
#     def __getitem__(self, idx):
#         user = self.data[idx][0]
#         item = self.data[idx][1]
#         label = self.label[idx]
#
#         return user, item, label
#
#     def __len__(self):
# #         return self.num_ng * len(self.users) + len(self.data_ps)
#         return len(self.data)
#
# def train(net, num_epochs, data_iter, lr):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     #     device = torch.device('cpu')
#     print('train on', device)
#     net = net.to(device)
#     optimizer = torch.optim.Adam(net.parameters(), lr=lr)
#     loss = nn.MSELoss()
#     for epoch in range(num_epochs):
#         print(epoch + 1)
#         l_sum, n = 0, 0
#         for user, item, label in data_iter:
#             # print(user, item, label)
#             user = user.to(device)
#             item = item.to(device)
#             label = label.to(device)
#             pred = net(user, item)
#             l = loss(pred.view(label.shape), label.float())
#
#             optimizer.zero_grad()
#             l.backward()
#             optimizer.step()
#             l_sum += l.cpu().item()
#             n += 1
#
#         print(epoch + 1, l_sum / n)
#
# def fill_rating(anime_id):
#     temp = animes.query.filter(animes.anime_id == anime_id).all()
#     return temp[0].weight
#
# def predict(net, user, num_animes):
#     all_items = torch.LongTensor([i for i in range(num_animes)]).cuda()
#     all_user = torch.LongTensor([user for _ in range(num_animes)]).cuda()
#
#     pred = net(all_user, all_items)
#     _, idx = torch.topk(pred, k=5, dim=0)
#
#     return idx
#
# def update():
#     # cursor = connection.cursor()
#     # cursor.execute('select * from testapp_ratings')
#
#     ratings, users, items = rating.query.filter().all(), [], []
#     print(ratings[0].user_id, len(ratings))
#     for d in ratings:
#         if d.user_id == 'user_id':
#             continue
#         users.append(d.user_id)
#         items.append(d.anime_id)
#
#     print('1')
#     users, items = list(set(users)), list(set(items))
#     num_users, num_animes = len(users), len(items)
#
#     user_to_idx = {int(user): idx for idx, user in enumerate(users)}
#     anime_to_idx = {int(anime): idx for idx, anime in enumerate(items)}
#
#     user_item_dic, data, labels = defaultdict(list), [], []
#
#     for d in ratings:
#         if d.user_id == "user_id":
#             continue
#         user_item_dic[user_to_idx[int(d.user_id)]].append(anime_to_idx[int(d.anime_id)])
#         data.append([user_to_idx[int(d.user_id)], anime_to_idx[int(d.anime_id)]])
#         if int(d.rating) != -1:
#             labels.append(int(d.rating) / 10)
#         else:
#             labels.append(float(fill_rating(d.anime_id)))
#
#
#     # print(data)
#     # print(labels)
#
#     print('2')
#     # users = [user_to_idx[i] for i in list(set(train_data[:, 0]))]
#     # num_users, num_animes = 100, 100
#     dataset = NCFDataset(data, labels)
#     data = Data.DataLoader(dataset, batch_size=10000, shuffle=True)
#
#     print("3")
#     MLP_net = NCF(num_users, num_animes, model='MLP')
#     train(MLP_net, 30, data, lr=0.0001)
#
#     print('4')
#     GMF_net = NCF(num_users, num_animes, model='GMF')
#     train(GMF_net, 30, data, lr=0.0001)
#
#     print('5')
#     NCF_net = NCF(num_users, num_animes, model='NCF', GMF_model=GMF_net, MLP_model=MLP_net)
#     train(NCF_net, 30, data, lr=0.0001)
#
#     # names = predict(NCF_net, user, num_animes)
#     return NCF_net
#
# if __name__ == "__main__":
#     net = update()