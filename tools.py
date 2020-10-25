import pickle, numpy as np, os
from face_recognition.clustering.functions import *
from itertools import combinations, product
from numpy.linalg import norm as l2
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans, DBSCAN

train_path=os.getenv("TRAIN_PATH")
test_path=os.getenv("TEST_PATH")
clusters_path=os.getenv("SVM_AND_CLUSTERS_PATH")

users=list(load_file(train_path).keys())

def n_greatest(v,n):
    a=v
    a=sorted(a)
    return [v.index(a[-1-i]) for i in range(n)]

def n_smallest(v,n):
    a=v
    a=sorted(a)
    return [v.index(a[i]) for i in range(n)]

def select_random():
    v=np.random.randint(0, 259, 30)
    return [users[i] for i in range(len(users)) if i in v]

def multiple_norm(comb):
    return [l2(a-b) for a,b in comb]

def compare_users(fts1, fts2):
    fts1=np.mean(fts1)
    fts2=np.mean(fts2)

class Batch:
    def __init__(self, users):
        self.users=users
        self.qtd_triplets=0
        self.max_per_user = [50, 0] # fix a maximum number of positive and negative triplets
                                     # to each user
        self.triplets={} # Will be made on the rule {"user0":[[A, P, N], [A, P, N]] "user1": ...}
        self.triplets_neg={} # Will be made on the rule {"user0":[[A, P, N], [A, P, N]] "user1": ...}
        self.report={} # Will be made on the rule {"user0":[[P1,P2], [P3,P4], .. ], "user1": ...}
                       # for each user the positive pairs path will be ordered
                       # from major(first) to minor(last).
        self.report_neg={}# Will be made on the rule {"user0":[[N1,N2], [N3,N4], .. ], "user1": ...}
                       # for each user the negative pairs path will be ordered
                       # from major(first) to minor(last).
        self.data = {}
        self.clusters={}
        self.n_clusters=int(0.125*len(users))
        self.alfa = 0.2
        for user, values in load_file(train_path).items():
            if user in self.users:
                self.data.update({user:values})

        self.clusterize()
        for user in self.users:
            self.hardest_positives_user(user)
            self.hardest_negatives_user(user)

        self.make_triplets()
        for i in range(len(self.users)):
            self.complete_user(self.users[i])

    def get_user_positives(self, user):
        return self.data[user]

    def get_user_negs(self, user):
        cluster = [v for c, v in self.clusters.items() if user in v["classes"]][0]
        aux = [np.array(dic) for name, dic in self.data.items()
               if (name != user and name in cluster["classes"])]
        return aux


    def get_user_negatives(self, user):
        cluster = [v for c,v in self.clusters.items() if user in v["classes"]][0]
        negs={}
        for name, values in self.data.items():
            if (name != user and name in cluster["classes"]):
                negs.update({name: values})
        return negs

    def hardest_positives_user(self, user):
        if user in self.report.keys():
            return self.report[user]

        pos=self.get_user_positives(user)
        pos_index = np.arange(len(pos[0]))
        comb = list(combinations(pos_index, 2))
        pos_vec = [[pos[0][a], pos[0][b]] for a,b in comb]
        pos_path = [[pos[1][a], pos[1][b]] for a,b in comb]
        pos_index=multiple_norm(pos_vec)
        pos_vec = np.array([pos_vec[i] for i in n_greatest(pos_index, len(pos_vec))])
        pos_path = np.array([pos_path[i] for i in n_greatest(pos_index, len(pos_vec))])
        self.report.update({user: {"paths":pos_path, "vecs":pos_vec}})

        return self.report[user]

        # print(f'Hardest {user} positives computed')

    def hardest_negatives_user(self,user):
        neg_aux = self.get_user_negs(user)
        pos = self.get_user_positives(user)
        neg = [[], []]
        for ft, path in neg_aux:
            neg[0].append(ft)
            neg[1].append(path)
        neg[0] = np.concatenate(neg[0])
        neg[1] = np.concatenate(neg[1])
        neg_index = np.arange(len(neg[0]))
        pos_index = np.arange(len(pos[0]))
        comb = list(product(pos_index, neg_index))
        neg_vec = [[pos[0][a], neg[0][b]] for a, b in comb]
        neg_path = [[pos[1][a], neg[1][b]] for a, b in comb]
        neg_index = multiple_norm(neg_vec)
        neg_vec = np.array([neg_vec[i] for i in n_smallest(neg_index, len(neg_vec))])
        neg_path = np.array([neg_path[i] for i in n_smallest(neg_index, len(neg_vec))])
        self.report_neg.update({user: {"paths": neg_path, "vecs": neg_vec}})


    def clusterize(self):
        dic={}
        names=[]
        k=0
        data=[]
        labels=[]
        for label,feature in self.data.items():
            dic.update({k:label})
            k += 1
            names.append(label)
            data.append(feature[0])
            labels.append([label for i in range(len(feature[0]))])

        data=np.concatenate(data)
        labels=np.concatenate(labels)
        data, labels_y = pair_shuffle(data, labels)
        data=data.reshape(data.shape[0], data.shape[2])
        encoder = LabelEncoder()
        labels = encoder.fit_transform(labels_y)
        model = KMeans(init='k-means++',
                       n_clusters=self.n_clusters,
                       n_init=10, tol=1e-8, max_iter=100,
                       random_state=10)
        model.fit(data)

        logger.info("Model trained successfully")
        ctrs = model.cluster_centers_
        classes = make_clusters(ctrs, names, dic, self.data)
        n=int(len(self.users)/10) - 2
        classes, ctr = cut_clusters(classes, ctrs, n)
        for idc, c in enumerate(classes):
            self.clusters.update({"cluster_"+str(c): {"classes":c, "center": ctr[idc], "dic":dic}})


    def positive_triplet_for_user(self, name, n_samples, n_hard=None, n_rand=None):
        self.triplets.setdefault(name,[])
        if n_hard != None:
            paths=self.report[name]["paths"][0:n_hard]
            vecs=self.report[name]["vecs"][0:n_hard]
        else:
            paths = self.report[name]["paths"]
            vecs = self.report[name]["vecs"]
        negs=self.get_user_negatives(name)
        triplets=[]
        count=0
        for idv, vec in enumerate(vecs):
            if n_rand != None:
                rand = random.choices(list(negs.keys()), k=n_rand)
            else:
                l=list(negs.keys())
                rand = random.choices(l, k=len(l))
            for idr, r in enumerate(rand):
                curr_user = r
                ints = np.arange(n_samples)
                np.random.shuffle(ints)
                ints = ints[0:n_samples]
                curr_paths = negs[r][1][0:n_samples]
                curr_fts = negs[r][0][0:n_samples]
                for idc, curr in enumerate(curr_fts):
                    # here I verify if it is a hard triplet
                    if l2(vec[0] - curr) <= l2(vec[0] - vec[1]) - self.alfa:
                        count+=1
                        self.triplets[name].append([[paths[idv][0],
                                                    paths[idv][1],
                                                    curr_paths[idc]], 1])
                        self.qtd_triplets+=1
                        if count >= self.max_per_user[0]:
                            return True
                    # now make the permutation to make another triplet with the same images
                    if l2(vec[1] - curr) <= l2(vec[0] - vec[1]) - self.alfa:
                        count += 1
                        self.triplets[name].append([[paths[idv][1],
                                                    paths[idv][0],
                                                    curr_paths[idc]], 1])
                        self.qtd_triplets += 1
                        if count >= self.max_per_user[0]:
                            return True


        return f'{len(triplets)} created to user {name}'


    def negative_triplet_for_user(self, name, n_samples, n_hard=None, n_rand=None):
        if n_hard != None:
            paths = self.report_neg[name]["paths"][0:n_hard]
            vecs = self.report_neg[name]["vecs"][0:n_hard]
        else:
            paths = self.report_neg[name]["paths"]
            vecs = self.report_neg[name]["vecs"]
        count = 0
        negs = self.get_user_negatives(name)
        for idv, vec in enumerate(vecs):
            if n_rand != None:
                rand = random.choices(list(negs.keys()), k=n_rand)
            else:
                l=list(negs.keys())
                rand = random.choices(l, k=len(l))
            for idr, r in enumerate(rand):
                curr_user = r
                ints = np.arange(n_samples)
                np.random.shuffle(ints)
                ints = ints[0:n_samples]
                curr_paths = negs[r][1][0:n_samples]
                curr_fts = negs[r][0][0:n_samples]
                for idc, curr in enumerate(curr_fts):
                    # here I verify if it is a hard triplet
                    x=l2(vec[0] - curr) - l2(vec[0] - vec[1])
                    if (4*self.alfa >= x and x >= 3*self.alfa):
                        count+=1
                        self.triplets[name].append([[paths[idv][0],
                                                    paths[idv][1],
                                                    curr_paths[idc]], 0])
                        self.qtd_triplets += 1
                        if count >= self.max_per_user[0]:
                            return True


        return True

    def how_many_triplets_for_user(self,name):
        if name not in list(self.triplets.keys()):
            self.triplets.update({user:[]})
        triplets=np.array(self.triplets[name])
        if len(triplets) > 0:
            pos = len(np.where(triplets[0:,1] == 1)[0])
            neg = len(np.where(triplets[0:,1] == 0)[0])
            return pos, neg
        return 0,0

    def complete_user(self, name):
        pos_0,neg_0 = self.how_many_triplets_for_user(name)
        if pos_0 < self.max_per_user[0]:
            pos = self.get_user_positives(name)
            negs = self.get_user_negatives(name)
            for i in range(self.max_per_user[0] - pos_0):
                rand = np.arange(len(pos))
                np.random.shuffle(rand)
                anchor = pos[1][rand[0]]
                positive = pos[1][rand[1]]
                rand = random.choices(list(negs.keys()), k=1)[0]
                r=np.random.randint(0, len(negs[rand]), 1)[0]
                negative = negs[rand][1][r]
                self.triplets[name].append([[anchor, positive, negative], 1])
        if neg_0 < self.max_per_user[1]:
            pos = self.get_user_positives(name)
            negs = self.get_user_negatives(name)
            for i in range(self.max_per_user[1] - neg_0):
                rand = np.arange(len(pos))
                np.random.shuffle(rand)
                anchor = pos[1][rand[0]]
                rand = random.choices(list(negs.keys()), k=1)[0]
                r = np.random.randint(0, len(negs[rand]), 1)[0]
                negative_1 = negs[rand][1][r]
                rand = random.choices(list(negs.keys()), k=1)[0]
                r = np.random.randint(0, len(negs[rand]), 1)[0]
                negative_2 = negs[rand][1][r]
                self.triplets[name].append([[anchor, negative_1, negative_2], 0])

        return True

    def make_triplets(self):
        for key, val in self.clusters.items():
            for idn, name in enumerate(val["classes"]):
                self.positive_triplet_for_user(name,
                                         n_samples=4) #n_hard, n_rand commented
                self.negative_triplet_for_user(name,
                                         n_samples=4) #n_hard, n_rand commented


            # print(f'Triplets for cluster {val["classes"]} created')
        return True

class BatchManager:

    def __int__(self):
        self.batches=[]

    def __repr__(self):
        return {"len-batches":len(self.batches)}

    def add_batch(self, batch):
        self.batches.append(batch)
        return True

    def del_batch(self, index):
        del self.batches[index]
        return True

    def update_batch(self, index, val):
        self.batches[index].alfa=val
        self.batches[index].make_clusters()
        for i in range(len(self.batches[i].users)):
            self.complete_user(self.batches[i].users[i])
        return True

    def search_for_user(self, name):
        data={}
        for i in range(len(self.batches)):
            if name in self.batches[i].users:
                data.update({'batch_'+str(i):self.batches[i].triplets[name]})
        string = f'triplets len for each batch to user {name}: {[len(d) for c,d in data.items()]}'
        return string

    def how_many_triplets_for_user(self,name):
        for i in range(len(self.batches)):
            if name in self.batches[i].users and len(self.batches[i].triplets[name]) > 0:
                triplets=np.array(self.batches[i].triplets[name])
                pos = len(np.where(triplets[0:,1] == 1)[0])
                neg = len(np.where(triplets[0:,1] == 0)[0])
                return pos, neg
        return False


manager = BatchManager()
manager.__int__()
np.random.shuffle(users)
for i in range(int(len(users)/86)):
    batch = Batch(users[i*86:i*86+86])
    manager.add_batch(batch)

# for user in users:
#     print(f'for user {user}: {manager.how_many_triplets_for_user(user)}')

triplets={}
for idb, batch in enumerate(manager.batches):
    triplets.update({"batch_"+str(idb):batch.triplets})

with open("./batch_manager_full", "wb") as f:
    pickle.dump(triplets, f)

