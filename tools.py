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
        self.triplets={} # Will be made on the rule {"user0":[[A, P, N], [A, P, N]] "user1": ...}
        self.report={} # Will be made on the rule {"user0":[[P1,P2], [P3,P4], .. ], "user1": ...}
                       # for each user the positive pairs path will be ordered
                       # from major(first) to minor(last).
        self.data = {}
        self.clusters={}
        self.n_clusters=int(0.375*len(users))
        self.alfa = 0.2
        for user, values in load_file(train_path).items():
            if user in self.users:
                self.data.update({user:values})

        self.clusterize()
        for user in self.users:
            self.hardest_positives_user(user)

        self.make_triplets()

    def get_user_positives(self, user):
        return self.data[user]

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
        classes, ctr = cut_clusters(classes, ctrs, 2)
        for idc, c in enumerate(classes):
            self.clusters.update({"cluster_"+str(c): {"classes":c, "center": ctr[idc], "dic":dic}})

        print(classes)

    def positive_triplet_for_user(self, name, n_hard, n_rand, n_samples):
        paths=self.report[name]["paths"][0:n_hard]
        vecs=self.report[name]["vecs"][0:n_hard]
        negs=self.get_user_negatives(name)
        triplets=[]
        for idv, vec in enumerate(vecs):
            rand = random.choices(list(negs.keys()), k=n_rand)
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
                        triplets.append([paths[idv][0], paths[idv][1], curr_paths[idc]])
                    # now make the permutation to make another triplet with the same images
                    if l2(vec[1] - curr) <= l2(vec[0] - vec[1]) - self.alfa:
                        triplets.append([paths[idv][1], paths[idv][0], curr_paths[idc]])

        self.triplets.update({name: triplets})

        return f'{len(triplets)} created to user {name}'


    def negative_triplet_for_user(self, name):
        pass
    def make_triplets(self):
        for key, val in self.clusters.items():
            for idn, name in enumerate(val["classes"]):
                self.positive_triplet_for_user(name, n_hard=3,
                                         n_rand=5,
                                         n_samples=4)

            print(f'Triplets for cluster {val["classes"]} created')
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
        return True

    def search_for_user(self, name):
        data={}
        for i in range(len(self.batches)):
            if name in self.batches[i].users:
                data.update({'batch_'+str(i):self.batches[i].triplets[name]})
        string = f'triplets len for each batch to user {name}: {[len(d) for c,d in data.items()]}'
        return string

manager = BatchManager()
manager.__int__()
for i in range(2):
    names=select_random()
    batch = Batch(names)
    manager.add_batch(batch)
#
# with open("./batch_manager", "wb") as f:
#     pickle.dump(manager, f)

triplets={}
for idb, batch in enumerate(manager.batches):
    triplets.update({"batch_"+str(idb):batch.triplets})
    print(len(batch.triplets))

print(triplets["batch_0"][0])

with open("./batch_manager", "wb") as f:
    pickle.dump(triplets, f)

