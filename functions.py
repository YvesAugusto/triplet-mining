from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_samples, silhouette_score
from numpy.linalg import norm as l2
import pickle, os, random, time
import numpy as np
from face_recognition.log import logger
from face_recognition.feature_extraction.extract_features import extract_features_of_roi
from collections import Counter

# imports from variables.sh
TRAIN_PATH = os.getenv("TRAIN_PATH")
DATAKEYS_PATH=os.getenv("DATAKEYS_PATH")
MODEL_PATH=os.getenv("MODEL_PATH")
CLASSES_PATH=os.getenv("CLASSES_PATH")
TEST_PATH=os.getenv("TEST_PATH")
DIC_PATH=os.getenv("DIC_PATH")
DATA_TEST_PATH=os.getenv("DATA_TEST_PATH")
CLUSTER_PATH=os.getenv("CLUSTER_PATH")

# def define_best_k(X,range_clusters): # function to compute silhouette for k-means clustering
#     for n_clt in range_clusters:
#         clusterer = KMeans(init='k-means++',
#                    n_clusters=n_clt,
#                    n_init=50, tol=1e-8, max_iter=100,
#                    random_state=10
#                    )
#         cluster_labels = clusterer.fit_predict(X)
#         silhouette_avg = silhouette_score(X, cluster_labels)
#         print("For n_clusters =", n_clt, "The average silhouette_score is :", silhouette_avg)
#     return 0

# function to shuffle data and label vectors in correspondance to each other
def pair_shuffle(x,y):
    c=list(zip(x,y))
    random.shuffle(c)
    x,y=zip(*c)
    return np.array(x), np.array(y)

# compute the average of a list
def Average(lst):
    lst=np.array(lst)
    print(lst.shape)
    for l in lst:
        if l is None:
            logger.info("None found")
            exit(1)
    return sum(lst) / len(lst)

# return the arguments of three greatest elements on a list
def n_greatest(v,n):
    a=v
    a=sorted(a)
    return [v.index(a[i]) for i in range(n)]

# compute the distance between a feature vector and each one of the k-means' centroids
def compute_distance(centroids, vector, n):
    return n_greatest([l2(vector-c) for c in centroids],n)

# load a file given it's name
def load_file(filename):
    with open(filename, "rb") as file:
        dataset=pickle.load(file)
        #logger.info("Dataset len: {}".format(len(dataset)))
        return dataset

# compute the embedding of all train images for each user
def save_train_features(data):
    start_timestamp=time.time()
    dataset={}
    full_dataset={}
    logger.info("Entered on reduce train features process")
    for user, paths in data.items():
        features=[extract_features_of_roi(path)[0] for path in paths]
        avg=np.array(Average(features))
        arg=np.array(l2(vector-avg) for vector in features).argmin()
        dataset.update({user:avg})
        full_dataset.update({user:[features,paths]})
    logger.info("Extraction concluded in {} seconds.".format(time.time() - start_timestamp))
    with open(DATAKEYS_PATH, "wb") as file:
        pickle.dump(dataset,file)
    with open(TRAIN_PATH, "wb") as file:
        pickle.dump(full_dataset,file)
    return 0

# compute the embedding of all test features for each user
def save_test_features(data):
    start_timestamp = time.time()
    dataset = {}
    logger.info("Entered on reduce test features process")
    for user, paths in data.items():
        features = [extract_features_of_roi(path)[0] for path in paths]
        dataset.update({user:[features,paths]})
    # saving test database
    with open(TEST_PATH, "wb") as file:
        pickle.dump(dataset,file)
    logger.info("Extraction concluded in {} seconds.".format(time.time() - start_timestamp))

# return user train features given it's name
def get_features_from_user(name):
    # loading train a catalogues
    with open(TRAIN_PATH, "rb") as file:
        train=pickle.load(file)

    # separing train and test features
    # logger.info('Getting features from user {}'.format(name))
    if name in train:
        t = np.array(train[name][0])
        return (np.reshape(t, (t.shape[0], t.shape[2])))
    else:
        logger.info("User {} not enrolled on dataset".format(name))
        return None, None, None
    # logger.info('len-train: {}'.format(len(train_features)))
    # logger.info('len-test: {}'.format(len(test_features)))



def lowest(v):
    a = v
    a = sorted(a)
    return v.index(a[0])

def compute_lowest_distance(centroids, vector):
    return np.array([l2(vector - c) for c in centroids]).argmin()

# detects for an user the cluster to where it's features converges more
def reallocate_user(vets, centroids):
    classification=[]
    for v in vets:
        closest=compute_lowest_distance(centroids,v)
        classification.append(closest)
    c=Counter(classification)
    return c.most_common(1)[0][0]

# function to delete clusters filled with less than X users, and redistribute it's users
# to the remaining clusters.
def cut_clusters(classes, centers, thrs=7):
    centers=np.array(centers)
    classes=np.array(classes)
    to_cut={}
    for idc,c in enumerate(classes):
        if(len(c)<=thrs):
            to_cut.update({idc:c})

    index=[index for index, users in to_cut.items()]
    new_classes = [classes[i] for i in range(len(classes)) if i not in index]
    new_centers = [centers[i] for i in range(len(centers)) if i not in index]
    logger.info("Index of wrong clusters {}".format(index))
    for idc,c in to_cut.items():
        logger.info("Cluster {} has minus than {} users".format(idc, thrs))
        # logger.info("Classes vector before remotion: {}".format(classes))
        # logger.info("Classes vector after remotion: {}".format(classes))
        for name in c:
            vets=get_features_from_user(name)
            new_cluster=reallocate_user(vets, new_centers)
            logger.info("New cluster for user {} is cluster number {}".format(name, new_cluster))
            new_classes[new_cluster].append(name)

    new_classes=np.array(new_classes)
    new_centers=np.array(new_centers)
    print("New shape for classes and centers: {}, {}".format(new_classes.shape, new_centers.shape))
    return new_classes, new_centers

# distribute the users, allong the clusters, according to which centroid
# each of it's features more tends to
def make_clusters(centers,names,dic, dataset):
    classes=[]
    for i in range(len(centers)):
        v=[]
        classes.append(v)
    for idn, n in enumerate(names):
        pred_clusters=[compute_lowest_distance(centers,feat[0]) for feat in dataset[n][0]]
        c=Counter(pred_clusters)
        # logger.info("For user {}, configuration: {}".format(n, c))
        arg=c.most_common(1)[0][0]
        # arg2=c.most_common(2)[1][0]
        classes[arg].append(n)
        # if c.most_common(1)[0][1] - c.most_common(2)[1][1] < 5:
        #     classes[arg2].append(n)

    return classes

def avaliate(classes, ctrs, dataset, n):
    sum=0
    qtd=0
    total_wrong=0
    for user, fts in dataset.items():
        true_centroid=[idc for idc, c in enumerate(classes) if user in c]
        report={}
        for i in range(len(fts[0])):
            winners = compute_distance(ctrs, fts[0][i], n)
            st = "centers_"
            st2 = ''
            for i in range(len(winners)):
                st2+=str(winners[i])+"_"
            st += st2
            if st not in report.keys():
                report.update({st:[]})

            distances=[]
            for i in range(len(winners)):
                distances.append(l2(ctrs[true_centroid] - ctrs[winners[i]]))
            # distance = l2(ctrs[true_centroid] - ctrs[winner])
            minimum=min(distances)
            if minimum > 0:
                total_wrong+=1
            report[st].append(minimum)
            sum+=minimum
            qtd+=1

        logger.info("Report for user {}: {}".format(user, report))
    logger.info("Quantization error: {}".format(sum/qtd))
    logger.info("Miss prediction tax: {}".format(total_wrong/qtd))

#
# now the functions to the online classification process
#

def classify_in_clusters(feat):
    clusters=load_file(CLUSTER_PATH)
    centroids=[c["center"] for c in clusters]
    args = compute_distance(centroids, feat)
    return [clusters[arg] for arg in args]

def jump_to_clusters(clusters, feat):
    preds=[n_greatest(c["svm"]["model"].predict_proba(feat), 3) for c in clusters]
    names=[]
    for idc, c in clusters:
        names_aux=[c["svm"]["dic"][pred] for pred in preds[idc]]
        names.append(names_aux[0])
    return names



