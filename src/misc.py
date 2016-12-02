from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re

from data import trainSet


def pd_stats():
    tbl = pd.read_table(trainSet, sep=',', header=None)
    users, items, ratings = tbl.iloc[:, 0], tbl.iloc[:, 1], tbl.iloc[:, 2]
    plt.xlabel('Rating')
    plt.ylabel('Proportion (training set)')
    plt.title('Ratings distribution')
    plt.bar(range(1, 6), ratings.value_counts(sort=False, normalize=True), color='black', align='center')
    plt.savefig('../output/ratings.png')

def training_set_stats():
    """
    Load customer affinity data (train and test sets), convert item to sparse matrix
    :return: train and test sparse matrices
    """

    item_multiplicities = defaultdict(int)
    user_multiplicities = defaultdict(int)
    m1, m2 = 0, 0
    s1, s2 = set(), set()
    hist = [0] * 5
    lines = 0
    curr_usr = None
    seq_count = 0
    max_users = 2000

    with open(trainSet) as train_file:
        for line in train_file:
            if not lines:
                lines += 1
                continue
            splits = [int(s) for s in line.split(",")]
            item = splits[1]
            user = splits[0]
            item_multiplicities[item] += 1
            user_multiplicities[user] += 1
            m1 = max(m1, user)
            m2 = max(m2, item)
            hist[splits[2] - 1] += 1
            s1.add(user)
            s2.add(item)
            lines += 1
            if len(s1) < max_users:
                if user == curr_usr:
                    seq_count += 1
                else:
                    print("User {} rated {} items".format(curr_usr, seq_count))
                    curr_usr = user
                    seq_count = 0

    n_s1, n_s2 = len(s1), len(s2)

    print("Number of entries: {}".format(lines))
    print("Max ids are {} and {}".format(m1, m2))
    print("Number of unique ids are {} and {}".format(n_s1, n_s2))
    print("Sparsity = {}".format(lines / (n_s1 * n_s2)))  # How sparse is the data?

    plt.scatter(np.arange(5), hist)
    plt.show()

    hist = [round(h / sum(hist), 4) for h in hist]
    print(hist)
    item_dist = np.array(list(item_multiplicities.values()))
    user_dist = np.array(list(user_multiplicities.values()))
    print_moments(item_dist)
    print_moments(user_dist)

    # Number of times each item was seen
    plt.hist(item_dist, bins=100)
    plt.show()
    # Number of times each user was seen
    plt.hist(user_dist, bins=100)
    plt.show()

def print_moments(distrib):
    print('Min = {}, Max = {}, Mean = {}, Median = {}, Var = {}, Std = {}' \
          .format(distrib.min(),
                  distrib.max(),
                  distrib.mean().round(1),
                  np.median(distrib),
                  distrib.var().round(1),
                  distrib.std().round(1)))

def parse_torch_output(fn):
    with open(fn) as txt:
        lines = map(str.split(), txt.readlines())
        lines = re.findall(re.compile('RMSE\b:\b[:digit:]'), lines)



def plt_cfn():
    u = [0.94680560914807,
         0.94457654256013,
         0.94343798942158,
         0.94262353890841,
         0.94179643950918,
         0.94138749280882,
         0.94067878768737,
         0.94051487431474,
         0.94017780681702,
         0.93998988772327,
         0.93953636179447,
         0.93929826838418,
         0.93932459663449,
         0.93917123392959,
         0.93896232582131,
         0.93898443653649,
         0.9390245555849,
         0.93886742567097,
         0.93882672893517,
         0.93879778126119,
         0.93880392876407,
         0.93869065843843,
         0.93874676069384,
         0.93860716124258,
         0.93857469735498,
         0.93865821097416,
         0.93864470058452,
         0.9386363336212,
         0.93857216656589,
         0.93850548062991,
         0.93848979704675,
         0.93837777948025,
         0.93838587885084,
         0.93830055314324,
         0.93821481659366,
         0.93817327092589,
         0.93811344428973,
         0.93807515036864,
         0.93812022593704,
         0.93813905445081,
         0.93815317754196,
         0.93814486953468,
         0.93806601811056,
         0.93796398474153,
         0.93795473256923,
         0.93796795676427,
         0.93790620653436,
         0.93790262730305,
         0.93786123494787,
         0.93785423872778]

    v = [0.9516564480298,
         0.94624321313797,
         0.93386294071527,
         0.93374232589354,
         0.93098874763469,
         0.93006397335772,
         0.93020289833589,
         0.92786580188732,
         0.92867086659801,
         0.92507065443873,
         0.92346292082992,
         0.92126587824001,
         0.92157606403286,
         0.92017814437837,
         0.92056197604904,
         0.92017408653293,
         0.91900622117043,
         0.91851182737705,
         0.91867242244114,
         0.91838582163118,
         0.91831686459034,
         0.91828582701634,
         0.91787455514221,
         0.91786202681441,
         0.91738355729863,
         0.91754266902253,
         0.91765688683396,
         0.91755275007883,
         0.91756679872128,
         0.91753528502328,
         0.91692782905146,
         0.91653978224101,
         0.91628371170339,
         0.91596253796015,
         0.9156666294161,
         0.91528653473881,
         0.91509066881721,
         0.91484012378175,
         0.91496661597289,
         0.91490910290921,
         0.91458121082754,
         0.91435425148988,
         0.9141682205061,
         0.91364953850769,
         0.91380633846658,
         0.91377774997133,
         0.9136832836751,
         0.91339156355506,
         0.91314126496059,
         0.91309129871606]

    als = [1.235835658334,
 1.0837206864586,
 0.96021594685431,
 0.94617554813641,
 0.93958796520239,
 0.93682547046789,
 0.93561060100812,
 0.9350286697578,
 0.9347147689982,
 0.93452517189567,
 0.93439964975556,
 0.93431067959882,
 0.93424453656915,
 0.9341937546781,
 0.93415392234357]

    plt.plot(u, label='U-CFN')
    plt.plot(v, label='V-CFN')
    #plt.plot(als, label='ALS-WR (d=100, lambda=0.01)')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE (test)')
    plt.title('Evolution of the testing error')
    plt.legend()
    plt.savefig('../output/cfn.png')

if __name__ == '__main__':
    plt_cfn()
