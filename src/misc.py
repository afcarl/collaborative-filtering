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
    print('Min = {}, Max = {}, Mean = {}, Median = {}, Var = {}, Std = {}'\
          .format(distrib.min(),
                  distrib.max(),
                  distrib.mean().round(1),
                  np.median(distrib),
                  distrib.var().round(1),
                  distrib.std().round(1)))

