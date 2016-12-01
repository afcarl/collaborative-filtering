import pandas as pd

def remove_ratings(file_path, nFirstRatings, nLastRatings):

    tbl = pd.read_csv(file_path, names=['user', 'item', 'rating'])

    tbl = tbl[0:20]

    n_rating = tbl.shape[0]

    nRatingPerUser = tbl['user'].value_counts()

    iUser = 0
    ratings = pd.DataFrame()

    while iUser < n_rating:

        iRating = 0
        nRatingOfUser = nRatingPerUser[tbl.iloc[iUser]['user']]
        print(tbl.iloc[iUser]['user'], nRatingOfUser)
        if nFirstRatings + nLastRatings >= nRatingOfUser:

            iUser += nRatingOfUser
            continue

        ratings = pd.concat([ratings, tbl[iUser + nFirstRatings: iUser + nRatingOfUser - nLastRatings]])
        iUser += nRatingOfUser


    pd.DataFrame(ratings).to_csv('../input/test.csv', index=False)

if __name__ == "__main__":

    remove_ratings("../input/customeraffinity_last5000.train", 0, 1)




