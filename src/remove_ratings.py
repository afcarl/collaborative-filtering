import pandas as pd

def remove_ratings(file_path, nFirstRatings, nLastRatings, outputFile):

    tbl = pd.read_csv(file_path, names=['user', 'item', 'rating'])

    n_rating = tbl.shape[0]

    nRatingPerUser = tbl['user'].value_counts()

    iUser = 0
    ratings = pd.DataFrame()

    while iUser < n_rating:

        iRating = 0
        nRatingOfUser = nRatingPerUser[tbl.iloc[iUser]['user']]

        if nFirstRatings + nLastRatings >= nRatingOfUser:

            iUser += nRatingOfUser
            continue

        ratings = pd.concat([ratings, tbl[iUser + nFirstRatings: iUser + nRatingOfUser - nLastRatings]])
        iUser += nRatingOfUser


    pd.DataFrame(ratings).to_csv(outputFile, index=False)

if __name__ == "__main__":

    preShrunk =  20
    postShrunk = 20

    output = "input/customeraffinity500k_shrunk_%d_%d.csv" % (preShrunk, postShrunk)

    remove_ratings("input/customeraffinity500k.train", preShrunk, postShrunk, output)




