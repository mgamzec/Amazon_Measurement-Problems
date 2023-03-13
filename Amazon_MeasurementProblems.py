import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df = pd.read_csv("Measurement_Problems/measurement_problems-220803-201851/measurement_problems/RatingProductSortingReviewsinAmazon-221119-111357/Rating Product&SortingReviewsinAmazon/amazon_review.csv")
df.shape
df.head(20)
df.dtypes
df["asin"].nunique()

df["overall"].mean()

df.dtypes
df["reviewTime"] = pd.to_datetime(df["reviewTime"])

df["reviewTime"].max()
current_date = pd.to_datetime('2014-12-07 0:0:0')

df["day"] = (current_date - df["reviewTime"]).dt.days
df["day"].max()

df["day"].describe().T

def time_based_weighted_average(dataframe, w1=35, w2=30, w3=20, w4=15):
    return dataframe.loc[df["day"] <= 280, "overall"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["day"] > 280) & (dataframe["day"] <= 430), "overall"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["day"] > 430) & (dataframe["day"] <= 600), "overall"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["day"] > 600), "overall"].mean() * w4 / 100

time_based_weighted_average(df)

df.loc[df["day"] <= 280, "overall"].mean() #= 4.6957928802588995

df.loc[(df["day"] > 280) & (df["day"] <= 430), "overall"].mean() #= 4.636140637775961

df.loc[(df["day"] > 430) & (df["day"] <= 600), "overall"].mean()  #= 4.571661237785016

df.loc[(df["day"] > 600), "overall"].mean() #= 4.4462540716612375

df["helpful_no"] = df["total_vote"] - df["helpful_yes"]
df["helpful_no"].head(10)
df["helpful_no"].sort_values(ascending=False).head()

df[["helpful_yes","helpful_no","total_vote"]].iloc[[4212, 2909]]

# •score_pos_neg_diff;

    def score_pos_neg_diff(helpful_yes, helpful_no):
        return helpful_yes - helpful_no

df["score_pos_neg_diff"] = df.apply(lambda df: score_pos_neg_diff(df["helpful_yes"], df["helpful_no"]), axis=1)
df.head()
df["score_pos_neg_diff"].describe().T

# •score_average_rating;

def score_average_rating(helpful_yes, helpful_no):
    if helpful_yes == 0:
        return 0
    elif helpful_no==0:
        return 0
    else:
        return helpful_yes / helpful_no

df["score_average_rating"] = df.apply(lambda df: score_average_rating(df["helpful_yes"], df["helpful_no"]), axis=1)
df["score_average_rating"].describe().T
df.head()

# •wilson_lower_bound;

"""Calculate Wilson Lower Bound Score
- The lower limit of the confidence interval to be calculated for the Bernoulli parameter p is accepted as the WLB score.
- The score to be calculated is used for product ranking.
- Note:
If the scores are between 1-5, 1-3 are marked as negative, 4-5 as positive and can be made to conform to Bernoulli.
This brings with it some problems. For this reason, it is necessary to make a bayesian average rating.

 Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence
    Returns
    -------
    wilson score: float"""

def wilson_lower_bound(helpful_yes, helpful_no, confidence=0.95):
    n = helpful_yes + helpful_no
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * helpful_yes / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)

df["wilson_lower_bound"].describe().T
df.head()


df= df[["reviewerID", "asin", "overall", "day", "total_vote", "helpful_no", "helpful_yes", "score_pos_neg_diff", "score_average_rating", "wilson_lower_bound"]]
df.head()
df.sort_values("wilson_lower_bound", ascending=False).head(20)
