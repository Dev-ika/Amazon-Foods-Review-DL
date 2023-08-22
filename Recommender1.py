from Database import Database
import  pandas  as  pd
import numpy as np
import matplotlib as plt
from scipy.stats.stats import pearsonr
import re
import operator
import pickle 

def pearson_sim(rest1_reviews, rest2_reviews, n_common):
    """
    Given a subframe of restaurant 1 reviews and a subframe of restaurant 2 reviews,
    where the reviewers are those who have reviewed both restaurants, return 
    the pearson correlation coefficient between the user average subtracted ratings.
    The case for zero common reviewers is handled separately. Its
    ok to return a NaN if any of the individual variances are 0.
    """
    if n_common==0:
        rho=0.
    else:
        diff1=rest1_reviews['Score']-rest1_reviews['user_avg']
        diff2=rest2_reviews['Score']-rest2_reviews['user_avg']
        rho=pearsonr(diff1, diff2)[0]
    return rho

def get_restaurant_reviews(restaurant_id, df, set_of_users):
    """
    given a resturant id and a set of reviewers, return the sub-dataframe of their
    reviews.
    """
    mask = (df.UserId.isin(set_of_users)) & (df.ProductId==restaurant_id)
    reviews = df[mask]
    reviews = reviews[reviews.UserId.duplicated()==False]
    return reviews
	
	
def recompute_frame(ldf):
    """
    takes a dataframe ldf, makes a copy of it, and returns the copy
    with all averages and review counts recomputed
    this is used when a frame is subsetted.
    """
    ldfu=ldf.groupby('UserId')
    ldfb=ldf.groupby('ProductId')
    user_avg=ldfu.Score.mean()
    user_review_count=ldfu.Id.count()
    product_avg=ldfb.Score.mean()
    product_review_count=ldfb.Id.count()
    nldf=ldf.copy()
    nldf.set_index(['ProductId'], inplace=True)
    nldf['product_avg']=product_avg
    nldf['product_review_count']=product_review_count
    nldf.reset_index(inplace=True)
    nldf.set_index(['UserId'], inplace=True)
    nldf['user_avg']=user_avg
    nldf['user_review_count']=user_review_count
    nldf.reset_index(inplace=True)
    return nldf
	
def calculate_similarity(rest1,rest2,df,similarity_func):     #calculates similarity between 2 restaurants using some similarity function
    
    reviewers_rest1=df[df.ProductId==rest1].UserId   #gets all the user ids from df for rest1
    reviewers_rest2=df[df.ProductId==rest2].UserId   #gets all the user ids from df for rest2
    
    #computes the intersection (common reviewers) by comparing reviewers for rest1 and rest2
    reviewers_common=set(reviewers_rest1).intersection(reviewers_rest2) 
    n_common=len(reviewers_common)  #computes the number of common reviewers (length)
    
    #gets the reviews by common reviewers found above for rest1 and rest2
    reviews_rest1=get_restaurant_reviews(rest1, df, reviewers_common) 
    reviews_rest2=get_restaurant_reviews(rest2, df, reviewers_common) 
    
    similarity=similarity_func(reviews_rest1,reviews_rest2,n_common)  #using similarity functino defined above to compute similarity
    
    #checks to see if similarity is NaN and if true, sets similarity to zero
    if np.isnan(similarity): 
        similarity=0
        
    return (similarity,n_common)

def shrunk_sim(sim, n_common, reg=3.):
    "takes a similarity and shrinks it down by using the regularizer"
    ssim=(n_common*sim)/(n_common+reg)
    return ssim
	
def knearest(restaurant_id,set_of_restaurants,dbase,k,reg):  
    
    set_of_restaurants=set_of_restaurants[set_of_restaurants!=restaurant_id]  #takes out the restaurant id provided from the set
    
    k_list=[]
    
    for rest_id in set_of_restaurants: 
        n_common=dbase.get(restaurant_id, rest_id)[1]        #using .get on instance of database class, gets common users
        sim=dbase.get(restaurant_id, rest_id)[0]            #using .get on instane of database class, gets similarity
        shrunk_sim_score=shrunk_sim(sim, n_common, reg=reg)    #applies shrunk_sim function to get new similarity after applying reg
        k_list.append((rest_id,shrunk_sim_score,n_common))       #appends the rest id, sim, and common users as a tuple in list
        
    k_list.sort(key=operator.itemgetter(1),reverse=True)      #sorts the list using shrunk sim
    
    if k is None:k=7      #if k is not provided, default is set to 7
        
    return k_list[:k]
	
def biznamefromid(df, theid):
    return df['biz_name'][df['ProductId']==theid].values[0]
def usernamefromid(df, theid):
    return df['ProfileName'][df['UserId']==theid].values[0]

	
def get_user_top_choices(UserId, df, numchoices=5):
    "get the sorted top 5 restaurants for a user by the star rating the user gave them"
    udf=df[df.UserId==UserId][['ProductId','Score']].sort_values(['Score'], ascending=False).head(numchoices)
    return udf
	
def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)



fulldf=pd.read_csv("C:/Users/Devika/Desktop/DAProject/Reviews.csv")
print("read review.csv")
fulldf=recompute_frame(fulldf)

#computing a subset from large dataframe where number of business reviews are more than 150 and user reviews more than 60
smalldf=fulldf[(fulldf.product_review_count>150) & (fulldf.user_review_count>60)]  
smalldf=recompute_frame(smalldf)   #usign the recompute function provided above to re-evaluate the average in smalldf
smalldf_unique_users=np.unique(smalldf.UserId).size   #getting number of unique users in new df 
smalldf_items=smalldf.shape[0]     #getting nuber of entries (rows) in new df

print('Items: ',smalldf_items)
print('Unique Users: ',smalldf_unique_users)

print(smalldf.head(100))
'''
db=Database(smalldf)
db.populate_by_calculating(pearson_sim)
save_object(db, 'simProducts.pkl')
'''
with open('C:/Users/Devika/Desktop/DAProject/simProducts.pkl', 'rb') as input:
    db= pickle.load(input)
print("got the db")
print(db.get("B003VXL0V6", "B006N3IG4K"))


testbizid="B006N3IG4K"
testbizid2="B000G6RYNE"

tops=knearest(testbizid, smalldf.ProductId.unique(), db, k=7, reg=3.)
print("For ",testbizid, ", top matches are:")
for i, (testbizid, sim, nc) in enumerate(tops):
    print(i,testbizid, "| Sim", sim, "| Support",nc)

testuserid="A2OEUROGZDTXUJ"
print("For user", usernamefromid(smalldf,testuserid), "top choices are:" )
bizs=get_user_top_choices(testuserid, smalldf)['ProductId'].values
print(bizs)