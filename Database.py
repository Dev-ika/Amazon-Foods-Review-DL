import  pandas  as  pd
import numpy as np

class Database:
    "A class representing a database of similaries and common supports"

    def __init__(self, df):
        "the constructor, takes a reviews dataframe like smalldf as its argument"
        database={}
        self.df=df
        self.uniquebizids={v:k for (k,v) in enumerate(df.ProductId.unique())}
        keys=self.uniquebizids.keys()
        l_keys=len(keys)
        self.database_sim=np.zeros([l_keys,l_keys])

        self.database_sup=np.zeros([l_keys, l_keys], dtype=np.int)
		
    def get_restaurant_reviews(self,restaurant_id, df, set_of_users):
        """
        given a resturant id and a set of reviewers, return the sub-dataframe of their
        reviews.
        """
        mask = (df.UserId.isin(set_of_users)) & (df.ProductId==restaurant_id)
        reviews = df[mask]
        reviews = reviews[reviews.UserId.duplicated()==False]
        return reviews
		
    def calculate_similarity(self,rest1,rest2,df,similarity_func):     #calculates similarity between 2 restaurants using some similarity function

        reviewers_rest1=df[df.ProductId==rest1].UserId   #gets all the user ids from df for rest1
        reviewers_rest2=df[df.ProductId==rest2].UserId   #gets all the user ids from df for rest2

        #computes the intersection (common reviewers) by comparing reviewers for rest1 and rest2
        reviewers_common=set(reviewers_rest1).intersection(reviewers_rest2) 
        n_common=len(reviewers_common)  #computes the number of common reviewers (length)

        #gets the reviews by common reviewers found above for rest1 and rest2
        reviews_rest1=self.get_restaurant_reviews(rest1, df, reviewers_common) 
        reviews_rest2=self.get_restaurant_reviews(rest2, df, reviewers_common) 

        similarity=similarity_func(reviews_rest1,reviews_rest2,n_common)  #using similarity functino defined above to compute similarity

        #checks to see if similarity is NaN and if true, sets similarity to zero
        if np.isnan(similarity): 
            similarity=0

        return (similarity,n_common)              

    def populate_by_calculating(self, similarity_func):
        """
        a populator for every pair of businesses in df. takes similarity_func like
        pearson_sim as argument
        """
        items=self.uniquebizids.items()
        for b1, i1 in items:
            for b2, i2 in items:
                if i1 < i2:
                    sim, nsup=self.calculate_similarity(b1, b2, self.df, similarity_func)
                    self.database_sim[i1][i2]=sim
                    self.database_sim[i2][i1]=sim
                    self.database_sup[i1][i2]=nsup
                    self.database_sup[i2][i1]=nsup
                elif i1==i2:
                    nsup=self.df[self.df.ProductId==b1].UserId.count()
                    self.database_sim[i1][i1]=1.
                    self.database_sup[i1][i1]=nsup
                                    
    
    def get(self, b1, b2):
        "returns a tuple of similarity,common_support given two business ids"
        sim=self.database_sim[self.uniquebizids[b1]][self.uniquebizids[b2]]
        nsup=self.database_sup[self.uniquebizids[b1]][self.uniquebizids[b2]]
        return (sim, nsup)
