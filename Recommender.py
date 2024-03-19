# Open New google colab. Write these codes on google colab, then run them.

!pip install wget

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import wget

# Download MovieLens from shared google drive. It consists of information about Movies.
file_url = 'https://drive.google.com/uc?id=1pF8IKUnS9r8Tb8KRTdV65kP2WGYJONOA'

file_name = 'moviemetadata.csv'

# Download the file
wget.download(file_url, file_name)

metadata = pd.read_csv('moviemetadata.csv', low_memory=False)


#  Define a TF-IDF Vectorizer Object. Remove all english stop words such
#  as 'the', 'a'

tfidf = TfidfVectorizer(stop_words='english')
#Replace NaN with an empty string
metadata['overview'] = metadata['overview'].fillna('')

#  Construct the required TF-IDF matrix by fitting and transforming the
#  data
tfidf_matrix = tfidf.fit_transform(metadata['overview'])

#Output the shape of tfidf_matrix
tfidf_matrix.shape

# Get feature names after fitting and transforming
feature_names = tfidf.get_feature_names_out()

# Compute the cosine similarity matrix
# Since it needs a lot of RAM I cut the length of tfidf_matrix to 27466. If you don't have RAM limit use this code:
# cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix) instead of the following code. It gives better results!
cosine_sim = linear_kernel(tfidf_matrix[:27466], tfidf_matrix[:27466])

#Construct a reverse map of indices and movie titles
indices = pd.Series(metadata.index, index=metadata\
                                             ['title']).drop_duplicates()

# Function that takes in movie title as input and outputs most similar movies
def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return metadata['title'].iloc[movie_indices]

print('Movies similar to The Godfather are: \n',get_recommendations('The Godfather'))
print('Movies similar to The Dark Knight Rises are: \n',get_recommendations('The Dark Knight Rises'))
