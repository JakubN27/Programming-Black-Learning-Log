### Week 1 – Research

Collaborative filtering is a recommendation technique that relies entirely on user behaviour—specifically, their taste preferences—rather than on the content of the items themselves. The core assumption behind collaborative filtering is: **if two users have similar preferences, they are likely to rate or enjoy new items in a similar way**.

To represent user preferences, we typically use a **user-item matrix**. Each row represents a user, each column an item (in this case, a song), and each cell contains a value (e.g., 1–5) representing how much a user likes a particular item. Empty cells indicate unknown preferences, such as when a user hasn't listened to a song yet.

For example:

| |1|2|3|4|5|6|7|8|
|---|---|---|---|---|---|---|---|---|
|u₁|5||1|4|||2|1|
|u₂|5||3|2|5||2||
|u₃|1|4|2|5|2|||5|

To uncover patterns in this matrix, we can apply **matrix factorization**, which helps us identify hidden relationships between users and items. One such method is **Singular Value Decomposition (SVD)**. SVD factorizes the original matrix into three smaller matrices:

- `U` – user latent factor matrix
    
- `Σ` – diagonal matrix of singular values
    
- `Vᵀ` – item latent factor matrix
    

This decomposition reveals **latent features** that capture user tastes and item characteristics, and is particularly effective for **explicit rating data** (e.g., 1 to 5 stars).

To identify similar users, we can calculate **cosine similarity** between their interaction vectors. This is computed as:

Rather than simply averaging ratings from similar users, we can apply a **weighted approach** using these similarity scores, which makes predictions more personalized and accurate.

However, a significant limitation of this model is **sparsity**. In real-world datasets, there are typically **millions of users and items**, meaning the user-item matrix is overwhelmingly empty. Most users interact with only a small subset of items, which makes reliable similarity computations and factorization challenging.

An alternative to SVD for such sparse and **implicit feedback** data (e.g., liked vs. unliked songs) is **Alternating Least Squares (ALS)**. ALS factorizes the interaction matrix into two lower-dimensional matrices:

- `X` – user matrix
    
- `Y` – item matrix
    

ALS alternates between fixing `Y` and solving for `X`, then fixing `X` and solving for `Y`, using **least squares optimization**. It's especially suited for binary interaction data, and can incorporate a **confidence score** to reflect the strength of a user’s interaction with an item.




## Week 2  - Preparing Data
Python libraries appropriate for implementing these systems are surprise and sci-kit. Surprise is designed specifically for recommendation systems, having these decompositions and cosine similarity built in as feature. Sci-kit has built is KNN methods but is not directly focused on recommendation system, but more general machine learning tasks.

I first attempted to create a user-item matrix out of the user listening dataset provided by Lucy using pandas db and create a user item matrix, however the dataset contained over a million users and items, and creating a user-item matrix out of those would mean over a trillion binary fields, which wouldn't be reasonable to compute efficiently. A fix to the sparsity issue is a tool i looked at previously, scipy. Scipy can be used to store data in sparse matrices, which store the indices of non zero elements and everything else is defaulted to 0, which is perfect for the sparse binary data i am using and massively improved efficiency. With the previous method it took 10 minutes to generate about 3000 rows of data, while with sparse matrices the whole data is processed into the matrix in just a few seconds. Now I have a complete user-item matrix with 1s for liked songs by users and 0 for unknown, which we will attempt to predict using machine learning.
## User-Item Matrix Example

| User  | Song A | Song B | Song C | Song D |
|-------|--------|--------|--------|--------|
| User 1 | 1      | 0      | 1      | 0      |
| User 2 | 0      | 1      | 0      | 1      |
| User 3 | 1      | 1      | 0      | 0      |
| User 4 | 0      | 0      | 1      | 1      |
Where user 1 has likes songs A and C, and not yet interacted/ not liked Song B and D etc.

## Week 3-4 - Developing Collaborative Filtering Algorithm 

I first approached this with a standard matrix decomposition using single value decomposition, but this was not effective for a large sparse dataset, and took an extremely long time to compute. I found "implicit" was a better library to use as it was built specifically for large sparse matrices, with much faster runtime. I stored my user- item matrix as a csr matrix which is a sparse matrix which can be operated on, and used implicit to train an alternating least squares model on it. Training this model took about 30 seconds each time which would be unreasonable to do every time the program is ran. To overcome this, i modularised my program to have a "train model" file, and the trained model is stored using the pickle library, which means it can be reused without training it each time the program is running. Using the trained model i can now query any user within the dataset, and recommend multiple songs to them based on the implicit values (0-1 with higher values meaning a better match for the user).

I now have a working recommendation using collaborative filtering, but it only works for users already within the dataset. The problem now is recommending to a user not yet in the dataset, as there is no existing listening data to work off. This is known as a 'cold start problem'. In our programs case, new users are likely to have existing listening data, which means we can add their data to our listening data matrix, and retrain the model to include their listening data. However a user with no past listening data will render collaborative filtering essentially useless, and we must rely on other techniques such as recommending by genre or simply recommending based on popularity. More complicated recommendation systems will look at more peripheral information about the user, such as location or demographic in order to try find a recommendation with similar groups of users. Others will use a hybrid filtering approach which utilises collaborative and content-based filtering which our group is doing, by developing collaborative filtering and content based filtering separately and working them together into one system.

## Week 5 - Merging Our Algorithms
During weeks 3 and 4, I developed a collaborative filtering algorithm while Waseem worked on a content-based one. This week, we’re merging them into a hybrid recommendation system. In a team meeting, we realised we had used different datasets with incompatible track IDs as mine was anonymised, and Spotify doesn't provide mass user listening data. To address this, I built a scraper to collect data from public Spotify playlists using keywords like “workout,” “chill,” and genre tags. Since playlists reflect user taste, they serve as a proxy for user preferences. I preserved original track IDs and stored the mapping in a `.pkl` file.

Now that both models use Spotify track IDs, they’re compatible. For new users without listening data, we default to recommending popular tracks. Due to the deprecation of Spotify’s audio features endpoint, content-based recommendations are limited to known songs, with collaborative filtering handling the rest. We’re evaluating the hybrid model using a weighted score from both algorithms to rank recommendations.