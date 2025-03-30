Collecting user's taste information. Disregards information on content and relies entirely on users taste in music. For example, if 2 users have similar tastes in music, they are more likely to have a more similar opinion on a song neither one of them has heard before.

Users' opinions can be represented with a matrix structure known as a user-item matrix, with users u1, u2, u3 and their opinions on 'items' (songs in this case), 1-8, ranked from 1-5 with blanks beings no opinion or haven't heard it before. This matrix can be factorised to extract similarities in user's opinions, to then know which user's tastes to use for recommendation.

|     | 1   | 2   | 3   | 4   | 5   | 6   | 7   | 8   |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| u₁  | 5   |     | 1   | 4   |     |     | 2   | 1   |
| u₂  | 5   |     | 3   | 2   | 5   |     | 2   |     |
| u₃  | 1   | 4   | 2   | 5   | 2   |     |     | 5   |
An algorithm for finding an appropriate factorisation of such a matrix is SVD (single-value decomposition). This breaks the matrix down into two matrices with lower rank, for user features. and for item features. Missing values are naively approximated as a product of the decomposition (**U** **V$^T}$**)]

Similarities in the user's tastes can be calculated using cosine similarity, which can be computed with the formula : ${Sim}(u_i, u_j) = \frac{u_i \cdot u_j}{\|u_i\| \|u_j\|} = \cos(\theta)$
Using the coefficient obtained through the cosine similarity, instead of just taking a naïve approach through averaging the other user's tastes, it can now be weighted (weighted inner product) based on each user's similarity to the user we are trying to recommend to. 

The main limitation of this model is sparsity, as the example was just a small table it had a few gaps, but in reality, there are so many users and so many items to choose from the a vast majority of a dataset will be blank and this needs to be accounted for to produce a more appropriate sample to work with.

Python libraries appropriate for implementing these systems are surprise and sci-kit. Surprise is designed specifically for recommendation systems, having these decompositions and cosine similarity built in as feature. Sci-kit has built is KNN methods but is not directly focused on recommendation system, but more general machine learning tasks.

I first attempted to create a user-item matrix out of the user listening dataset provided by Lucy using pandas db and create a user item matrix, however the dataset contained over a million users and items, and creating a user-item matrix out of those would mean over a trillion binary fields, which wouldn't be reasonable to compute efficiently. A fix to the sparsity issue is a tool i looked at previously, scipy. Scipy can be used to store data in sparse matrices, which store the indices of non zero elements and everything else is defaulted to 0, which is perfect for the sparse binary data i am using and massively improved efficiency. With the previous method it took 10 minutes to generate about 3000 rows of data, while with sparse matrices the whole data is processed into the matrix in just a few seconds. Now I have a complete user-item matrix with 1s for liked songs by users and 0 for unknown, which we will attempt to predict using machine learning.
