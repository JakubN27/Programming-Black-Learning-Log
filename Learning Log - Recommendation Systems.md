### Week 1 – Research
### Goals 
 - Plan the project and the knowledge I need to start
 - Learn about recommendation systems and the techniques I will use
### What I did
We reated a six week plan for he project and divided responsibilities among my team. Then I started to research my part of the project which mainly focused on collaborative filtering. After my research I decided on a suitable approach I should take to begin developing my algorithm.
### What I learned

Collaborative filtering is a recommendation technique that relies entirely on user's taste preferences rather than on the content of the items themselves. The core assumption behind collaborative filtering is: if two users have similar preferences, they are likely to rate or enjoy new items in a similar way.

To represent user preferences, we typically use a user-item matrix. Each row represents a user, each column an item (in this case, a song), and each cell contains a value (e.g., 1–5) representing how much a user likes a particular item. Empty cells indicate unknown preferences, such as when a user hasn't listened to a song yet.

For example:

| |1|2|3|4|5|6|7|8|
|---|---|---|---|---|---|---|---|---|
|u₁|5||1|4|||2|1|
|u₂|5||3|2|5||2||
|u₃|1|4|2|5|2|||5|

To uncover patterns in this matrix, we can apply matrix factorisation, which helps us identify hidden relationships between users and items. One such method is Singular Value Decomposition (SVD). SVD factorises the original matrix into three smaller matrices:

- U – user latent factor matrix
- `Σ` – diagonal matrix of singular values
- `Vᵀ` – item latent factor matrix

This decomposition reveals latent features that capture user tastes and item characteristics, and is good for explicit rating data (e.g., 1 to 5 stars).

To identify similar users, we can calculate cosine similarity between their interaction vectors. This is computed as:
$${cosine_similarity}(\mathbf{A}, \mathbf{B}) = \frac{\mathbf{A} \cdot \mathbf{B}}{|\mathbf{A}| |\mathbf{B}|} = \frac{\sum_{i=1}^n A_i B_i}{\sqrt{\sum_{i=1}^n A_i^2} \sqrt{\sum_{i=1}^n B_i^2}}$$

However, a significant limitation of this model is sparsity. In real-world datasets, there are typically millions of users and items, meaning the user-item matrix is overwhelmingly empty. Most users interact with only a small subset of items, which makes reliable similarity computations and factorisation challenging.

An alternative to SVD for such sparse and implicit feedback data (e.g, liked vs. unliked songs) is Alternating Least Squares (ALS). ALS factorises the interaction matrix into two lower-dimensional matrices:

- `X` – user matrix
- `Y` – item matrix

ALS alternates between fixing Y and solving for X, then fixing X and solving for Y, using least squares optimisation. It's especially suited for binary interaction data, and can incorporate a confidence score to reflect the strength of a user’s interaction with an item.

### Next Steps
 - Plan how I can apply the data from Lucy's data analysis and cleaning and apply it to the model.
 - Prepare a user item matrix in a suitable format to train an ALS model.
 - Determine which python libraries I will use

## Week 2 - Preparing Data

### What I did
I initially tried building a user-item matrix from Lucy’s dataset using pandas, but with over a million users and items, it would have resulted in a trillion binary fields which was too inefficient to compute. I switched to using scipy's sparse matrices, which store only non-zero values, significantly improving performance. What took 10 minutes for 3000 rows now processes the full dataset in seconds. The resulting matrix uses 1s for liked songs and 0s for unknowns, ready for machine learning.

### User-Item Matrix Example

| User   | Song A | Song B | Song C | Song D |
| ------ | ------ | ------ | ------ | ------ |
| User 1 | 1      | 0      | 1      | 0      |
| User 2 | 0      | 1      | 0      | 1      |
| User 3 | 1      | 1      | 0      | 0      |
| User 4 | 0      | 0      | 1      | 1      |

Where user 1 has likes songs A and C, and not yet interacted/ not liked Song B and D etc.
### What I learned
- Pandas isn't efficient for large binary matrices.
- Scipy's sparse matrices are highly efficient for storing and processing large sparse datasets.
### Goals/ Changes to Goals
No changes to goals, I am still following my initial plan to develop a collaborative filtering model based off the data.
### Next Steps
I want to apply a suitable machine learning model (I decided on ALS) to the sparse matrix.
While Waseem and I build filtering algorithms, Lucy will work on a front end which will display data from our models.
## Week 3- Developing Collaborative Filtering Algorithm

### What I did
I used the implicit library to train an Alternating Least Squares model on the sparse matrix. Since training took around 30 seconds, I modularised the code and saved the trained model with pickle, allowing reuse without retraining. The model recommends songs to existing users based on implicit scores (0-1, with higher being better).

However, it only works for users already in the dataset and not new users (known as a cold start problem). We can retrain the model if they have past listening data. If not, we fall back on alternatives like genre-based or popularity-based recommendations.
### What I learned
- ALS using the implicit library is far more efficient than SVD
- Pickle allows saving trained models between uses.
- Cold start problems are a major limitation in collaborative filtering, which is why a hybrid approach is required
- More complicated recommendation systems will look at more peripheral information about the user, such as location or demographic in order to try find a recommendation with similar groups of users.
### Goals/Changes to Goals
- Goal achieved: I have a basic working collaborative filtering system.
- New goal: I had to find a way to overcome cold start problems.
### Next Steps
Use a hybrid filtering approach which utilises collaborative and content-based filtering which our group is doing by developing collaborative filtering and content based filtering separately and working them together into one system.

## Week 5 - Merging Our Algorithms

### What I did
Waseem developed content-based filtering while I worked on collaborative filtering. In a team meeting we realised we used incompatible datasets (mine anonymised, his not), so I built a scraper to gather playlist data and mapped original Spotify track IDs in a JSON file. While playlists reflect taste, they introduce bias as they're made for curated tastes.

For new users, we recommend popular tracks by default. With Spotify’s audio features deprecated, content-based filtering is limited to known songs. Our hybrid model ranks recommendations using a weighted score from both algorithms. New users get content-based suggestions first and are added to the dataset for more personalised results over time.
### What I learned
- Playlist data introduces bias but is useful in the absence of real user histories.
- Better planning of what data is used and how it is used is crucial for these kinds of projects.
- A hybrid model can be progressively personalised by adding unseen data to the dataset to improve recommendations over time,
### Goals/ Changes to Goals
- Instead of using the original dataset I had to simulate by scraping playlists.
### Next Steps
Connect the hybrid filtering algorithm to the frontend Lucy is building.

## Week 6 - Deployment

### What I did
Lucy's website runs on node js and allows the user can connect their Spotify to it. When the user logs in it fetches their top songs and their id. This was my first time using python as part of a web app so I used the child process module to run the hybrid filtering model, and returned a JSON so that the server could read it and display the recommendations on the page. Once the data was passed through correctly the recommendation system as before, and displayed correctly on the page as it was tested with sample data.
### What I learned
- We could've planned better by using a django or flask server to make it more compatible with our python programs, and allow us to use the same libraries without having to make cmd calls to run the python programs.
- JS web apps can run python programs using child processes
### Changes to Goals
We achieved our goals for the project, as we now have a hybrid filtering model which recommends songs based off a Spotify profile, and displays them on a web app.
### Next Steps
To further develop the recommendation system, I would work on increasing datasets and getting more reliable data for more accurate recommendations, since while our dataset is big, it doesn't compare to the amount of music out there, and when testing with personal profiles, a majority of top songs were not in our dataset which gave inaccurate recommendations.