# iRev: Review Aware Recommender Systems Framework

The **iRev** framework is designed for creating recommendation systems that leverage user reviews to enhance the recommendation process. By incorporating user comments, iRev improves the understanding of user preferences and item characteristics.

## Repository Structure

- **Algorithm Configurations**: Define configurations and hyperparameters for various recommendation models.
- **Preprocessing and Dataset Preparation**: Contains scripts for cleaning, processing, and preparing the input datasets.
- **Models**: Implementations of recommendation models that use review-based data.
- **Metrics**: Provides evaluation metrics to measure the performance of the models.
- **Results Module**: Centralizes analysis and visualization of the results generated by the models.

------

## Taxonomy of RARS:

iRev categorizes Review-Aware Recommender Systems (RARS) into three main approaches:

1. **Document Modeling**: Concatenates all user comments, including those on items, into a single document representation.
2. **Sentence Modeling**: Focuses on identifying key words and sentences to form a structured representation of users and items.
3. **Rating Aggregation**: Associates numerical ratings with respective comments, similar to traditional recommendation systems using collaborative filtering.

**Sentiment Extraction**: Articles that apply sentiment analysis techniques to extract information from user comments.
**Aspect Extraction**: Articles that propose methods for identifying latent or implicit aspects within textual comments.

**Prediction Systems**: Focus on predicting user preferences or ratings.
**Explanation Systems**: Aim to explain recommendations based on user comments.
**Hybrid Systems**: Combine both prediction and explanation approaches.

**Non-Neural Architectures**: Utilize traditional learning techniques for recommendation, including methods like matrix factorization, clustering algorithms, and graph-based approaches.
**Neural Architectures**: Employ artificial neural network-based methods, such as convolutional techniques, attention mechanisms, recurrent networks, and graph neural networks.

**Content-Based**: Recommendations based on the attributes of items.
**Collaborative Filters**: Recommendations based on user interactions and similarities.
**Hybrid**: Combines both content-based and collaborative filtering strategies.

------

## 1. Installation:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/repository-name.git
   cd repository-name
2. Virtual Environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   pip install -r requirements.txt

3. Preprocessing:
   Modify ssh to point to your preferred database.

   ```bash
   cd preprocess_data
   bash preprocess.sh

5. Training:
   Put the name of the databases that were pre-processed in the bash file.
   Choose the models to train.
   Choose hyper parameters.

   ```bash
   bash train.sh

7. Testing:
   Put the name of the databases that were pre-processed in the bash file.
   Choose the models trained.
   Choose hyper parameters.

   ```bash
   bash test.sh

8. Analyse results:

   A csv file will be generated with the results of all metrics in the results folder. Just load it with pandas and analyze the results.

--------

All mapped and categorized algorithms can be checked in the following google sheets:

https://docs.google.com/spreadsheets/d/1WAp7J67QqoRB2wdJCdTTi3SGcYmafY6krVI2cw9sydw/edit?gid=0#gid=0
