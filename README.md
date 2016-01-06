#Predicting and Analyzing Reddit Comment Karma

##SEE IT LIVE [HERE](https://analyzekarma.herokuapp.com/)!

###Uses various data-science and NLP methods to analyze comment karma throughout Reddit and provide this information with visualizations and a RESTful API.

##Dependencies:
* Flask (0.9)
* Flask-Limiter (0.9.1)
* WTForms (2.1)
* gunicorn (0.17.2)
* textblob (0.11.0)
* numpy (1.10.1)
* scikit-learn (0.17)
* Python 2.7.11
* [Optional] Pickle files present in pickles folder

##Implements:
* K-means unsupervised clustering
* Linear SVM supervised classification
* RESTful API
    * Documented
    * Rate-limited
* Extensive JSON parsing
* Various facets of Natural Language Processing
* Graph visualization
    * JS (cytoscape.js)
* Various facets of web-design
    * HTML
    * CSS (Bootstrap)

Inspired by the work done [here](http://cs229.stanford.edu/proj2014/Daria%20Lamberson,Leo%20Martel,%20Simon%20Zheng,Hacking%20the%20Hivemind.pdf) and [here](http://users.wpi.edu/~hsahay/assets/PredictingRedditPostPopularity.pdf), this application utilizes two classifiers and k-means clustering to provide insights on comment karma on Reddit. The first classifier used n-grams to classify comments as either positive or negative, and the second, which uses the classification from the first as one of its features, classifies comments into one of five bins based on score ranges based on various metadata and calculated NLP-related features. The application also, independent from the aforementioned tasks, parsed the aggregate data to cluster comments by average karma per comment, and created a clustered graph visualization of the top subreddits in the data-set. Lastly, the application implemented a fully-documented and rate-limited RESTful API to allow developers to use the prediction service in their own applications.
