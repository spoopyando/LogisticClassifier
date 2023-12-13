# LogisticClassifier
A Python script for training a classifier to predict the subreddit from a given comment using preloaded word vectors.

How it works:

    The program sets a model using the word vectors as a corpus.

    The classifier is trained using reddit comments found in a .jsonlist file.
        Note: The default .jsonlist file is provided as "redditComments_train.jsonlist",
              where each json object contains a "body" (which is a subreddit comment) and a
              "subreddit" (the name of a subreddit where that comment is from.)

    The classifier is then tested on sample sentences

IMPORTANT: 

    The default pre-trained word vector can be found and downloaded from "https://github.com/mmihaltz/word2vec-GoogleNews-vectors".
    Extract it to the path where the script is located.

    This word vector is used as a corpus and must be included for the script to work as intended.

    There's a lot of data to process when using the word vectors and evaluating a large jsonlist!
    As such, the time it takes for the script to run may take a few minutes, depending on the vectors, your jsonlist, and your machine.