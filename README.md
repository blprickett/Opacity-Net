# Opacity-Net

* Neural network designed to learn bleeding, feeding, counterbleeding, and counterfeeding phonological interactions. For more information about the model and the motivation behind it, see [this manuscript](https://brandon-prickett.com/wp-content/uploads/2021/07/Opacity-Manuscript-Prickett-2019.pdf) or [the published paper in *Phonology*](https://www.cambridge.org/core/journals/phonology/article/abs/learning-biases-in-opaque-interactions/A912635748ED114CE2974E8A8E2CAF5E).

* All of the training data is hardcoded into the python file, so no input files are necessary.

* Two kinds of output files will be produced: 

  * A file with learning curves for each repetition of each language, averaged across all data.

  * A file with learning curves for each repetition of each language, but averaged across the four kinds of data present in training (faithful, deleting, palatalizing, and interacting).

* You'll obviously need all of the packages I import at the beginning of the program (and their dependencies):

  * [Keras](https://keras.io/)
  * The "Seq2Seq.py" script included in this repository
  * [numpy](http://www.numpy.org/)
  * [random](https://docs.python.org/2/library/random.html)
  * [sys](https://docs.python.org/2/library/sys.html)

(Note that a previous version of this script used [recurrentshop](https://github.com/farizrahman4u/recurrentshop) and [seq2seq](https://github.com/farizrahman4u/seq2seq), however those have been replaced by the custom Seq2Seq.py. If you get slightly different results from what I report in the paper, it's likely due to this difference.)

* The only inline arguments needed are the number of repetitions, the amount of epochs per repetition, and the feature type (either "byHand" or "oneHot"; the results from the paper use "byHand").

  * So to run these simulations, run the following command from the "Seq2Seq_Simulation" directory:

    ```bash
            python Opacity_Seq2Seq.py REPS EPOCHS FEATURETYPE 
    ```

* Then, if you want to create the plots that I used in the paper, run the R script "MakePlots_Seq2Seq.R" with the "Curves by trial type (NN-ForcedChoice).csv" file in the working directory you're using.
