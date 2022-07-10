import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import stats
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

from makeInputTensor import makeInputTensor


def gru(units):
    if tf.test.is_gpu_available():
        return tf.keras.layers.CuDNNGRU(units,
                                        return_sequences=True,
                                        return_state=True,
                                        recurrent_initializer='glorot_uniform')
    else:
        return tf.keras.layers.GRU(units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_activation='relu',
                                   recurrent_initializer='glorot_uniform')


class EmoGRU(tf.keras.Model):
    def __init__(self, time_size, embedding_dim, hidden_units, batch_sz, output_size):
        super(EmoGRU, self).__init__()
        self.batch_sz = batch_sz
        self.hidden_units = hidden_units
        # --- layers ---
        self.embedding = tf.keras.layers.Embedding(time_size, embedding_dim)
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.gru = gru(self.hidden_units)
        self.fc = tf.keras.layers.Dense(output_size)

    def get_config(self):
        config = super(EmoGRU, self).get_config
        return config

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        out = output[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.hidden_units))


def loss_function(y, prediction):
    return tf.compat.v1.losses.softmax_cross_entropy(y, logits=prediction)


def accuracy(y, yhat):
    yhat = tf.argmax(yhat, 1).numpy()
    y = tf.argmax(y, 1).numpy()
    return np.sum(y == yhat) / len(y)


class Evaluate:
    def va_dist(self, prediction, target, va_df, binarizer, name='', silent=False):
        """ Computes distance between actual and prediction through cosine distance """
        va_matrix = va_df.loc[binarizer.classes_][['valence', 'arousal']].values
        y_va = target.dot(va_matrix)
        F_va = prediction.dot(va_matrix)

        # dist is a one row vector with size of the test data passed(emotion)
        dist = metrics.pairwise.paired_cosine_distances(y_va, F_va)
        res = stats.describe(dist)

        # print by default (if silent=False)
        if not silent:
            print('%s\tmean: %f\tvariance: %f' % (name, res.mean, res.variance))

        return {
            'distances': dist,
            'dist_stat': res
        }

    def evaluate_class(self, predictions, target, target2=None, silent=False):
        """ Compute only the predicted class """
        p_2_annotation = dict()

        precision_recall_fscore_support = [
            (pair[0], pair[1].mean()) for pair in zip(
                ['precision', 'recall', 'f1', 'support'],
                metrics.precision_recall_fscore_support(target, predictions)
            )
        ]

        metrics.precision_recall_fscore_support(target, predictions)

        # confusion matrix
        le = LabelEncoder()
        target_le = le.fit_transform(target)
        predictions_le = le.transform(predictions)
        cm = metrics.confusion_matrix(target_le, predictions_le)

        # prediction if two annotations are given on test data
        if target2:
            p_2_annotation = pd.DataFrame(
                [(pred, pred in set([t1, t2])) for pred, t1, t2 in zip(predictions, target, target2)],
                columns=['emo', 'success']
            ).groupby('emo').apply(lambda emo: emo.success.sum() / len(emo.success)).to_dict()

        if not silent:
            print("Default Classification report")
            print(metrics.classification_report(target, predictions))

            # print if target2 was provided
            if len(p_2_annotation) > 0:
                print('\nPrecision on 2 annotations:')
                for emo in p_2_annotation:
                    print("%s: %.2f" % (emo, p_2_annotation[emo]))

            # print accuracies, precision, recall, and f1
            print('\nAccuracy:')
            print(metrics.accuracy_score(target, predictions))
            print("Correct Predictions: ", metrics.accuracy_score(target, predictions, normalize=False))
            for to_print in precision_recall_fscore_support[:3]:
                print("%s: %.2f" % to_print)

            # normalizing the values of the consfusion matrix
            print('\nconfusion matrix\n %s' % cm)
            print('(row=expected, col=predicted)')
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            self.plot_confusion_matrix(cm_normalized, le.classes_, 'Confusion matrix Normalized')

        return {
            'precision_recall_fscore_support': precision_recall_fscore_support,
            'accuracy': metrics.accuracy_score(target, predictions),
            'p_2_annotation': p_2_annotation,
            'confusion_matrix': cm
        }

    def predict_class(self, X_train, y_train, X_test, y_test,
                      pipeline, silent=False, target2=None):
        """ Predicted class,then run some performance evaluation """
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        print("predictions computed....")
        return self.evaluate_class(predictions, y_test, target2, silent)

    def evaluate_prob(self, prediction, target_rank, target_class, binarizer, va_df, silent=False, target2=None):
        """ Evaluate through probability """
        # Run normal class evaluator
        predict_class = binarizer.classes_[prediction.argmax(axis=1)]
        class_eval = self.evaluate_class(predict_class, target_class, target2, silent)

        if not silent:
            print('\n - First Emotion Classification Metrics -')
            print('\n - Multiple Emotion rank Metrics -')
            print('VA Cosine Distance')

        classes_dist = [
            (
                emo,
                self.va_dist(
                    prediction[np.array(target_class) == emo],
                    target_rank[np.array(target_class) == emo],
                    va_df,
                    binarizer,
                    emo,
                    silent)
            ) for emo in binarizer.classes_
        ]
        avg_dist = self.va_dist(prediction, target_rank, va_df, binarizer, 'avg', silent)

        coverage_error = metrics.coverage_error(target_rank, prediction)
        average_precision_score = metrics.average_precision_score(target_rank, prediction)
        label_ranking_average_precision_score = metrics.label_ranking_average_precision_score(target_rank, prediction)
        label_ranking_loss = metrics.label_ranking_loss(target_rank, prediction)

        # recall at 2
        # obtain top two predictions
        top2_pred = [set([binarizer.classes_[i[0]], binarizer.classes_[i[1]]]) for i in
                     prediction.argsort(axis=1).T[-2:].T]
        recall_at_2 = pd.DataFrame(
            [
                t in p for t, p in zip(target_class, top2_pred)
            ], index=target_class, columns=['recall@2']).groupby(level=0).apply(lambda emo: emo.sum() / len(emo))

        # combine target into sets
        if target2:
            union_target = [set(t) for t in zip(target_class, target2)]
        else:
            union_target = [set(t) for t in zip(target_class)]

        # precision at k
        top_k_pred = [
            [set([binarizer.classes_[i] for i in i_list]) for i_list in prediction.argsort(axis=1).T[-i:].T]
            for i in range(2, len(binarizer.classes_) + 1)]
        precision_at_k = [
            ('p@' + str(k + 2), np.array([len(t & p) / (k + 2) for t, p in zip(union_target, top_k_pred[k])]).mean())
            for k in range(len(top_k_pred))]

        # do this if silent= False
        if not silent:
            print('\n')
            print(recall_at_2)
            print('\n')
            print('p@k')
            for pk in precision_at_k:
                print(pk[0] + ':\t' + str(pk[1]))
            print('\ncoverage_error: %f' % coverage_error)
            print('average_precision_score: %f' % average_precision_score)
            print('label_ranking_average_precision_score: %f' % label_ranking_average_precision_score)
            print('label_ranking_loss: %f' % label_ranking_loss)

        return {
            'class_eval': class_eval,
            'recall_at_2': recall_at_2.to_dict(),
            'precision_at_2': precision_at_k,
            'classes_dist': classes_dist,
            'avg_dist': avg_dist,
            'coverage_error': coverage_error,
            'average_precision_score': average_precision_score,
            'label_ranking_average_precision_score': label_ranking_average_precision_score,
            'label_ranking_loss': label_ranking_loss
        }

    def predict_prob(self, X_train, y_train, X_test, y_test, label_test, pipeline, binarizer, va_df, silent=False,
                     target2=None):
        """ Output predictions based on training and labels """
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict_proba(X_test)
        pred_to_mlb = [np.where(pipeline.classes_ == emo)[0][0] for emo in binarizer.classes_.tolist()]
        return self.evaluate_prob(predictions[:, pred_to_mlb], y_test, label_test, binarizer, va_df, silent, target2)

    def plot_confusion_matrix(self, cm, my_tags, title='Confusion matrix', cmap=plt.cm.Blues):
        """ Plotting the confusion_matrix """
        plt.rc('figure', figsize=(4, 4), dpi=100)
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(my_tags))
        target_names = my_tags
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

        # add normalized values inside the Confusion matrix
        fmt = '.2f'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')


def main():
    mode = "test"
    dir_path = "./Physiological"
    model_name = "best"
    data_type = "EDA"
    data_input, data_label = makeInputTensor(mode, dir_path, data_type)

    DATA_BUFFER_SIZE = len(data_input)
    BATCH_SIZE = 64
    DATA_N_BATCH = DATA_BUFFER_SIZE // BATCH_SIZE
    embedding_dim = 256
    units = 1024
    timeline = data_input.shape[1]
    target_size = data_label.shape[1]

    model = EmoGRU(timeline, embedding_dim, units, BATCH_SIZE, target_size)

    input_dataset = tf.data.Dataset.from_tensor_slices((data_input, data_label)).shuffle(DATA_BUFFER_SIZE)
    input_dataset = input_dataset.batch(BATCH_SIZE, drop_remainder=True)

    model.built = True
    model.load_weights(model_name + '.h5')
    test_accuracy = 0
    all_predictions = []
    x_raw = []
    y_raw = []
    hidden = model.initialize_hidden_state()
    for (batch, (inp, targ)) in enumerate(input_dataset):
        predictions, _ = model(inp, hidden)
        batch_accuracy = accuracy(targ, predictions)
        test_accuracy += batch_accuracy
        x_raw = x_raw + [x for x in inp]
        y_raw = y_raw + [y for y in targ]
        all_predictions.append(predictions)
    print("Test Accuracy: ", test_accuracy / DATA_N_BATCH)

    evaluator = Evaluate()
    final_predictions = []
    for p in all_predictions:
        for sub_p in p:
            final_predictions.append(sub_p)
    predictions = [np.argmax(p).item() for p in final_predictions]
    targets = [np.argmax(t).item() for t in y_raw]
    correct_predictions = float(np.sum(predictions == targets))
    predictions_human_readable = ((x_raw, predictions))
    target_human_readable = ((x_raw, targets))
    emotion_dict = {0: 'Happy',
                    1: 'Surprise',
                    2: 'Sadness',
                    3: 'Startle',
                    4: 'Skeptical',
                    5: 'Embarrassment',
                    6: 'Fear',
                    7: 'Physical',
                    8: 'Angry',
                    9: 'Disgust'}
    model_test_result = pd.DataFrame(predictions_human_readable[1], columns=["emotion"])
    test = pd.DataFrame(target_human_readable[1], columns=["emotion"])
    model_test_result.emotion = model_test_result.emotion.map(lambda x: emotion_dict[int(float(x))])
    test.emotion = test.emotion.map(lambda x: emotion_dict[int(x)])
    evaluator.evaluate_class(model_test_result.emotion, test.emotion)


if __name__ == '__main__':
    main()
