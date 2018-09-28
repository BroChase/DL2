# Chase Brown
# SID 106015389
# DeepLearning PA 2: ANN/CNN using Keras

# MiscFunctions:plot_history, plot_confusion_matrix, model_summary supplied by Professor Ashis Biswas
import numpy as np
import matplotlib.pyplot as plt
import itertools
import cv2
from keras.utils.vis_utils import plot_model
from sklearn.metrics import classification_report, confusion_matrix


class MiscFunctions:
    # Plot the model fit history
    @staticmethod
    def plot_history(history, model_type):
        loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
        val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
        acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
        val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]

        if len(loss_list) == 0:
            print('Loss is missing in history')
            return

        # As loss always exists
        epochs = range(1, len(history.history[loss_list[0]]) + 1)

        # Loss
        plt.figure(1)
        for l in loss_list:
            plt.plot(epochs, history.history[l], 'b',
                     label='Training loss (' + str(str(format(history.history[l][-1], '.5f')) + ')'))
        for l in val_loss_list:
            plt.plot(epochs, history.history[l], 'g',
                     label='Validation loss (' + str(str(format(history.history[l][-1], '.5f')) + ')'))
        title = model_type + '_loss'
        plt.title(title)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        t = model_type + '_loss.png'
        plt.savefig(t)

        # Accuracy
        plt.figure(2)
        for l in acc_list:
            plt.plot(epochs, history.history[l], 'b',
                     label='Training accuracy (' + str(format(history.history[l][-1], '.5f')) + ')')
        for l in val_acc_list:
            plt.plot(epochs, history.history[l], 'g',
                     label='Validation accuracy (' + str(format(history.history[l][-1], '.5f')) + ')')
        title = model_type + '_Accuracy'
        plt.title(title)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        t = model_type + '_accuracy.png'
        plt.savefig(t)
        plt.show()

    @staticmethod
    def plot_confusion_matrix(cm, model_type, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        :param cm: confusion matrix
        :param classes: class_names 0-9
        :param normalize: normalize true/false
        :param title: Title of plot :type: String
        :param cmap: color map
        :param model_type: nn model type
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix\n============================")
            t = model_type + '_norm_cfm.png'
        else:
            print('Confusion matrix, without normalization\n============================')
            t = model_type + '_cfm.png'

        print(cm)
        print("\n")

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.savefig(t)

    @staticmethod
    def model_summary(model, model_type):
        """
        Print the model summary
        :param model: nn model
        :param model_type: name of file for summary to be saved as :type: string 'nn.png'
        """
        print(model.summary())
        plot_model(model, to_file=model_type, show_shapes=True, show_layer_names=True)
        im = cv2.imread(model_type)
        height, width, channels = im.shape
        print("Height = %d, Width = %d, Channels = %d" % (height, width, channels))
        plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        plt.show()

    @staticmethod
    def final_eval(model, x_test, y_test, history, class_names, model_type):
        """
        Final Evaluation of the model
        :param model: nn model
        :param x_test: Independent Attributes
        :param y_test: Real y values
        :param history: Models run history
        :param class_names: Models class names 0-9
        :param model_type: nn model type
        """
        # Baseline error and accuracy of the model
        scores = model.evaluate(x_test, y_test, verbose=0)
        print('Baseline error: %.2f' % (1 - scores[1]))
        print("Accuracy: %.2f" % scores[1])

        # Print/plot the training history
        MiscFunctions.plot_history(history, model_type)
        # Predicted values from x_test
        y_pred = model.predict_classes(x_test)
        # Original 'actual' values
        y_test_original = np.argmax(y_test, axis=1)

        # Print classification report
        print("Classification report \n=======================")
        print(classification_report(y_true=y_test_original, y_pred=y_pred))
        print("Confusion matrix \n=======================")
        print(confusion_matrix(y_true=y_test_original, y_pred=y_pred))

        # Compute confusion matrix
        cnf_matrix = confusion_matrix(y_true=y_test_original, y_pred=y_pred)
        np.set_printoptions(precision=2)

        # Plot non-normalized confusion matrix
        plt.figure()
        MiscFunctions.plot_confusion_matrix(cnf_matrix, model_type, classes=class_names,
                                            title='Confusion matrix, without normalization')

        # Plot normalized confusion matrix
        plt.figure()
        MiscFunctions.plot_confusion_matrix(cnf_matrix, model_type, classes=class_names, normalize=True,
                                            title='Normalized confusion matrix')

        plt.show()

    @staticmethod
    def reshape_x(x_array):
        new_x = []
        for row in x_array:
            img = row.reshape(28, 28)
            new_x.append(img)
        x = np.array(new_x)
        return x