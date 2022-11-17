"""Evaluate Model."""

import logging
import warnings
import seaborn as sns
import matplotlib.pyplot as plt


warnings.filterwarnings("ignore")
logger = logging.getLogger("main_logger")


class Evaluate:
    """Class to evaluate model and visualize."""

    def __init__(self, conf, history):
        self.conf = conf
        self.history = history

    def learning_curves(self):
        """Plot the learning curves of loss and macro f1 score
        for the training and validation datasets.

        Args:
            history: history callback of fitting a tensorflow keras model
        Returns:
            Loss and F1 Score for train and validation.
            Saves a plot of learning curves.
        """
        logger.info("Evaluating the model...")
        loss = self.history.history["loss"]
        val_loss = self.history.history["val_loss"]

        macro_f1 = self.history.history["macro_f1"]
        val_macro_f1 = self.history.history["val_macro_f1"]

        path = (
            self.conf["paths"]["Outputs_path"] +
            self.conf["paths"]["folder_evaluation"]
        )

        # Use plot styling from seaborn.
        sns.set(style="darkgrid")
        # Increase the plot size and font size.
        sns.set(font_scale=1.5)
        plt.rcParams["figure.figsize"] = (12, 6)
        # Plot the learning curve.
        plt.plot(loss, "b-o", label="training loss")
        plt.plot(val_loss, "r-o", label="validation loss")
        # Label the plot.
        plt.title("Training and Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(path + "loss.jpg")
        plt.show()
        plt.clf()

        # Use plot styling from seaborn.
        sns.set(style="darkgrid")
        # Increase the plot size and font size.
        sns.set(font_scale=1.5)
        plt.rcParams["figure.figsize"] = (12, 6)
        # Plot the learning curve.
        plt.plot(macro_f1, "b-o", label="F1 score")
        plt.plot(val_macro_f1, "r-o", label="validation F1 score")
        # Label the plot.
        plt.title("Training and Validation F1-score")
        plt.xlabel("Epoch")
        plt.ylabel("F1 score")
        plt.legend()
        plt.savefig(path + "validation.jpg")
        plt.show()
        plt.clf
        logger.info(f"Saving Evaluation curve to:{path}")
        return loss, val_loss, macro_f1, val_macro_f1
