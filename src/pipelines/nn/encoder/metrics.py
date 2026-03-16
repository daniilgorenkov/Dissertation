from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt


class Metrics:
    def __init__(self, logger):
        self.logger = logger

    def log_fault_pr_curve(self, metrics, epoch: int):
        y_true = metrics["y_fault"]
        y_score = metrics["fault_prob"][:, 1]  # вероятность класса "fault"

        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(recall, precision, label=f"AP = {ap:.4f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Fault PR Curve")
        ax.legend(loc="best")
        ax.grid(True)

        self.logger.report_matplotlib_figure(
            title="pr_curve_fault",
            series="val",
            iteration=epoch,
            figure=fig,
        )

        plt.close(fig)

        self.logger.report_scalar(
            title="metrics",
            series="fault_average_precision",
            iteration=epoch,
            value=float(ap),
        )

    def log_confusion_matrices(self, metrics, epoch: int):
        fault_cm = confusion_matrix(metrics["y_fault"], metrics["fault_pred"])
        profile_cm = confusion_matrix(metrics["y_profile"], metrics["profile_pred"])

        self.logger.report_confusion_matrix(
            title="confusion_matrix_fault",
            series="val",
            iteration=epoch,
            matrix=fault_cm,
            xaxis="Predicted",
            yaxis="True",
            xlabels=["no_fault", "fault"],
            ylabels=["no_fault", "fault"],
            yaxis_reversed=True,
        )

        self.logger.report_confusion_matrix(
            title="confusion_matrix_profile",
            series="val",
            iteration=epoch,
            matrix=profile_cm,
            xaxis="Predicted",
            yaxis="True",
            xlabels=["profile_0", "profile_1", "profile_2"],
            ylabels=["profile_0", "profile_1", "profile_2"],
            yaxis_reversed=True,
        )

    def log_profile_pr_curves(self, metrics, epoch: int, num_classes: int = 3):
        y_true = metrics["y_profile"]
        y_prob = metrics["profile_prob"]

        for cls in range(num_classes):
            y_true_bin = (y_true == cls).astype(int)
            y_score = y_prob[:, cls]

            precision, recall, _ = precision_recall_curve(y_true_bin, y_score)
            ap = average_precision_score(y_true_bin, y_score)

            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(recall, precision, label=f"class {cls}, AP = {ap:.4f}")
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title(f"Profile PR Curve OvR - class {cls}")
            ax.legend(loc="best")
            ax.grid(True)

            self.logger.report_matplotlib_figure(
                title="pr_curve_profile_ovr",
                series=f"class_{cls}",
                iteration=epoch,
                figure=fig,
            )
            plt.close(fig)

            self.logger.report_scalar(
                title="metrics",
                series=f"profile_class_{cls}_average_precision",
                iteration=epoch,
                value=float(ap),
            )
