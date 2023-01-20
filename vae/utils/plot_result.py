import os
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score


def plot_learning_curve(iteration_list,
                        train_loss_teacher_forcing,
                        train_loss_no_teacher_forcing,
                        valid_loss_no_teacher_forcing,
                        save_directory,
                        title: str):

    plt.figure(figsize=(8, 8), dpi=300)
    plt.plot(iteration_list, train_loss_teacher_forcing, 'b-', label='Train Loss (teacher forcing)')
    plt.plot(iteration_list, train_loss_no_teacher_forcing, 'g-', label='Train Loss (no teacher forcing)')
    plt.plot(iteration_list, valid_loss_no_teacher_forcing, 'r-', label='Validation Loss (no teacher forcing)')

    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("%s" % title, fontsize=12)

    plt.legend()
    plt.savefig(os.path.join(save_directory, "%s_learning_curve.png" % title))
    plt.show()
    plt.close()


def plot_regression(y_train, y_train_pred, y_test, y_test_pred, save_directory, train_test=True):
    if train_test:
        # compute score
        train_r2_score = r2_score(y_train, y_train_pred)
        test_r2_score = r2_score(y_test, y_test_pred)

        # plot the result
        plt.figure(figsize=(8, 8))
        x_min = min([min(y_train), min(y_train_pred), min(y_test), min(y_test_pred)])
        x_max = max([max(y_train), max(y_train_pred), max(y_test), max(y_test_pred)])

        plt.plot(y_train, y_train_pred, 'bo', label='Train R2 %.3f' % train_r2_score)
        plt.plot(y_test, y_test_pred, 'ro', label='Test R2 %.3f' % test_r2_score)
        plt.plot((x_min, x_max), (x_min, x_max), 'k--', alpha=0.3)

        plt.xlabel('True', fontsize=12)
        plt.ylabel('Prediction', fontsize=12)
        plt.title('Regression', fontsize=12)

        plt.legend(fontsize=12)
        plt.savefig(os.path.join(save_directory, 'regression.png'))
        plt.show()
        plt.close()

    else:
        score = r2_score(y_train, y_train_pred)

        # plot the result
        plt.figure(figsize=(8, 8))
        x_min = min([min(y_train), min(y_train_pred)])
        x_max = max([max(y_train), max(y_train_pred)])

        plt.plot(y_train, y_train_pred, 'bo', label='R2 %.3f' % score)
        plt.plot((x_min, x_max), (x_min, x_max), 'k--', alpha=0.3)

        plt.xlabel('True', fontsize=12)
        plt.ylabel('Prediction', fontsize=12)
        plt.title('Regression', fontsize=12)

        plt.legend(fontsize=12)
        plt.savefig(os.path.join(save_directory, 'regression.png'))
        plt.show()
        plt.close()
