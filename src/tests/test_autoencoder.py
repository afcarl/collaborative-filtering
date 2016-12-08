import tensorflow as tf

from src.Utils import preprocess
from src.untested_models.autoenc import one_hot_encode, masked_mse


def test_masked_mse(do_preprocess, x1=0, x2=0):
    with tf.Session().as_default():
        y_true = one_hot_encode(5, x1, 10, do_preprocess).reshape(1, -1)
        y_pred = one_hot_encode(5, x2, 10, do_preprocess).reshape(1, -1)

        mse_func = masked_mse(do_preprocess)
        mse = mse_func(y_true, y_pred)
        diff = x1 - x2 if not do_preprocess else preprocess(x1) - preprocess(x2)

        predicted_mse = mse.eval()
        true_mse = float(diff) ** 2

        assert predicted_mse == true_mse


if __name__ == '__main__':
    test_masked_mse(False)
    test_masked_mse(False, 1,2)
    test_masked_mse(True)
    test_masked_mse(True, 1, 2)
