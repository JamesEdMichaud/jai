import numpy as np
import tensorflow as tf
# Disable GPU on Apple architecture. It's slower sometimes.
# tf.config.set_visible_devices([], 'GPU')
from utils import JaiUtils
from models import JaiNN, JaiLR, JaiSVM

# print("TensorFlow version: ", tf.__version__)


random_seed = 638
weights_random_seed = 438

# Set some utils defaults
utils = JaiUtils(
    vid_path="training_data",
    img_size=(64, 64),
    max_seq_len=40,
    train_split=0.9,
    val_split=0.2,
    rand_seed=random_seed,
    weights_seed=weights_random_seed,
    training_data_updated=False,
    epochs=300,
    learning_rate=0.001,   # LogReg: 0.0025   , SVM: 0.0006
    l2_reg=0.00,           # LogReg: No effect, SVM: No effect?
    batch_size=64,
)

data = utils.get_data()

print(f"Training data shape: {data['train_data'].shape}")
print(f"Test data shape: {data['test_data'].shape}")

# model = JaiLR(utils)
# model = JaiSVM(utils)
# model = JaiNN(utils)

# Override the utils defaults here
common_args = {'data': data, 'epochs': 300}

# Set parameter ranges for tuning
lr_lr_args = {
    'method_to_call': utils.learning_rate_tuning_curve,
    'param_range': [0, 101, 1],
    'param_factor': 0.0001,
    'l2reg': 0.0,
}
lr_reg_args = {
    'method_to_call': utils.l2_tuning_curve,
    'param_range': [1, 10000, 211],
    'param_factor': 0.00001,
    'lr': 0.0025,
}
svm_lr_args = {
    'method_to_call': utils.learning_rate_tuning_curve,
    'param_range': [0, 101, 1],
    'param_factor': 0.00001,
    'l2reg': 0.0,
}
svm_reg_args = {
    'method_to_call': utils.l2_tuning_curve,
    'param_range': [1, 1000, 5],
    'param_factor': 0.0001,
    'lr': 0.0006,
}
learning_curve_args = {
    'method_to_call': utils.learning_curve,
    'param_range': [3, 670, 7],
    'param_factor': 1
}
loss_over_epochs = {'method_to_call': utils.loss_over_epochs}
lr_args = {'lr': 0.0025, 'l2reg': 0.001}
svm_args = {'lr': 0.0006, 'l2reg': 0.001}

models_and_args = zip([JaiLR(utils), JaiSVM(utils)],
                      [(lr_lr_args, lr_reg_args), (svm_lr_args, svm_reg_args)])

test_data, test_labels = data['test_data'], data['test_labels']
for model, (lr_params, reg_params) in models_and_args:
    tf.random.set_seed(random_seed)
    model.prepare_and_run(**common_args, **lr_params)  # learning rate tuning
    tf.random.set_seed(random_seed)
    model.prepare_and_run(**common_args, **reg_params) # regularization tuning
    tf.random.set_seed(random_seed)
    model.prepare_and_run(**common_args, **learning_curve_args, **svm_args)
    tf.random.set_seed(random_seed)
    model.prepare_and_run(**common_args, **loss_over_epochs, **svm_args)

    predictions = np.array(model.model.predict_on_batch(test_data)).argmax(axis=-1)
    utils.plot_roc_auc(test_labels, predictions)
    utils.error_analysis(test_data, test_labels, predictions)

# model = JaiLR(utils)
# model.prepare_and_run(**common_args, **loss_over_epochs, **lr_args)
# predictions = np.array(model.model.predict_on_batch(test_data)).argmax(axis=-1)
# utils.plot_roc_auc(test_labels, predictions)
# utils.error_analysis(test_data, test_labels, predictions)
