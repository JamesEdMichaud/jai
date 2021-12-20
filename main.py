import numpy as np
import tensorflow as tf
# Disable GPU on Apple architecture. It's slower sometimes.
tf.config.set_visible_devices([], 'GPU')
from utils import JaiUtils
from models import JaiNN, JaiLR, JaiSVM

# print("TensorFlow version: ", tf.__version__)


random_seed = 638
weights_random_seed = 438

utils = JaiUtils(
    vid_path="training_data",
    img_size=(64, 64),
    max_seq_len=40,
    train_split=0.9,
    val_split=0.2,
    l1_reg=0.0,
    c=1,
    sigma=0.1,
    rand_seed=random_seed,
    weights_seed=weights_random_seed,
    epochs=300,
    # Found using parameter tuning curve:
    # Logistic Regression: ~0.0000005
    # SVM: ~0.0007
    # Neural Network: 0.01 -- ??
    learning_rate=0.01,
    # Found using parameter tuning curve:
    # Logistic Regression: No effect
    # SVM: ~0.01
    # Neural Network: ???
    l2_reg=0.01,
    training_data_updated=False,
    batch_size=256,
    using_augmentation=False,
)

data = utils.get_data()

print(f"Training data shape: {data['train_data'].shape}")
print(f"Test data shape: {data['test_data'].shape}")

model = JaiLR(utils)
# model = JaiSVM(utils)
# model = JaiNN(utils)

# Override the utils defaults here
args = {
    'data': data,
    'method_to_call': utils.loss_over_epochs,
    # 'method_to_call': utils.learning_rate_tuning_curve,
    # 'method_to_call': utils.l2_tuning_curve,
    # 'method_to_call': utils.learning_curve,
    'epochs': 3,
    'param_range': [3, 670, 22],    # Does nothing for loss /epochs
    'param_factor': 1          # Does nothing for loss /epochs
}
tf.random.set_seed(random_seed)
best_model = model.prepare_and_run(**args)

test_data, test_labels = data['test_data'], data['test_labels']
tf.random.set_seed(random_seed)
rand = tf.random.shuffle(np.arange(len(test_labels)))[0]
pred_input = test_data[rand]
utils.prediction(model, pred_input, test_labels[rand])
