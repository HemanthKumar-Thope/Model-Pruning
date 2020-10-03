# Load MNIST dataset
# Here we ar with the tensorflow default datasets Loading and segregating the dataset into train and test 
#--------------------------------------------------------------------------------------------------------
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
#--------------------------------------------------------------------------------------------------------
# Normalize the input image so that each pixel value is between 0 to 1.
train_images = train_images / 255.0
test_images = test_images / 255.0
#--------------------------------------------------------------------------------------------------------
# Define the model architecture.
model = keras.Sequential([
  keras.layers.InputLayer(input_shape=(28, 28)),
  keras.layers.Reshape(target_shape=(28, 28, 1)),
  keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation='relu'),
  keras.layers.MaxPooling2D(pool_size=(2, 2)),
  keras.layers.Flatten(),
  keras.layers.Dense(10)
])
#--------------------------------------------------------------------------------------------------------
# Train the digit classification model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(
  train_images,
  train_labels,
  epochs=4,
  validation_split=0.1,
)
#--------------------------------------------------------------------------------------------------------
#This is the model base without pruning
#Let's save the model first and check the accuracy 
#Evaluate the model and check the accuracy and save the model
_, baseline_model_accuracy = model.evaluate(
    test_images, test_labels, verbose=0)

print('Baseline test accuracy:', baseline_model_accuracy)

_, keras_file = tempfile.mkstemp('.h5')
tf.keras.models.save_model(model, keras_file, include_optimizer=False)
print('Saved baseline model to:', keras_file)
#--------------------------------------------------------------------------------------------------------
#Model Pruning 
#--------------------------------------------------------------------------------------------------------
#Tensorflow API's have made the implementations also ver easy just by calling TensorFlow API's
#--------------------------------------------------------------------------------------------------------
import tensorflow_model_optimization as tfmot
#taking the object instance from tfmot and naming it appropirately
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
#--------------------------------------------------------------------------------------------------------
# Compute end step to finish pruning after 2 epochs.
batch_size = 128
epochs = 2
validation_split = 0.1
# 10% of training set will be used for validation set. 

num_images = train_images.shape[0] * (1 - validation_split)
end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs

# Define model for pruning.
pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,final_sparsity=0.80,begin_step=0,end_step=end_step)
      }

model_for_pruning = prune_low_magnitude(model, **pruning_params)

# `prune_low_magnitude` requires a recompile.
model_for_pruning.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model_for_pruning.summary()

#--------------------------------------------------------------------------------------------------------
#Train and evaluate the model against baseline
#--------------------------------------------------------------------------------------------------------
logdir = tempfile.mkdtemp()
#tfmot.sparsity.keras.UpdatePruningStep is required during training
#tfmot.sparsity.keras.PruningSummaries provides logs for tracking progress and debugging.

callbacks = [
  tfmot.sparsity.keras.UpdatePruningStep(),
  tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
]
model_for_pruning.fit(train_images, train_labels,
                  batch_size=batch_size, epochs=epochs, validation_split=validation_split,callbacks=callbacks)
#--------------------------------------------------------------------------------------------------------
#Check the pruned model accuracy and the baseline so that even after there are some discarded neurons
#model with 50% sparsity (50% zeros in weights) and end with 80% sparsity
#the model still gives an appropriately good accuracy
#reducing the weights, increasing the speed and reducing the power capacity resulting in light weight agile Machine Learning Model.
_, model_for_pruning_accuracy = model_for_pruning.evaluate(
   test_images, test_labels, verbose=0)

print('Baseline test accuracy:', baseline_model_accuracy) 
print('Pruned test accuracy:', model_for_pruning_accuracy)
#credits: Tensorflow documentations
#credits: https://www.tensorflow.org/model_optimization
#credits: https://www.youtube.com/watch?v=3JWRVx1OKQQ
#credits: https://www.youtube.com/watch?v=4iq-d2AmfRU
#credits: https://www.youtube.com/watch?v=3yOZxmlBG3Y
