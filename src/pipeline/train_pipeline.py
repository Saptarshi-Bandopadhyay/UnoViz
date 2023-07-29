import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os
from tensorflow import keras
from src.components.dataGeneration import MapFunction, TripletGenerator
from src.components.model import get_embedding_module, get_siamese_network, SiameseModel

gpus = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs available : ", len(gpus))
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# path to training and testing data
TRAIN_DATASET = "./artifacts/train"
TEST_DATASET = "./artifacts/test"
# model input image size
IMAGE_SIZE = (224, 224)
# batch size and the buffer size
BATCH_SIZE = 32
BUFFER_SIZE = BATCH_SIZE * 2
# define autotune
AUTO = tf.data.AUTOTUNE
# define the training parameters
LEARNING_RATE = 0.001
STEPS_PER_EPOCH = 100
VALIDATION_STEPS = 10
EPOCHS = 100
# define the path to save the model
OUTPUT_PATH = "artifacts"
MODEL_PATH = os.path.join(OUTPUT_PATH, "siamese_network")


def model_train():
    print("[INFO] building the train and validation generators...")
    trainTripletGenerator = TripletGenerator(
        datasetPath=TRAIN_DATASET)
    valTripletGenerator = TripletGenerator(
        datasetPath=TRAIN_DATASET)
    print("[INFO] building the train and validation `tf.data` dataset...")
    trainTfDataset = tf.data.Dataset.from_generator(
        generator=trainTripletGenerator.get_next_element,
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.string),
        )
    )
    valTfDataset = tf.data.Dataset.from_generator(
        generator=valTripletGenerator.get_next_element,
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.string),
        )
    )

    mapFunction = MapFunction(imageSize=IMAGE_SIZE)
    print("[INFO] building the train and validation `tf.data` pipeline...")
    trainDs = (trainTfDataset
               .map(mapFunction)
               .shuffle(BUFFER_SIZE)
               .batch(BATCH_SIZE)
               .prefetch(AUTO)
               )
    valDs = (valTfDataset
             .map(mapFunction)
             .batch(BATCH_SIZE)
             .prefetch(AUTO)
             )

    # build the embedding module and the siamese network
    print("[INFO] build the siamese model...")
    embeddingModule = get_embedding_module(imageSize=IMAGE_SIZE)
    siameseNetwork = get_siamese_network(
        imageSize=IMAGE_SIZE,
        embeddingModel=embeddingModule
    )
    siameseModel = SiameseModel(
        siameseNetwork=siameseNetwork,
        margin=0.7,
        lossTracker=keras.metrics.Mean(name="loss"),
    )
    # compile the siamese model
    siameseModel.compile(
        optimizer=keras.optimizers.Adam(LEARNING_RATE)
    )

    my_callbacks = [
        EarlyStopping(patience=7),
        #     ModelCheckpoint(filepath='checkpoint1/model.h5',monitor='val_loss', mode=min, save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=4, min_lr=0.000001)
    ]

    # train and validate the siamese model
    print("[INFO] training the siamese model...")
    history = siameseModel.fit(
        trainDs,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=valDs,
        validation_steps=VALIDATION_STEPS,
        callbacks=my_callbacks,
        epochs=100
    )

    model_save(siameseModel)


def model_save(siameseModel):
    # check if the output directory exists, if it doesn't, then
    # create it
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    # save the siamese network to disk
    modelPath = MODEL_PATH
    print(f"[INFO] saving the siamese network to {modelPath}...")
    keras.models.save_model(
        model=siameseModel.siameseNetwork,
        filepath=modelPath,
        include_optimizer=False,
    )


if __name__ == '__main__':
    model_train()
