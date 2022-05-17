import os
import cv2
from imutils import paths
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

from pocovidnet import MODEL_FACTORY

class SetConfModel(object):
    def __init__(self, args):
        # Inicialización de hiperparámetros
        self.args = args
        self.DATA_DIR = args['data_dir']
        self.MODEL_NAME = args['model_name']
        self.FOLD = args['fold']
        self.MODEL_DIR = os.path.join(args['model_dir'], self.MODEL_NAME, f'_fold_{self.FOLD}')
        self.LR = args['learning_rate']
        self.EPOCHS = args['epochs']
        self.BATCH_SIZE = args['batch_size']
        self.MODEL_ID = args['model_id']
        self.TRAINABLE_BASE_LAYERS = args['trainable_base_layers']
        self.IMG_WIDTH, self.IMG_HEIGHT = args['img_width'], args['img_height']
        self.LOG_SOFTMAX = args['log_softmax']
        self.HIDDEN_SIZE = args['hidden_size']

    def check_model(self):
        # Check if model class exists
        if self.MODEL_ID not in MODEL_FACTORY.keys():
            raise ValueError(
                f'Model {self.MODEL_ID} not implemented. Choose from {MODEL_FACTORY.keys()}'
            )
        if not os.path.exists(self.MODEL_DIR):
            os.makedirs(self.MODEL_DIR)

        print('Configuración de hiperparámetros:\n')
        for key, value in self.args.items():
            print(f'{key}: {value}')
    
    def train_test_dataset(self):
        # grab the list of images in our dataset directory, then initialize
        # the list of data (i.e., images) and class images
        print('Cargando imágenes...')
        imagePaths = list(paths.list_images(self.DATA_DIR))
        data = []
        labels = []
        print('OK!.', len(imagePaths), 'imágenes cargadas')
        print(f'Fold seleccionado: {self.FOLD}')
        
        train_labels, test_labels = [], []
        train_data, test_data = [], []
        # test_files = []
        
        # Iteración sobre cada fold
        for imagePath in imagePaths:
            path_parts = imagePath.split(os.path.sep) # Ejm: ['.', 'image_cross_val', 'split0', 'regular', 'Reg-Grep-Alines.mp4_frame40.jpg']
            # Extraemos el número del split. Ejem: 0
            train_test = path_parts[-3][-1]
            # Extraemos la clase. Ejem: regular
            label = path_parts[-2]
            # Cargamos la imagen, intecambiamos los canales y redimensionamos para que tengan un tamaño fijo
            # 224x224 pixeles mientras se ignoran aspectos del radio
            image = cv2.imread(imagePath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (self.IMG_WIDTH, self.IMG_HEIGHT))

            # Separamos las imágenes en conjuntos de entremaniento y pruebas
            # El split es homogéneo, entonces escoger cualquier fold para el test debería tener similar resultado
            if train_test == str(self.FOLD):
                test_labels.append(label)
                test_data.append(image)
                # test_files.append(path_parts[-1])
            else:
                train_labels.append(label)
                train_data.append(image)

        # Preparamos los datos para el modelo
        print(
            f'\nNumber of training samples: {len(train_labels)} \n'
            f'Number of testing samples: {len(test_labels)}' '\n'
            f'Total samples: {len(train_labels) + len(test_labels)}'
        )

        assert len(set(train_labels)) == len(set(test_labels)), (
            'Something went wrong. Some classes are only in train or test data.'
        )
        
        return train_data, test_data, train_labels, test_labels
    
    def one_hot_encoding(self, train_data, test_data, train_labels, test_labels):
        #convert the data and labels to NumPy arrays while scaling the pixel
        #intensities to the range [0, 255]
        train_data = np.array(train_data) / 255.0
        test_data = np.array(test_data) / 255.0
        train_labels_text = np.array(train_labels)
        test_labels_text = np.array(test_labels)

        num_classes = len(set(train_labels))

        # perform one-hot encoding on the labels
        lb = LabelBinarizer()
        lb.fit(train_labels_text)

        train_labels = lb.transform(train_labels_text)
        test_labels = lb.transform(test_labels_text)

        if num_classes == 2:
            train_labels = to_categorical(train_labels, num_classes=num_classes)
            test_labels = to_categorical(test_labels, num_classes=num_classes)

        trainX = train_data
        trainY = train_labels
        testX = test_data
        testY = test_labels
        print('Class mappings are:', lb.classes_)
        
        return lb.classes_, trainX, testX, trainY, testY
    
    def compile_model(self, model):
        # compile model
        print('Compiling model...')
        opt = Adam(lr=self.LR, decay=self.LR / self.EPOCHS)
        loss = (
            tf.keras.losses.CategoricalCrossentropy() if not self.LOG_SOFTMAX else (
                lambda labels, targets: tf.reduce_mean(
                    tf.reduce_sum(
                        -1 * tf.math.multiply(tf.cast(labels, tf.float32), targets),
                        axis=1
                    )
                )
            )
        )

        model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])

        print(f'Model has {model.count_params()} parameters')
        print(f'Model summary {model.summary()}')
        
    def fit_model(self, model, trainAug, earlyStopping, mcp_save, reduce_lr_loss, metrics, trainX, testX, trainY, testY):
        print('Starting training model...')
        H = model.fit(
            trainAug.flow(trainX, trainY, batch_size=self.BATCH_SIZE),
            steps_per_epoch=len(trainX) // self.BATCH_SIZE,
            validation_data=(testX, testY),
            validation_steps=len(testX) // self.BATCH_SIZE,
            epochs=self.EPOCHS,
            callbacks=[earlyStopping, mcp_save, reduce_lr_loss, metrics]
        )
        return H