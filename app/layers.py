#Custom L1 Distance layer module

import tensorflow as tf
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.layers import Layer, Conv2D, MaxPooling2D, Dense, Flatten, Input
from tensorflow.python.keras.models import Model

#Custom L1 Distance layer from Jupyter
class L1Dist(Layer):
    
    def __init__(self, **kwargs):
        super().__init__()

    #similarity calculation
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)

def creatingSiameseModel():

    def make_embedding(): 
        inp = Input(shape=(100,100,3), name='input_image')
        
        # First block
        c1 = Conv2D(64, (10,10), activation='relu')(inp)
        m1 = MaxPooling2D(64, (2,2), padding='same')(c1)
        
        # Second block
        c2 = Conv2D(128, (7,7), activation='relu')(m1)
        m2 = MaxPooling2D(64, (2,2), padding='same')(c2)
        
        # Third block 
        c3 = Conv2D(128, (4,4), activation='relu')(m2)
        m3 = MaxPooling2D(64, (2,2), padding='same')(c3)
        
        # Final embedding block
        c4 = Conv2D(256, (4,4), activation='relu')(m3)
        f1 = Flatten()(c4)
        d1 = Dense(4096, activation='sigmoid')(f1)
        
        
        return Model(inputs=[inp], outputs=[d1], name='embedding')

    embedding = make_embedding()

    def make_siamese_model(): 
        
        # Anchor image input in the network
        input_image = Input(name='input_img', shape=(100,100,3))
        
        # Validation image in the network 
        validation_image = Input(name='validation_img', shape=(100,100,3))
        
        # Combine siamese distance components
        siamese_layer = L1Dist()
        siamese_layer._name = 'distance'
        distances = siamese_layer(embedding(input_image), embedding(validation_image))
        
        # Classification layer 
        classifier = Dense(1, activation='sigmoid')(distances)
        
        return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')
    
    return make_siamese_model()
    