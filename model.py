from keras.layers import Input, Conv2D, BatchNormalization, MaxPool2D, Conv2DTranspose, Concatenate, Activation
from keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG19

def conv_block(inputs, no_filters):
    x = Conv2D(no_filters, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = Conv2D(no_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    return x

def decoder_block(inputs, skip_features, no_filters):
    x = Conv2DTranspose(no_filters, (2, 2), strides=2, padding="same")(inputs)

    x = Concatenate(axis=3)([skip_features, x])
    x = conv_block(x, no_filters)
    return x

def build_vgg19_unet(input_shape):
    """ Input """
    input_imgs = Input(input_shape, name = 'RGB_Input')

    """ Pretrained VGG19 Model """
    vgg19 = VGG19(include_top=False, weights="imagenet", input_tensor=input_imgs)
    
#     vgg19.trainable = False

    """ Skip Features """
    e1 = vgg19.layers[2].output # (265, 265, 64)
    e2 = vgg19.layers[5].output # (128, 128, 128)
    e3 = vgg19.layers[10].output # (64, 64, 265)
    e4 = vgg19.layers[15].output # (32, 32, 512)

    """ Bridge """
    b1 = vgg19.layers[20].output # (16, 16, 512)

    """ Decoder """
    d1 = decoder_block(b1, e4, 64)
    d2 = decoder_block(d1, e3, 32)
    d3 = decoder_block(d2, e2, 16)
    d4 = decoder_block(d3, e1, 8)

    """ Outputs """
    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    model = Model(input_imgs, outputs)
    return model