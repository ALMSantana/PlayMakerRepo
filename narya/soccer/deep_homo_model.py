import mxnet as mx
import cv2
import numpy as np
import tensorflow as tf
import segmentation_models as sm

sm.set_framework('tf.keras')

RESNET_ARCHI_TF_KERAS_PATH = (
    "https://storage.googleapis.com/narya-bucket-1/models/deep_homo_model_1.h5"
)
RESNET_ARCHI_TF_KERAS_NAME = "deep_homo_model_1.h5"
RESNET_ARCHI_TF_KERAS_TOTAR = False

def _build_homo_preprocessing(input_shape):
    """Builds the preprocessing function for the Deep Homography estimation Model.
    """

    def preprocessing(input_img, **kwargs):

        if len(input_img.shape) == 4:
            print(
                "Only preprocessing single image, we will consider the first one of the batch"
            )
            image = input_img[0]
        else:
            image = input_img

        image = cv2.resize(image, input_shape)
        image = torch_img_to_np_img(
            normalize_single_image_torch(np_img_to_torch_img(image))
        )
        return image

    return preprocessing

def pyramid_layer(
    x, indx, activation="tanh", output_size=8, nb_neurons=[512, 512, 256, 128]
):
    """Fully connected layers to add at the end of a network.
    Arguments:
        x: a tf.keras Tensor as input
        indx: Integer, an index to add to the name of the layers
        activation: String, name of the activation function to add at the end
        output_size: Size of the last layer, number of outputs
        nb_neurons: Size of the Dense layer to add
    Returns:
        output: a tf.keras Tensor as output
    Raises:
    """
    dense_name_base = "full_" + str(indx)
    for indx, neuron in enumerate(nb_neurons):
        x = tf.keras.layers.Dense(
            neuron, name=dense_name_base + str(neuron) + "_" + str(indx)
        )(x)
#     x = tf.keras.layers.Dense(output_size, name=dense_name_base + "output")(x)
    output = tf.keras.layers.Activation(activation)(x)
    return output

def _build_keypoint_preprocessing(input_shape, backbone):
    """Builds the preprocessing function for the Field Keypoint Detector Model.
    """
    sm_preprocessing = sm.get_preprocessing(backbone)

    def preprocessing(input_img, **kwargs):

        to_normalize = False if np.percentile(input_img, 98) > 1.0 else True

        if len(input_img.shape) == 4:
            print(
                "Only preprocessing single image, we will consider the first one of the batch"
            )
            image = input_img[0] * 255.0 if to_normalize else input_img[0] * 1.0
        else:
            image = input_img * 255.0 if to_normalize else input_img * 1.0

        image = cv2.resize(image, input_shape)
        image = sm_preprocessing(image)
        return image

    return preprocessing


def _build_resnet18():
    """Builds a resnet18 model in keras from a .h5 file.
    Arguments:
    Returns:
        a tf.keras.models.Model
    Raises:
    """
    resnet18_path_to_file = tf.keras.utils.get_file(
        RESNET_ARCHI_TF_KERAS_NAME,
        RESNET_ARCHI_TF_KERAS_PATH,
        RESNET_ARCHI_TF_KERAS_TOTAR,
    )

    resnet18 = tf.keras.models.load_model(resnet18_path_to_file)
    resnet18.compile()

    inputs = resnet18.input
    outputs = resnet18.layers[-2].output

    return tf.keras.models.Model(inputs=inputs, outputs=outputs, name="custom_resnet18")


class DeepHomoModel:
    """Class for Keras Models to predict the corners displacement from an image. These corners can then get used 
    to compute the homography.
    Arguments:
        pretrained: Boolean, if the model is loaded pretrained on ImageNet or not
        input_shape: Tuple, shape of the model's input 
    Call arguments:
        input_img: a np.array of shape input_shape
    """

    def __init__(self, pretrained=False, input_shape=(256, 256)):

        self.input_shape = input_shape
        self.pretrained = pretrained

        self.resnet_18 = _build_resnet18()

        inputs = tf.keras.layers.Input((self.input_shape[0], self.input_shape[1], 3))
        x = self.resnet_18(inputs)
        outputs = pyramid_layer(x, 2)
        
        self.model = tf.keras.models.Model(
            
            inputs=[inputs], outputs=outputs, name="DeepHomoPyramidalFull"
        )

        self.preprocessing = _build_homo_preprocessing(input_shape)

    def __call__(self, input_img):

        img = self.preprocessing(input_img)
        corners = self.model.predict(np.array([img]))

        return corners

    def load_weights(self, weights_path):
        try:
            self.model.load_weights(weights_path)
            print("Succesfully loaded weights from {}".format(weights_path))
        except:
            orig_weights = "Randomly"
            print(
                "Could not load weights from {}, weights will be loaded {}".format(
                    weights_path, orig_weights
                )
            )