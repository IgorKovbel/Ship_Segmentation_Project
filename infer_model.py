from keras.models import load_model
from utilities import *
from preprocessing import *
from model import build_vgg19_unet
import argparse

def predict(img_path, model):
    c_img = imread(img_path)
    img = np.expand_dims(c_img, 0)/255.0
    if IMG_SCALING is not None:
        img = img[:, ::IMG_SCALING[0], ::IMG_SCALING[1]]
    return model.predict(img)[0, :, :, 0]


def load_my_model():
    # custom_objects = {'IoU': IoU, 'dice_coef': dice_coef}

    # model = load_model('model/Unet_VGG19_model.h5', custom_objects=custom_objects)
    model = build_vgg19_unet((256, 256, 3))
    model.load_weights('model/VGG19_weights.best.hdf5')
    return model


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", required=True)
    args = parser.parse_args()

    # Check if the folder path exists
    if not os.path.exists(args.image_path):
        print(f"Path '{args.image_path}' does not exist.")
        return

    model = load_my_model()
    seg = predict(args.image_path, model)

    normalized_seg = ((seg - np.min(seg)) / (np.max(seg) - np.min(seg)) * 255).astype(np.uint8)
    img = Image.fromarray(normalized_seg)

    img.save("result.jpeg")
    print("Image saved to 'result.jpeg'.")

    plt.imshow(seg, vmin = 0, vmax = 1)
    plt.show()

if __name__ == "__main__":
    main()
