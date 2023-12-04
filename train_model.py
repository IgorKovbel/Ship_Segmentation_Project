from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from utilities import IoU, dice_coef, CustomDataGen
from model import build_vgg19_unet, Adam
from preprocessing import get_data


def train_model(input_shape, train_generator, valid_x, valid_y):
    model = build_vgg19_unet(input_shape)
    
    weight_path="weights.best.hdf5"
    checkpoint = [ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only = True)]
    
    model.compile(optimizer=Adam(1e-3), loss=IoU, metrics=[dice_coef, 'binary_accuracy'])
    model.fit(
        train_generator, 
        steps_per_epoch=20, 
        epochs=500,
        validation_data=(valid_x, valid_y),
        callbacks=checkpoint,
        verbose=1
            )
    model.save('trained_model.h5')

if __name__ == "__main__":
    train_dir = 'F:/Ship_Dataset/train_v2/'
    mask_dir = 'F:/Ship_Dataset/train_ship_segmentations_v2.csv'

    train_df, valid_df = get_data(mask_dir)

    train_generator = CustomDataGen(train_df, train_dir, False)
    valid_x, valid_y = next(iter(CustomDataGen(valid_df, train_dir, 900, False)))

    shape = valid_x.shape[1:]
    train_model(shape, train_generator, valid_x, valid_y)