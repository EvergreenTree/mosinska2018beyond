from unet_server import *

path = './data/'

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
myGene = trainGenerator(2,'data/membrane/train','image','label',data_gen_args,save_to_dir = None)
model = unet()

if os.path.isfile('unet_membrane.hdf5')
    model.load_weights("unet_membrane.hdf5")

model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
# training
model.fit_generator(myGene,steps_per_epoch=2000,epochs=5,callbacks=[model_checkpoint]) # train

# base_model = VGG19(include_top=False,weights='imagenet',input_shape=(224,224,3))#shape cannot be changed
# model = Model(inputs=base_model.input, outputs=base_model.get_layer('block1_conv2').output)

