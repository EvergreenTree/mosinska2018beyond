from unet_server import *

# test on training data
testGene = testGenerator('data/membrane/train/image')
model = unet()
model.load_weights("unet_membrane.hdf5")
results = model.predict_generator(testGene,10,verbose=1)
saveResult('data/membrane/trainresult/',results)

# test on test data
testGene = testGenerator('data/membrane/test')
results = model.predict_generator(testGene,10,verbose=1)
saveResult('data/membrane/testresult/',results)



