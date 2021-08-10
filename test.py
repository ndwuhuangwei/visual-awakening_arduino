from micronet_model import MicroNet

model = MicroNet(input_shape=(160, 160, 3), alpha=1.0, weights=None, classes=2)
model.trainable = True
model.summary()

