from model.cnn import *

cnn_model = create_model()
print(len(cnn_model.layers))
''
cnn_model = init_model(cnn_model)
print(cnn_model.summary())

trained_model = train_model(cnn_model)
''