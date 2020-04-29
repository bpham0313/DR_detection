from model.cnn import create_model
from model.cnn import init_model
from model.cnn import save_model
from keras.utils import plot_model
from contextlib import redirect_stdout
#from keras.models import model_from_json

TEXT_FILE = '..\\model\\model.txt'
IMAGE_FILE = '..\\model\\model.png'


def to_text(model):
    with open(TEXT_FILE, 'w') as my_file:
        with redirect_stdout(my_file):
            model.summary()

        print("Exported to {0} successfully".format(TEXT_FILE))


def to_image(model):
    plot_model(model, to_file=IMAGE_FILE, show_layer_names=True,
               show_shapes=True)
    print("Exported to {0} successfully".format(IMAGE_FILE))


cnn_model = create_model()
cnn_model = init_model(cnn_model)
to_text(cnn_model)
to_image(cnn_model)
save_model(cnn_model)
