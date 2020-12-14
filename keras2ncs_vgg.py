import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from keras.models import model_from_json
from tensorflow.compat.v1.keras import backend as K
import keras

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in
                                tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                        output_names, freeze_var_names)
    return frozen_graph


model_file = "vgg_pretrained.h5"
model=keras.models.load_model(model_file)

K.set_learning_phase(0)


frozen_graph=freeze_session(K.get_session(),
                            output_names=[out.op.name for out in model.outputs])



tf.train.write_graph(frozen_graph, "TF_model_vgg/", "tf_model.pb",as_text=False)