from model.data_utils import CoNLLDataset, segment_data
from model.config import Config

from model.ripple_model import RippleModel


def main():
    # create instance of config，这里的config实现了load data的作用
    #拥有词表、glove训练好的embeddings矩阵、str->id的function
    config = Config()
    config.nepochs          = 200
    config.dropout          = 0.5
    config.batch_size       = 100
    config.lr_method        = "adam"
    config.lr               = 0.0001
    config.lr_decay         = 1.0
    config.clip             = -2.0 # if negative, no clipping
    config.nepoch_no_imprv  = 5

    config.dir_model = config.dir_output + "model.finetuning.weights/"
    
    # build model
    model = RippleModel(config)
    model.build("fine_tuning")
    model.restore_session("results/test/model.weights/", indicate="fine_tuning")

    # model.restore_session("results/crf/model.weights/") # optional, restore weights
    # model.reinitialize_weights("proj")

    # create datasets [(char_ids), word_id]
    dev = CoNLLDataset(config.filename_dev, config.processing_word,
                       config.processing_action, config.max_iter)
    train = CoNLLDataset(config.filename_train, config.processing_word,
                         config.processing_action, config.max_iter)
    test = CoNLLDataset(config.filename_test, config.processing_word,
                        config.processing_action, config.max_iter)

    # train model
    train = segment_data(train, model.idx_to_action, config.max_size)

    model.train(train, dev, test)

if __name__ == "__main__":
    main()
