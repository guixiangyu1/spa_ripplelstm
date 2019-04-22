from model.data_utils import CoNLLDataset, segment_data
from model.config import Config

from model.ripple_model import RippleModel


def main():
    # create instance of config，这里的config实现了load data的作用
    #拥有词表、glove训练好的embeddings矩阵、str->id的function
    config = Config()


    # build model
    model = RippleModel(config)
    model.build("train")


    # model.restore_session("results/crf/model.weights/") # optional, restore weights
    # model.reinitialize_weights("proj")

    # create datasets [(char_ids), word_id]
    dev = CoNLLDataset(config.filename_dev, config.processing_word,
                        config.processing_action, config.max_iter)
    train = CoNLLDataset(config.filename_train, config.processing_word,
                         config.processing_action, config.max_iter)
    test = CoNLLDataset(config.filename_test, config.processing_word,
                       config.processing_action, config.max_iter)

    train = segment_data(train, model.idx_to_action)

    # train model
    model.train(train, dev, test)




if __name__ == "__main__":
    main()
