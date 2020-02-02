from config import get_config
from utils import Vocab, get_loader, load_pickle
import solvers


def main():
    config = get_config(mode='test')

    vocab = Vocab()
    vocab.load(config.word2id_path, config.id2word_path)
    print(f'Vocabulary size: {vocab.vocab_size}')
    config.vocab_size = vocab.vocab_size

    if config.users:
        test_users = load_pickle(config.convs_users_path)
        config.user_size = max([x for xx in test_users for x in xx]) + 1
        print(f'User size: {config.user_size}')
    else:
        test_users = None

    data_loader = get_loader(convs=load_pickle(config.convs_path),
                             convs_length=load_pickle(config.conversations_length_path),
                             utterances_length=load_pickle(config.utterances_length_path),
                             vocab=vocab, batch_size=config.batch_size, shuffle=False, convs_users=test_users)

    model_solver = getattr(solvers, "Solver{}".format(config.model))
    test_solver = model_solver(config, None, data_loader, vocab=vocab, is_train=False)

    test_solver.build()
    test_solver.export_samples()


if __name__ == '__main__':
    main()
