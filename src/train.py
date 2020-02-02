from utils import get_loader
from config import get_config
from utils import Vocab
import os
import solvers
from utils import load_pickle


if __name__ == '__main__':
    config = get_config(mode='train')
    val_config = get_config(mode='valid')
    with open(os.path.join(config.save_path, 'config.txt'), 'w') as f:
        print(config, file=f)

    vocab = Vocab()
    vocab.load(config.word2id_path, config.id2word_path)
    print(f'Vocabulary size: {vocab.vocab_size}')
    config.vocab_size = vocab.vocab_size

    if config.users:
        train_users = load_pickle(config.convs_users_path)
        config.user_size = max([x for xx in train_users for x in xx]) + 1
        print(f'User size: {config.user_size}')
        eval_users = load_pickle(val_config.convs_users_path)
    else:
        train_users = None
        eval_users = None
        
    train_data_loader = get_loader(convs=load_pickle(config.convs_path),
                                   convs_length=load_pickle(config.conversations_length_path),
                                   utterances_length=load_pickle(config.utterances_length_path),
                                   vocab=vocab, convs_users=train_users,
                                   batch_size=config.batch_size)

    eval_data_loader = get_loader(convs=load_pickle(val_config.convs_path),
                                  convs_length=load_pickle(val_config.conversations_length_path),
                                  utterances_length=load_pickle(val_config.utterances_length_path),
                                  vocab=vocab, shuffle=False, convs_users=eval_users,
                                  batch_size=val_config.eval_batch_size)

    model_solver = getattr(solvers, "Solver{}".format(config.model))
    solver = model_solver(config, train_data_loader, eval_data_loader, vocab=vocab, is_train=True)

    solver.build()
    solver.train()
