from .vhred_solver import SolverVHRED


class SolverVHCR(SolverVHRED):
    def __init__(self, config, train_data_loader, eval_data_loader, vocab, is_train=True, model=None):
        super(SolverVHCR, self).__init__(config, train_data_loader, eval_data_loader, vocab, is_train, model)

    def test(self):
        raise NotImplementedError
