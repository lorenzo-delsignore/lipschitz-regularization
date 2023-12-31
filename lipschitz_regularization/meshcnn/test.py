from lipschitz_regularization.meshcnn.options.test_options import TestOptions
from lipschitz_regularization.meshcnn.data import DataLoader
from lipschitz_regularization.meshcnn.models import create_model
from lipschitz_regularization.meshcnn.util.writer import Writer


def run_test(epoch=-1):
    print("Running Test")
    opt = TestOptions().parse()
    opt.serial_batches = True  # no shuffle
    dataset = DataLoader(opt)
    model = create_model(opt)
    writer = Writer(opt)
    # test
    writer.reset_counter()
    for i, data in enumerate(dataset):
        model.set_input(data)
        ncorrect, nexamples = model.test()
        writer.update_counter(ncorrect, nexamples)
    writer.print_acc(epoch, writer.acc)
    return writer.acc


if __name__ == "__main__":
    run_test()
