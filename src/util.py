import torch


####### Device handling #######

def get_device():
    """ Returns the device that is used for computation, depending on the availability of cuda and mps."""
    return (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )


####### Formatting #######

def make_bold(string):
    """ Returns the given string in bold for output in the terminal."""
    return '\033[1m' + string + '\033[0m'


####### Tensor handling #######

def ensure_is_tensor(tensor):
    """ Ensures that the given tensor is a torch.Tensor."""
    if not type(tensor) == torch.Tensor:
        tensor = torch.tensor(tensor)
    return tensor


def ensure_tensor_is_batch(tensor):
    """ Ensures that the given tensor is a batch, i.e. has at least 2 dimensions."""
    if len(tensor.shape) == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


def ensure_tensor_and_batched(input):
    """ Ensures that the given input is a torch.Tensor and a batch, i.e. has at least 2 dimensions."""
    input = ensure_is_tensor(input)
    input = ensure_tensor_is_batch(input)
    return input


def ensure_tensor_and_batched_for_all(input_list):
    """ Ensures that all given inputs are torch.Tensors and batches, i.e. have at least 2 dimensions."""
    for i in range(len(input_list)):
        input_list[i] = ensure_tensor_and_batched(input_list[i])
    return input_list


def assert_same_length(input_list):
    """ Asserts that all given tensors have the same length."""
    length = len(input_list[0])
    for t in input_list:
        assert len(t) == length, "All input tensors must have the same length. Got: {}".format([len(t) for t in input_list])
    