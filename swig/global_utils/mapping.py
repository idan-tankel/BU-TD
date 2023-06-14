
def get_mapping(word_file):
    """
    get_mapping is a function that returns a dictionary mapping words to indices and a list of words

    Args:
        word_file (_type_): _description_

    Returns:
        _type_: _description_
    """
    dict = {}
    word_list = []
    with open(word_file) as f:
        k = 0
        for line in f:
            word = line.split('\n')[0]
            dict[word] = k
            word_list.append(word)
            k += 1
    return dict, word_list