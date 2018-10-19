
class EsposallesChar(object):
    """
    Transform the target string to an integer array
    """

    def __call__(self, text):
        return esposalles_char(text)


def esposalles_char(text):
    
