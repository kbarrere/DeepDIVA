import logging
import numpy as np
import torch
import sys

class CTCLabelToTensor(object):
    """
    transform CTC labels (an array of integers) to a tensor
    """
    
    def __call__(self, ctc_label):
        return self.ctc_label_to_tensor(ctc_label)
    
    def ctc_label_to_tensor(self, ctc_label):
        ctc_label_array = np.array(ctc_label)
        ctc_label_tensor = torch.from_numpy(ctc_label_array)
        return ctc_label_tensor

class PadToFixedSize(object):
    """
    Complete the value of the CTC labels so that the array has a fixed size
    """
    
    def __init__(self, max_size, fill_value=0):
        self.max_size = max_size
        self.fill_value = fill_value
    
    def __call__(self, ctc_label):
        return self.pad_to_fixed_size(ctc_label)
    
    def pad_to_fixed_size(self, ctc_label):
        n = len(ctc_label)
        
        if n > self.max_size:
            logging.warning("Cannot pad image of size " + str(n) + " to a size of " + str(self.max_size))
            sys.exit(-1)

        for i in range(self.max_size - n):
            ctc_label.append(self.fill_value)
        
        return ctc_label

class EsposallesCharToCTCLabel(object):
    """
    Transform the target string to an integer array
    """
    
    def __init__(self):
        self.dic = esposalles_dict()
        
    def __call__(self, text):
        return self.esposalles_char_to_ctc_label(text)

    def esposalles_char_to_ctc_label(self, text):
        label = []
        for c in text:
            label.append(self.dic.get(c))
        return label
    
def esposalles_dict():
    dico = {
        "#" : 1,
        "0" : 2,
        "1" : 3,
        "2" : 4,
        "3" : 5,
        "4" : 6,
        "5" : 7,
        "6" : 8,
        "7" : 9,
        "8" : 10,
        "9" : 11,
        "a" : 12,
        "A" : 13,
        "b" : 14,
        "B" : 15,
        "c" : 16,
        "C" : 17,
        "ç" : 18,
        "d" : 19,
        "D" : 20,
        "e" : 21,
        "E" : 22,
        "f" : 23,
        "F" : 24,
        "g" : 25,
        "G" : 26,
        "h" : 27,
        "H" : 28,
        "i" : 29,
        "I" : 30,
        "j" : 31,
        "J" : 32,
        "l" : 33,
        "L" : 34,
        "m" : 35,
        "M" : 36,
        "n" : 37,
        "N" : 38,
        "o" : 39,
        "O" : 40,
        "p" : 41,
        "P" : 42,
        "q" : 43,
        "Q" : 44,
        "r" : 45,
        "R" : 46,
        "s" : 47,
        "S" : 48,
        "t" : 49,
        "T" : 50,
        "u" : 51,
        "U" : 52,
        "v" : 53,
        "V" : 54,
        "x" : 55,
        "X" : 56,
        "y" : 57,
        "Y" : 58,
        "z" : 59,
        " " : 60
    }

    return dico
