import numpy as np
import logging

# Revert dictionnaries

esposalles_chars = [
    '<BLANK>', #0
    '#', #1
    '0', #2
    '1', #3
    '2', #4
    '3', #5
    '4', #6
    '5', #7
    '6', #8
    '7', #9
    '8', #10
    '9', #11
    'a', #12
    'A', #13
    'b', #14
    'B', #15
    'c', #16
    'C', #17
    '<cedilla>', #18
    'd', #19
    'D', #20
    'e', #21
    'E', #22
    'f', #23
    'F', #24
    'g', #25
    'G', #26
    'h', #27
    'H', #28
    'i', #29
    'I', #30
    'j', #31
    'J', #32
    'l', #33
    'L', #34
    'm', #35
    'M', #36
    'n', #37
    'N', #38
    'o', #39
    'O', #40
    'p', #41
    'P', #42
    'q', #43
    'Q', #44
    'r', #45
    'R', #46
    's', #47
    'S', #48
    't', #49
    'T', #50
    'u', #51
    'U', #52
    'v', #53
    'V', #54
    'x', #55
    'X', #56
    'y', #57
    'Y', #58
    'z', #59
    ' ' #60
]

def sample_text(probs, acts_len=[], blank_index=0, char_list=esposalles_chars):
    """
    Generates the predicted sequence based on the characters probabilities.
    The rules to generates the text are the following :
     - For each frame, we get the character that has the hightest value for the hightest value/probability
     - We remove the duplicated character (subsequent character that are the sames)
     - We then remove all the blanks characters
    
    Parameters
    ----------
    probs : tensor
        Tensor/array containing the outputs of the network / the probability of each character for each frame
        expected shape of : batch_size x frames_number x character_number
    blank_index : int
        The index of the blank character.
        Default is blank
    char_list : char list
        A list to convert indices into characters
        default is esposalles_chars, that contains all characters from the esposalles database

    Returns
    -------
    sequences : string array
        The predicted sequence in an array of size batch_size
    """
    
    sequences = []
    batch_size = len(probs)
    max_seq_size = len(probs[0])
    character_number = len(probs[0][0])
    
    for i in range(batch_size):
        
        frames = probs[i]
        sequence_raw = []
        act_len = max_seq_size
        if len(acts_len) != 0:
            act_len = acts_len[i]
        
        # Get the characters with the hightest values
        for j in range(act_len):
            frame = frames[j]
            predicted_char = np.argmax(frame)
            sequence_raw.append(predicted_char)
        
        # Remove the duplicated characters
        sequence_without_duplicates = []
        previous_char = -1
        for char in sequence_raw:
            if char != previous_char:
                sequence_without_duplicates.append(char)
                previous_char = char
        
        # Remove the blanks
        sequence = []
        for char in sequence_without_duplicates:
            if char != blank_index:
                sequence.append(char)
        
        # Convert to characters
        chars_sequence = convert_int_to_chars(sequence, char_list)
            
        sequences.append(chars_sequence)
        
    return sequences

def convert_int_to_chars(indices, char_list):
    chars_sequence = ""
    
    for char_index in indices:
        chars_sequence += char_list[char_index]
    
    return chars_sequence

def convert_batch_to_sequence(labels, char_list=esposalles_chars):
    char_sequences = []
    
    for indices in labels:
        # First remove padding
        indices = indices[indices.nonzero()]
        char_sequences.append(convert_int_to_chars(indices, char_list))
    
    return char_sequences

def levenshtein(a, b, w_ins=1, w_del=1, w_sub=1):
    """
    Computes the Levenshtein distance aka the edit distance between a and b
    It is the Wagner–Fischer algorithm.
    
    Parameters
    ----------
    a : array
        a is the the first word
        It could be a string, an array, anything that is iterable in principle
    b : array
        b is the the second word
        It could be a string, an array, anything that is iterable in principle
    w_ins : int
        w_ins is the cost of an insertion of a letter of word a to go to word b
        default value is 1
    w_del : int
        w_del is the cost of a deletion of a letter of word a to go to word b
        default value is 1
    w_sub : int
        w_sub is the cost of a substition of a letter of word a by one of word b
        default value is 1

    Returns
    -------
    edit_distance : int
        The edit distance or the levenshtein distance between a and b
    """
    
    n = len(a)
    m = len(b)
    
    D = np.zeros((m + 1, n + 1), dtype=np.int16)
    
    # First initializes the first column
    for i in range(1, m+1):
        D[i][0] = i * w_del
    
    # Then initializes the first row
    for j in range(1, n+1):
        D[0][j] = j * w_ins
    
    #Now compute the whole matrix
    for i in range(1, m+1):
        for j in range(1, n+1):
            if a[j-1] == b[i-1]:
                D[i][j] = D[i-1][j-1]
            else:
                D[i][j] = min(D[i-1][j] + w_del, D[i][j-1] + w_ins, D[i-1][j-1] + w_sub)
    
    return D[m][n]

def cer(prediction, reference):
    """
    Computes the Character Error Rate between the sentences prediction and reference
    It uses the Levenstein distance between the two words normalized by the length of the reference
    
    Parameters
    ----------
    prediction: array
        prediction is the word to being compared to the reference
        It could be a string, an array, anything that is iterable in principle
    reference: array
        reference is the word to that prediction is being compared with
        It could be a string, an array, anything that is iterable in principle
    
    Returns
    -------
    cer : float
    """
    
    N = len(reference)
    if N == 0:
        return len(prediction)
    return levenshtein(prediction, reference) / len(reference)

def wer(prediction, reference):
    """
    Computes the Word Error Rate between the sentences prediction and reference
    It uses the Levenstein distance between the two words normalized by the length of the reference
    The difference with the Character Error Rate is that the word are the minimal unit
    
    Parameters
    ----------
    prediction: array
        prediction is the word to being compared to the reference
        It could be a string, an array, anything that is iterable in principle
    reference: array
        reference is the word to that prediction is being compared with
        It could be a string, an array, anything that is iterable in principle
    
    Returns
    -------
    cer : float
    """
    
    pred_words = prediction.split(" ")
    ref_words = reference.split(" ")
    
    N = len(ref_words)
    if N == 0:
        return len(pred_words)
    return levenshtein(pred_words, ref_words) / len(ref_words)
    
def batch_cer(predictions, references):
    """
    Computes the Character Error Rate between the sentences of predictions and references
    It uses the Levenstein distance between the two words normalized by the length of the reference
    
    Parameters
    ----------
    predictions: array
        predictions contains the word to being compared to the reference
        the words could be a string, an array, anything that is iterable in principle
        expected shape of: batch_size x whatever
    references: array
        references contains the word to that prediction is being compared with
        the words could be a string, an array, anything that is iterable in principle
        expected shape of: batch_size x whatever
    Returns
    -------
    cer : float
    """
    
    cer_ = 0
    batch_size = len(predictions)
    
    for i in range(batch_size):
        cer_ += cer(predictions[i], references[i])
    
    return cer_ / batch_size

def batch_wer(predictions, references):
    """
    Computes the Word Error Rate between the sentences predictions and references
    It uses the Levenstein distance between the two words normalized by the length of the reference
    The difference with the Character Error Rate is that the word are the minimal unit
    
    Parameters
    ----------
    predictions: array
        predictions contains the word to being compared to the reference
        the words could be a string, an array, anything that is iterable in principle
        expected shape of: batch_size x whatever
    references: array
        references contains the word to that prediction is being compared with
        the words could be a string, an array, anything that is iterable in principle
        expected shape of: batch_size x whatever
    
    Returns
    -------
    wer : float
    """
    
    wer_ = 0
    batch_size = len(predictions)
    
    for i in range(batch_size):
        wer_ += wer(predictions[i], references[i])
    
    return (wer_ / batch_size)
