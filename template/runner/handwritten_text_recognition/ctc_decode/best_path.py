import numpy as np

def best_path(probs, char_list, max_len=-1, blank_index=0):
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
        expected shape of : frames_number x character_number
    char_list : character list
        A list containing every characters of the datasets
        It has to be in the same order that the text is encoded
    max_len : int
        The maximum length to where the decoding shoudl stop
        By default, the value is set to -1
        A value of -1 means that the decoding should stop when the algorithm reaches the end of the matrix probs
    blank_index : int
        The index of the blank character.
        Default is blank

    Returns
    -------
    sequences : string array
        The predicted sequence in an array of size batch_size
    """
    
    # Get the max_len, maximum number of algorithm iterations
    # The probability matrix could cover more cases with padding
    if max_len == -1:
        max_len = len(probs)
    else:
        max_len = min(max_len, len(probs))
    
    character_number = len(char_list)
        
    sequence_raw = []
    
    # Get the characters with the hightest values
    for j in range(max_len):
        frame = probs[j]
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
    char_sequence = convert_int_to_chars(sequence, char_list)
        
    return char_sequence

def best_path_batch(probs, char_list, acts_len=[], blank_index=0):
    """
    Generates the predicted sequence based on the characters probabilities.
    The rules to generates the text are the following :
     - For each frame, we get the character that has the hightest value for the hightest value/probability
     - We remove the duplicated character (subsequent character that are the sames)
     - We then remove all the blanks characters
     It is equivalent to beam search decoding, but with a beam width of 1.
     It could be called a greedy decoding.
    
    Parameters
    ----------
    probs : tensor
        Tensor/array containing the outputs of the network / the probability of each character for each frame
        expected shape of : batch_size x frames_number x character_number
    blank_index : int
        The index of the blank character.
        Default is blank
    dictionnary_name : string
        Name of the dictionnary used.
        Determine the number of characters in the dataset.

    Returns
    -------
    sequences : string array
        The predicted sequence in an array of size batch_size
    """
    
    sequences = []
    
    batch_size = len(probs)
    
    for i in range(batch_size):
        probs_unit = probs[i]
        max_len = len(probs[0])
        if len(acts_len) != 0:
            max_len = acts_len[i]
        sequence = best_path(probs_unit, char_list, max_len, blank_index)
        sequences.append(sequence)
    
    return sequences

def convert_int_to_chars(indices, char_list):
    chars_sequence = ""
    
    for char_index in indices:
        chars_sequence += char_list[char_index]
    
    return chars_sequence
