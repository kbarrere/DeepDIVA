import logging
import sys

# DeepDIVA HTR
from template.runner.handwritten_text_recognition.ctc_decode.best_path import best_path_batch
from template.runner.handwritten_text_recognition.ctc_decode.beam_search import beam_search_batch


def ctc_decode(probs, acts_len, dictionnary_name, ctc_decoder, beam_width, **kwargs):
    # Get parameters according to dictionnary_name
    char_list = get_char_list(dictionnary_name)
    blank_index=0
    
    # Call to the chossen decoder
    if ctc_decoder == "bestpath":
        return best_path_batch(probs, char_list, acts_len, blank_index)
    
    elif ctc_decoder == "beamsearch":
        if not beam_width:
            beam_width = 10
        
        # Apply a softmax to the probabilities
        s = nn.Softmax(dim=2)
        probs = s(probs)
        
        return beam_search_batch(probs, char_list, acts_len, blank_index, beam_width)
    
    else:
        logging.error("ctc_decoder not recognized: " + ctc_decoder)
        sys.exit(-1)

def get_char_list(dictionnary_name):
    if dictionnary_name == "iam":
       return [
                "<BLANK>",
                "-",
                ",",
                ";",
                ":",
                "!",
                "?",
                "/",
                ".",
                "'",
                "\"",
                "(",
                ")",
                "*",
                "&",
                "#",
                "+",
                "0",
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "7",
                "8",
                "9",
                "a",
                "A",
                "b",
                "B",
                "c",
                "C",
                "d",
                "D",
                "e",
                "E",
                "f",
                "F",
                "g",
                "G",
                "h",
                "H",
                "i",
                "I",
                "j",
                "J",
                "k",
                "K",
                "l",
                "L",
                "m",
                "M",
                "n",
                "N",
                "o",
                "O",
                "p",
                "P",
                "q",
                "Q",
                "r",
                "R",
                "s",
                "S",
                "t",
                "T",
                "u",
                "U",
                "v",
                "V",
                "w",
                "W",
                "x",
                "X",
                "y",
                "Y",
                "z",
                "Z",
                " "
              ]
    elif dictionnary_name == "read2018":
        return [
                "<BLANK>"
                '=',
                '¬',
                '|',
                '°',
                '┌',
                '│',
                ' ',
                '-',
                ',',
                ';',
                ':',
                '!',
                '?',
                '/',
                '.',
                '·',
                '\\',
                '’',
                '"',
                '”',
                '«',
                '»',
                '(',
                ')',
                '[',
                ']',
                '§',
                '\'',
                '&',
                '—',
                '0',
                '1',
                '2',
                '3',
                '4',
                '5',
                '6',
                '7',
                '8',
                '9',
                'a',
                'A',
                'æ',
                'b',
                'B',
                'c',
                'C',
                'd',
                'D',
                'e',
                'E',
                'f',
                'F',
                'g',
                'G',
                'h',
                'H',
                'i',
                'I',
                'j',
                'J',
                'k',
                'K',
                'l',
                'L',
                'm',
                'M',
                'n',
                'N',
                'o',
                'O',
                'ø',
                'p',
                'P',
                'q',
                'Q',
                'r',
                'R',
                's',
                'S',
                'ß',
                't',
                'T',
                'u',
                'U',
                'v',
                'V',
                'w',
                'W',
                'x',
                'X',
                'y',
                'Y',
                'z',
                'Z',
                'ʒ',
                '–',
               ]
    elif dictionnary_name == "esposalles":
        return [
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
    else:
        logging.error("dictionnary name not recognized: " + dictionnary_name)
        sys.exit(-1)
