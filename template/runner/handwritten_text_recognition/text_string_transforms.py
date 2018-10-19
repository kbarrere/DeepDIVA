
class EsposallesChar(object):
    """
    Transform the target string to an integer array
    """
	
	def __init__(self):
		self.dic = esposalles_dict()
		
    def __call__(self, text):
        return esposalles_char(text)


def esposalles_char(text):
    return [self.dic.get(c) for c in text]
    
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
