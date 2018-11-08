class Beam:
	"""
	Contains probabilities of a single beam
	The label are linked in class beams
	"""
	def __init__(self):
		self.pb = 0 # Probability for a beam ending with a Blank
		self.pnb = 0 # Probability for a beam ending with a Non Blank
		self.ptot = 0 # Total Probability (ptot = pb + pnb)
	
	def add_pb(self, value):
		self.pb += value
		self.ptot += value
	
	def add_pnb(self, value):
		self.pnb += value
		self.ptot += value
	
	def get_score(self):
		return self.ptot


class Beams:
	"""
	Stocks each label and associate to them a beam containing the probabilities
	"""
	def __init__(self):
		self.beams = {}
	
	def add_beam(self, label):
		self.beams[label] = Beam()
		
	def get_beam(self, label):
		return self.beams[label]
	
	def best_beams(self, beam_width):
		"""
		Computes the N best beams where N is the beam width
		It returns a sorted list of labels
		"""
		
		best_beams = []
		nbr_beams = 0
		
		for label in self.beams:
			score = self.get_beam(label).get_score()
			
			# Start by adding the current label if possible
			if nbr_beams < beam_width:
				best_beams.append(label)
				nbr_beams += 1
			elif score > self.get_beam(best_beams[beam_width-1]).get_score():
				best_beams.append(label)
				nbr_beams += 1
			
			# Compare it with currently stored beams
			for i in range(nbr_beams - 1, 0, -1):
				if score > self.get_beam(best_beams[i-1]).get_score():
					best_beams[i] = best_beams[i-1]
					best_beams[i-1] = label
				else:
					break
			
			# If the beam is to large, remove the last one
			if nbr_beams > beam_width:
				best_beams.pop()
				nbr_beams -= 1
			
		return best_beams
			
	def keep_best_beams(self, best_beams):
		for label in self.beams:
			if label not in best_beams:
				del self.beams[label]	
	
	def is_in(self, label):
		return label in self.beams	


def beam_search(probs, char_list, max_len=-1, beam_width=10, blank_index=0):
	nbr_char = len(char_list)
	
	# Initialize with only the beam "" with a probability of finishing by a blank of 1
	beams = Beams()
	beams.add_beam("")
	beams.get_beam("").add_pb(1)
	
	# Get the max_len, maximum number of algorithm iterations
	# The probability matrix could cover more cases with padding
	if max_len == -1:
		max_len = len(probs)
	else:
		max_len = min(max_len, len(probs))
	
	for t in range(max_len):
		best_beams = beams.best_beams(beam_width)
		beams.keep_best_beams(best_beams) # Now beam contains only the N best beams (where N = beeam_width)
		
		curr_beams = Beams()
		
		for beam_label in best_beams:
			curr_beam = beams.get_beam(beam_label)
			
			for i in range(nbr_char):
				
				
				if i == blank_index:
					# Extend the current beam with a blank
					curr_label = beam_label
					
					if not curr_beams.is_in(curr_label):
						curr_beams.add_beam(curr_label)
					
					curr_beams.get_beam(curr_label).add_pb(beams.get_beam(beam_label).ptot * probs[t][i])
				
				elif len(beam_label) > 0 and beam_label[-1] == char_list[i]:
					# Extend the beam with the same last letter
					# Case 1 : the path finished by the letter
					curr_label = beam_label
					
					if not curr_beams.is_in(curr_label):
						curr_beams.add_beam(curr_label)
					
					curr_beams.get_beam(curr_label).add_pnb(beams.get_beam(beam_label).pnb * probs[t][i])
					
					# Case 2 : the path finished by a blank
					curr_label = beam_label + char_list[i]
					
					if not curr_beams.is_in(curr_label):
						curr_beams.add_beam(curr_label)
					
					curr_beams.get_beam(curr_label).add_pnb(beams.get_beam(beam_label).pb * probs[t][i])
				
				else:
					#extend the beam with a letter that is not the last one
					curr_label = beam_label + char_list[i]
					
					if not curr_beams.is_in(curr_label):
						curr_beams.add_beam(curr_label)
					
					curr_beams.get_beam(curr_label).add_pnb(beams.get_beam(beam_label).ptot * probs[t][i])
			
		beams = curr_beams
	
	best_beams = beams.best_beams(1)
	
	return best_beams[0], beams.get_beam(best_beams[0]).get_score()
