class palette():
	"""
		It gives a list of colors to be used by the library plot.ly
	"""

	def __init__(self):

		#From Android Palette https://material.io/guidelines/style/color.html#color-color-palette

		self._COLORS = {
			"red":			{100:"#FFCDD2", 200:"#EF9A9A", 300:"#E57373", 400:"#EF5350", 500:"#F44336", 600:"#E53935", 700:"#D32F2F", 800:"#C62828", 900:"#B71C1C"},
			"pink":			{100:"#F8BBD0", 200:"#F48FB1", 300:"#F06292", 400:"#EC407A", 500:"#E91E63", 600:"#D81B60", 700:"#C2185B", 800:"#AD1457", 900:"#880E4F"},
			"purple":		{100:"#E1BEE7", 200:"#CE93D8", 300:"#BA68C8", 400:"#AB47BC", 500:"#9C27B0", 600:"#8E24AA", 700:"#7B1FA2", 800:"#6A1B9A", 900:"#4A148C"},
			"deep purple":	{100:"#D1C4E9", 200:"#B39DDB", 300:"#9575CD", 400:"#7E57C2", 500:"#673AB7", 600:"#5E35B1", 700:"#512DA8", 800:"#4527A0", 900:"#311B92"},
			"indigo":		{100:"#C5CAE9", 200:"#9FA8DA", 300:"#7986CB", 400:"#5C6BC0", 500:"#3F51B5", 600:"#3949AB", 700:"#303F9F", 800:"#283593", 900:"#1A237E"},
			"blue":			{100:"#BBDEFB", 200:"#90CAF9", 300:"#64B5F6", 400:"#42A5F5", 500:"#2196F3", 600:"#1E88E5", 700:"#1976D2", 800:"#1565C0", 900:"#0D47A1"},
			"light blue":	{100:"#B3E5FC", 200:"#81D4FA", 300:"#4FC3F7", 400:"#29B6F6", 500:"#03A9F4", 600:"#039BE5", 700:"#0288D1", 800:"#0277BD", 900:"#01579B"},
			"cyan":			{100:"#B2EBF2", 200:"#80DEEA", 300:"#4DD0E1", 400:"#26C6DA", 500:"#00BCD4", 600:"#00ACC1", 700:"#0097A7", 800:"#00838F", 900:"#006064"},
			"teal":			{100:"#B2DFDB", 200:"#80CBC4", 300:"#4DB6AC", 400:"#26A69A", 500:"#009688", 600:"#00897B", 700:"#00796B", 800:"#00695C", 900:"#004D40"},
			"green":		{100:"#C8E6C9", 200:"#A5D6A7", 300:"#81C784", 400:"#66BB6A", 500:"#4CAF50", 600:"#43A047", 700:"#388E3C", 800:"#2E7D32", 900:"#1B5E20"},
			"light green":	{100:"#DCEDC8", 200:"#C5E1A5", 300:"#AED581", 400:"#9CCC65", 500:"#8BC34A", 600:"#7CB342", 700:"#689F38", 800:"#558B2F", 900:"#33691E"},
			"lime":			{100:"#F0F4C3", 200:"#E6EE9C", 300:"#DCE775", 400:"#D4E157", 500:"#CDDC39", 600:"#C0CA33", 700:"#AFB42B", 800:"#9E9D24", 900:"#827717"},
			"yellow":		{100:"#FFF9C4", 200:"#FFF59D", 300:"#FFF176", 400:"#FFEE58", 500:"#FFEB3B", 600:"#FDD835", 700:"#FBC02D", 800:"#F9A825", 900:"#F57F17"},
			"amber":		{100:"#FFECB3", 200:"#FFE082", 300:"#FFD54F", 400:"#FFCA28", 500:"#FFC107", 600:"#FFB300", 700:"#FFA000", 800:"#FF8F00", 900:"#FF6F00"},
			"orange":		{100:"#FFE0B2", 200:"#FFCC80", 300:"#FFB74D", 400:"#FFA726", 500:"#FF9800", 600:"#FB8C00", 700:"#F57C00", 800:"#EF6C00", 900:"#E65100"},
			"deep orange":	{100:"#FFCCBC", 200:"#FFAB91", 300:"#FF8A65", 400:"#FF7043", 500:"#FF5722", 600:"#F4511E", 700:"#E64A19", 800:"#D84315", 900:"#BF360C"},
			"brown":		{100:"#D7CCC8", 200:"#BCAAA4", 300:"#A1887F", 400:"#8D6E63", 500:"#795548", 600:"#6D4C41", 700:"#5D4037", 800:"#4E342E", 900:"#3E2723"},
			"grey":			{100:"#F5F5F5", 200:"#EEEEEE", 300:"#E0E0E0", 400:"#BDBDBD", 500:"#9E9E9E", 600:"#757575", 700:"#616161", 800:"#424242", 900:"#212121"},
			"blue grey":	{100:"#CFD8DC", 200:"#B0BEC5", 300:"#90A4AE", 400:"#78909C", 500:"#607D8B", 600:"#546E7A", 700:"#455A64", 800:"#37474F", 900:"#263238"}
		}

	def decode_hex(self, hex_text):
		"""
			Transform hex string to a RGB list

			Args:
				hex_text:	hex code

			Returns:
				list of 3 numbers representing an RGB color
		"""

		import struct

		if hex_text[:1] == "#":
			hex_text = hex_text[1:]

		return struct.unpack('BBB', hex_text.decode('hex'))

	def get_colors_by_index(self, list_of_colors):
		"""
			Gives a list of colors using a list of index of the colors needed

			Args:
				list_of_colors:	list of indexs of the colors to be used

			Returns:
				list of colors (plot.ly format)
		"""
		
		output = []

		for color, index in list_of_colors:
			output.append(self._COLORS[color.lower()][index])


		#If there is only 1 color it shouldn't return a list
		return output if len(list_of_colors) > 1 else output[0]

	def get_colors_by_num(self, num_of_colors):
		"""
			Gives a list of colors with num_of_colors elements

			Args:
				num_of_colors:	how many colors are needed

			Returns:
				list of colors (plot.ly format)
		"""

		m_colors = [("light blue", 500), ("grey", 500), ("green", 500), ("yellow", 500), ("brown", 500), ("purple", 500), ("amber", 500), ("indigo", 500), ("deep orange", 500)]

		return self.get_colors_by_index(m_colors[:num_of_colors])