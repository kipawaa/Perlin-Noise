# generates a map using perlin noise

# colours are poorly done, could use better shading and more intelligent landscape design :(

# library imports
import numpy as np
from PIL import Image

# personal imports
from perlin_noise import noise

#width and height define the resolution of the array, minNoise and maxNoise define how many boxes (the corners of which are used to generate vectors that create the noise) are used to create the noise
def generateHeights(width, height, numLayers, minNoiseRes=2, maxNoiseRes=15):
	heightMap = np.zeros((height, width))
	for i in range(numLayers):
		# used to determine the noise resolution
		numBoxes = (maxNoiseRes // numLayers) + minNoiseRes
		# generates a new layer of noise and adds the value to the height map (scaled)
		heightMap += (1 - (i / numLayers)) * noise(width, height, numBoxes * width // height, numBoxes * height // width)
	
	# normalizes the map to the range(0, 255)
	heightMap -= heightMap.min()
	heightMap = (heightMap / heightMap.max() * 255).astype(np.uint8)
	return heightMap

def makeImage(height_map):
	# stores the colour values of the map
	img = np.zeros((height, width, 3), 'uint8')

	for x in range(width):
		for y in range(height):
			# snowy caps and rocky mountains
			if height_map[y, x] > 225:
				# same values but scaled within range(128, 255)
				img[y, x] = ((heightMap[y, x] - 200) * 4.5, (heightMap[y, x] - 128) * 4.5, (heightMap[y, x] - 128) * 4.5)
			# attempt at making things pretty green colours
			elif height_map[y, x] > 80:
				img[y, x] = (50, ((-(height_map[y, x] - 80) / (225-80)) + 1) * 200 + 50, 50)
			# watery depths
			elif height_map[y, x] <= 80:
				img[y, x] = (50, 50, (height_map[y, x] / 80) * 150)
	return img

if __name__ == '__main__':
	# resolution of the map
	userChoice = 0
	if userChoice == 1:
		width = int(input("input the width (in pixels) of the map you would like rendered: "))
		height = int(input("input the height (int pixels) of the map you would like rendered: "))
	else:
		width = 1920
		height = 1080
	
	# generates the heights
	print("generating heights")
	heightMap = generateHeights(width, height, 3)
	img = makeImage(heightMap)
	
	# creates an image from the map
	print("generating image")
	Image.fromarray(img).show()
	
	print("complete")
