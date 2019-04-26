import Augmentor

p = Augmentor.Pipeline("/Users/leonardotanzi/Desktop/MasterThesis/PreProcessing-DataAgumentation/ToAugment")

# The probability parameter controls how often the operation is applied. 

# p.skew_tilt(probability=1.0)
# p.zoom(probability=0.5, min_factor=1.1, max_factor=1.2)
p.rotate(probability=0.5, max_left_rotation=10, max_right_rotation=10)
p.random_distortion(1.0, 16, 16, 8)
p.flip_left_right(probability=0.5)
p.shear(0.5, 7, 7)
p.gaussian_distortion(0.8, 16, 16, 8, "bell", "in")
p.histogram_equalisation(0.1)
p.random_brightness(0.9, 0.5, 1.5)
p.random_contrast(0.7, 0.5, 1.0)
p.resize(1.0, 256, 256)

p.sample(100)