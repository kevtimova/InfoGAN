image = require 'image'
require 'torch'
require 'io'

-- Parameters
img_size = 28
str_len = 4
num_imgs = torch.floor(img_size/str_len)

-- Generate white vertical stripes on black background
for i=1, num_imgs do
	img = torch.Tensor(3,img_size,img_size):zero()
	img[{{}, {}, {4*(i-1)+1,4*i}}]:fill(255)
	image.save("img"..i..".jpg", img)
end