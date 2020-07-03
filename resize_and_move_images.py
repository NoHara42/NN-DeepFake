from PIL import Image
import os
import shutil

"""
image = Image.open("Sion.png")
new_image = image.resize((64,64))
print(new_image.size)
new_image.save('image_thumbnail.png')

"""

#path = os.getcwd()
#print ("The current working directory is %s" % path)
n = 0

for file in os.listdir("D:\\DCGAN_tutorial\\lol_champs\\lol_data"):
    file_to_resize = "D:\\DCGAN_tutorial\\lol_champs\\lol_data\\" + file
    image = Image.open(file_to_resize).convert('RGB')
    new_image = image.resize((64,64))
    new_image.save('D:\\DCGAN_tutorial\\lol_champs\\1.png')
    folder_to_move_to = 'D:\\DCGAN_tutorial\\lol_champs\\dataset\\' + str(n) + '\\1.jpg'
    shutil.move('D:\\DCGAN_tutorial\\lol_champs\\1.png', folder_to_move_to)
    n += 1

