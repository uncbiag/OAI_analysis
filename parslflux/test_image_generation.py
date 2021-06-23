import pickle
import sys
import random
import string

def test_image_generation(quantity):
    images = []
    letters = string.ascii_lowercase
    print ('generating test images')
    for index in range (0, quantity):
        images.append([str(index), ''.join(random.choice(letters) for i in range(10)), ''.join(random.choice(letters) for i in range(10))])
    pickle.dump(images, open("images.p", "wb"))

if __name__ == "__main__":
    quantity = int(sys.argv[1])
    test_image_generation(quantity)