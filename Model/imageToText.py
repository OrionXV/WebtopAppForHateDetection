import pytesseract
from PIL import Image, ImageEnhance, ImageFilter

def imageToText(path):
    im = Image.open(path) # the second one 
    im = im.filter(ImageFilter.MedianFilter())
    enhancer = ImageEnhance.Contrast(im)
    im = enhancer.enhance(2)
    im = im.convert('1')
    newpath = path[:-3] + 2 + path[-3:]
    im.save(newpath)
    text = pytesseract.image_to_string(Image.open(newpath))
    return text        
