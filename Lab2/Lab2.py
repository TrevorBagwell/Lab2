
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import random
from PIL import Image
#from scipy.misc import imresize
#from scipy.misc import imsave
from scipy.optimize import fmin_l_bfgs_b   # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html
from tensorflow.keras.applications import vgg19
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import warnings


# To make things deteriororatstic? That doesn't sound right
# Something about make a descision. It begins with a "D". 
# Determinable. Detrimant. Something idk rn.
random.seed(1618)
np.random.seed(1618)
tf.set_random_seed(1618)


tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# I have multiple content images and Style pictures so I will just simply assign a number
# to a variable for both and then append that to the generic content and style path format
content_number = 0
style_number = 0

cwd = os.getcwd()


# Path of my content image
CONTENT_IMG_PATH = cwd + "/Images/Content" + str(content_number) + ".jpg"

# Path of my style image         
STYLE_IMG_PATH = cwd + "/Images/Style" + str(style_number) + ".jpg"             

# The height and width of the content image
CONTENT_IMG_H = 500
CONTENT_IMG_W = 500

# The height and width of the style image
STYLE_IMG_H = 500
STYLE_IMG_W = 500

CONTENT_WEIGHT = 0.1    # Alpha weight. This weight reigns dominant and will go on to reproduce.
STYLE_WEIGHT = 1.0      # Beta weight. This weight is submissive and will not go on to reproduce.
TOTAL_WEIGHT = 1.0


# TODO: Figure out what this does
TRANSFER_ROUNDS = 3


#NVM, he already made these for us

# Content layers where we will taking the output from
#content_layers = ['block5_conv2']

# Style layer's we will be looking at
#style_layers = ['block1_conv1','block2_conv1','block3_conv1','block4_conv1','block5_conv1']



#=============================<Helper Fuctions>=================================
'''
TODO: implement this.
This function should take the tensor and re-convert it to an image.
'''
def deprocessImage(img):
    
    # TODO: Implement this.
    return img

# Conduxts gram matrix execution on the code allowing us to make a gram matrix out of x
def gramMatrix(x):
    
    # TODO: Figure out what this does
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    
    # We dot the features with it's transpose to do sum of squares
    gram = K.dot(features, K.transpose(features))
    
    # Returns the gram Matrix equivalent for the matrix
    return gram



#========================<Loss Function Builder Functions>======================

#
def styleLoss(style, gen):
    
    # TODONE??: Implement
    
    return K.sum(K.square(gramMatrix(style) - gramMatrix(gen)))/( 4. * (STYLE_IMG_H*STYLE_IMG_W) ** 2 )

# Gets the loss for the content image based of of gen
def contentLoss(content, gen):
    
    # This implements a simple loss function for the content using Euclidean distance squared
    return K.sum(K.square(gen - content))

# Gets the total loss for x
# What the fuck is x?
def totalLoss(x):

    # TODO: Implement

    return None




#=========================<Pipeline Functions>==================================

# Loads the image
def getRawData():

    # Prints off an update to the user
    print("   Loading images.")
    
    # Prints off the path of the content image
    print("      Content image URL:  \"%s\"." % CONTENT_IMG_PATH)
    
    # Prints off the path of the style image
    print("      Style image URL:    \"%s\"." % STYLE_IMG_PATH)
    
    # This loads the content image
    cImg = load_img(CONTENT_IMG_PATH)

    w, h = cImg.size

    print("Shape of the content image: (" + str(w) + "," + str(h) + ")" )
    
    # No idea why we copy the image but there you go
    tImg = cImg.copy()

    # This loads the style image
    sImg = load_img(STYLE_IMG_PATH)
    
    # Prints off an update to the user that:
    # Ladies and gentlemen, we got em
    print("      Images have been loaded.")
    
    # Return an array containing each image and it's metadata
    return ((cImg, CONTENT_IMG_H, CONTENT_IMG_W), (sImg, STYLE_IMG_H, STYLE_IMG_W), (tImg, CONTENT_IMG_H, CONTENT_IMG_W))





#
def preprocessData(raw):
    
    # TODO: Figure out how to resize the image using keras
    

    # Holds the image matrix and the desired image dimensions
    img, ih, iw = raw
    
    # Converts the 2 dimensional object to a datastream
    img = img_to_array(img)
    
    # Hahahaha, wtF?
    with warnings.catch_warnings():
        
        warnings.simplefilter("ignore")
        
        # Resizes the imaged to something with the height width and RGB colour scheme
        img = img.resize((ih, iw, 3))
    
    # Converts all items in the image to floating point numbers
    img = img.astype("float64")
    
    # Adds a dimension to the image at the beginning of the image tensor
    img = np.expand_dims(img, axis=0)
    
    # TODO: Figure out what this does
    img = vgg19.preprocess_input(img)
    
    # Img is the nicely adjusted image 
    return img



'''
TODO: Allot of stuff needs to be implemented in this function.

First, make sure the model is set up properly.

Then construct the loss function (from content and style loss).

Gradient functions will also need to be created, or you can use K.Gradients().
Finally, do the style transfer with gradient descent.
Save the newly generated and deprocessed images.
'''
def styleTransfer(cData, sData, tData):
    
    # Prints out an update to the user to tell it that the style transfer process has started
    print("   Building transfer model.")
    
    # Makes a tensor off the content data stream
    contentTensor = K.variable(cData)
    
    # Makes a tensor off the style data stream
    styleTensor = K.variable(sData)
    
    # This is the generic starter tensor
    genTensor = K.placeholder((1, CONTENT_IMG_H, CONTENT_IMG_W, 3))
    
    # This is the tensor we will put in the model?
    inputTensor = K.concatenate([contentTensor, styleTensor, genTensor], axis=0)
    
    

    # TODONE: Implement the model

    # Loads the image in the vgg19

    # Gets the vgg19 model
    model = tf.keras.applications.VGG19(include_top=False,weights='imagenet')
    
    # Excuse me... WTF? What the fuck is this syntax legal for?
    outputDict = dict([(layer.name, layer.output) for layer in model.layers])
    
    # Get's the probabilities of the model
    

    print("   VGG19 model loaded.")
    # This holds the loss of the model
    loss = 0.0
    
    # These are the desired style layers that we will be looking at
    styleLayerNames = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"]
    
    # This is the desired layer that we will extract our content from
    contentLayerName = "block5_conv2"
    


    print("   Calculating content loss.")
    
    contentLayer = outputDict[contentLayerName]

    # This is the content layer, the first image we put in
    contentOutput = contentLayer[0, :, :, :]
    
    # This is the generator layer, the 3rd image we put in
    genOutput = contentLayer[2, :, :, :]
    
    #loss += None   #TODO: implement.
    loss += CONTENT_WEIGHT*contentLoss(contentOutput,genOutput)
    
    print("   Calculating style loss.")
    
    for layerName in styleLayerNames:
        
        # Get the layer with the given layername
        styleLayer = outputDict[layerName]

        # This is the style layer, the first image we put in
        styleOutput = styleLayer[1, :, :, :]
    
        # This is the generator layer, the 3rd image we put in
        genOutput = styleLayer[2, :, :, :]

        # TODO: Figure out why we need to add 2 things to the loss variable instead of 1

        loss += STYLE_WEIGHT*styleLoss(styleOutput,genOutput)
    
    
    # TODO: Setup gradients or use K.gradients().
    
    print("   Beginning transfer.")
    
    for i in range(TRANSFER_ROUNDS):
    
    
        print("   Step %d." % i)
    
        #TODO: perform gradient descent using fmin_l_bfgs_b.
        tLoss = 0
    
        print("      Loss: %f." % tLoss)

    
        img = deprocessImage(x)
    
        # This is the save file and it saves the transfer with the content number and the style number
        saveFile = "Images/Transfer_c" + content_number + "_s" + style_number + ".jpg"
    
        #imsave(saveFile, img)   #Uncomment when everything is working right.

        # Prints off the image file that the saved transfer image is
        print("      Image saved to \"%s\"." % saveFile)
    
    
    # Prints off an update to the user that the transfer is complete
    print("   Transfer complete.")





#=========================<Main>================================================
# This is the main handling function
def main():
    print("Starting style transfer program.")

    # Loads the images and then 
    raw = getRawData()

    # This processes the content image and converts it to a nice easy to use datastream
    cData = preprocessData(raw[0])
    
    # This processes the stylr image and converts it to a nice easy to use datastream
    sData = preprocessData(raw[1]) 
    
    # This processes the transferable(originally just content) image and converts it 
    # to a nice easy to use datastream
    tData = preprocessData(raw[2]) 
    
    # This transfers the style to the content image and then saves it in a file
    styleTransfer(cData, sData, tData)

    # This indicates to the user that we are infact done
    print("Done. Goodbye.")


# This is the main function
if __name__ == "__main__":
    main()