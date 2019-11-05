#!/usr/bin/python
#
# Removal of periodic features using the FFT
#
# Use Python 2.7 with these packages: numpy, PyOpenGL, Pillow

## Quinn Pollock 20018131
## Jack Guinane 20018078

import sys, os, math, pprint

import numpy as np

from PIL import Image, ImageOps

from OpenGL.GLUT import *
from OpenGL.GL import *
from OpenGL.GLU import *


# Globals

windowWidth  = 1000 # window dimensions (not image dimensions)
windowHeight =  800

showMagnitude = True            # for the FT, show the magnitude.  Otherwise, show the phase
doHistoEq = False               # do histogram equalization on the FT to make features more obvious

texID = None                    # for OpenGL

zoom = 1.0                      # amount by which to zoom images
translate = (0.0,0.0)           # amount by which to translate images


# Image

imageDir      = 'images'
imageFilename = 'small.png'
imagePath     = os.path.join( imageDir, imageFilename )

image    = None                 # the image as a 2D np.array
imageFT  = None                 # the image's FT as a 2D np.array

gridImage   = None              # the grid, isolated from the image
gridImageFT = None              # the grid's FT

resultImage = None              # the final image


# Remove the grid from the global 'image'.  Return the result image
# AND a list of [ [angle1,distance1], [angle2,distance2] ] describing
# the two principal grid lines.
#
# The angle is the angle, in degrees, of the grid line from the horizontal.
#
# The distance is the distance from the origin, in pixels, of the
# first peak in the Fourier Transform corresponding to the lines at
# the given angle.  This will later be used to calculate the line
# spacing.
#
# Do the following in the compute() function:
#
#   1. Compute the FT of the image.  Store it in 'imageFT'.
#
#   2. Compute and store the FT magnitudes.  Find the maximum
#      magnitude, EXCLUDING the DC component in [0,0]. 
#
#   3. Set to zero the components of 'imageFT' that have magnitude
#      less than 40% the maximum magnitude.  Store this new FT in
#      'gridImageFT'.  Record in a list the (x,y) locations of the
#      non-zero magnitudes of 'gridImageFT'.
#
#   4. From the locations of the non-zero magnitudes, find the angles
#      of the two principal grid lines and, for each such line, find
#      the distance of the closest non-zero magnitude to the origin.
#
#      THIS IS DIFFICULT and can be left until you have zeroed the grid
#      line pixels as described in Step 6.
#
#   5. Apply the inverse FT to 'gridImageFT' to get 'gridImage'.
#
#   6. For each (x,y) location in 'gridImage' that has a bright pixel
#      of value > 16 (i.e. is one of the grid lines), set the
#      corresponding pixel in the original 'image' to the average of
#      the pixels on either side of the grid line that are not also
#      grid line pixels.  Do not modify 'image'; instead, store your
#      result in 'resultImage'.
#
#      FINDING THE AVERAGE IS DIFFICULT, so first set the grid line
#      pixels to zero and debug your code to make sure that this is
#      working.  Only after this is working should you try to set the
#      grid line pixels to the average from either side.  You will need 
#      to know the angles from Step 4 to do this.
#
# Test your code on all files in the 'images' directory.  Debug using the 'small.png' file.
#
# DO NOT USE NUMPY for anything but the simplest operations, and for
# the FT and inverse FT.  In the compute() function below, you should
# always iterate over the image in your own code, rather than call
# some NumPy function to do the iteration for you.


def compute():

  global image, imageFT, gridImage, gridImageFT, resultImage

  height = image.shape[0]
  width  = image.shape[1]

  # Forward FT

  print '1. compute FT'
  print type(image)
  imageFT = forwardFT(image)
  height_ft = imageFT.shape[0]
  width_ft = imageFT.shape[1]
  print type(imageFT)
  print height
  print width
  print height_ft
  print width_ft
  
      
  # Compute magnitudes and find the maximum (excluding the DC component)

  print '2. computing FT magnitudes'
  max = 0 
  newImage = np.copy(imageFT)
#   print newImage
  actualimage_ft = np.transpose(newImage);
  print newImage
  print actualimage_ft
  for i in range (width_ft):
    for j in range (height_ft):
#       print i
#       print j
      pixel = actualimage_ft[i][j]
#       print pixel
      if max < pixel: 
        if i != 0 and j != 0:
            max = pixel
  max = np.sqrt(np.real(max)**2 + np.imag(max)**2)
  print max
  filter_max = max * 0.13
  print filter_max
  
  
  
    
#   Zero the components that are less than 40% of the max

  print '3. removing low-magnitude components'
  # create copy of image to find dots
  image_copy = np.copy(actualimage_ft)
  
  # Get the half the size of the image for for loop as image is centred at 0 
  val_x = getSize(actualimage_ft)[0]
  val_y = getSize(actualimage_ft)[1]
#   print val_y
  # if either x dimension or y dimension was odd add 1 to it so you dont skip the edge
  if val_x[1] == 1:
    range_x = val_x[0] + 1
  else:
    range_x = val_x[0]
  if val_y[1] == 1:
    print val_y[0]
    range_y = val_y[0] + 1
  else:
    range_y = val_y[0]

#   print range_y
  
  # for loop through image to remove all thing less the filter_max
  for i in range(-range_x,range_x):
    for j in range(-range_y,range_y):
      if abs(i) > 50 or abs(j) > 50:
        actualimage_ft[i][j] = actualimage_ft[i][j] * 0.5
      elif abs(i) > 100 or abs(j) > 100:
        actualimage_ft[i][j] = 0
  gridImageFT = np.copy(np.transpose(actualimage_ft))
  if gridImageFT is None:
    gridImageFT = np.zeros( (height,width), dtype=np.complex_ )
 
 
  print '4. finding angles and distances of grid lines'

  # list of all high magnitudes points
  list_of_all = []
  # get all dots from image in to a list to find angle and distance
  for i in range(-range_x,range_x):
    for j in range(-range_y,range_y):
      if getMaginitude(image_copy[i][j]) >= filter_max:
         list_of_all.append((j,i))
      
  # get the max point in x-axis and y-axis    
  y_max = (0,0)
  x_max = (0,0)  
  for i in list_of_all:
    if y_max[0] < i[0]:
      y_max = i
    if x_max[1] < i[1]:
      x_max = i
  
  # the angle is the line drawn from the max-point to the center
  # this is found using arctan
  angle_x = np.arctan2(x_max[0],x_max[1]) * 180 / np.pi
  angle_y = np.arctan2(y_max[0],y_max[1]) * 180 / np.pi

  # list of points on the y-axis (used to find distance)
  list_of_locy = []
  # list of points on the x-axis
  list_of_locx = []
  
  for i in list_of_all:
    # all points that are on an angle (to center) within 5 degrees of angle_x are added to x-locations
    # otherwise add to y-locations
    if abs((np.arctan2(i[0], i[1]) * 180 / np.pi) - angle_x) > 5:
      list_of_locy.append(i)
    else:
      list_of_locx.append(i)
  
  # get distances between points for each line
  dis_x = get_avg_distance(list_of_locx)
  dis_y = get_avg_distance(list_of_locy)


  lines = [[angle_x,dis_x],[angle_y,dis_y]]

  # Convert back to spatial domain to get a grid-like image

  print '5. inverse FT'
  
  gridImage = inverseFT(gridImageFT)

  if gridImage is None:
    gridImage = np.zeros( (height,width), dtype=np.complex_ )

  # Remove grid image from original image
  resultImage = image.copy()

  print '6. remove grid'
  for i in range(height):
    for j in range(width):
      if gridImage[i][j] > 0.1*255:
        val = get_replace_val(i,j)
        resultImage[i][j] = val
        
#       
#   if resultImage is None:
#     resultImage = image.copy()

  print 'done'

  return resultImage, lines


      

# File dialog

if sys.platform != 'darwin':
    import Tkinter, tkFileDialog
    root = Tkinter.Tk()
    root.withdraw()


# Do a forward FT
#
# Input is a 2D numpy array of complex values.
# Output is the same.

def forwardFT( image ):

  return np.fft.fft2( image )



# Do an inverse FT
#
# Input is a 2D numpy array of complex values.
# Output is the same.


def inverseFT( image ):

  return np.fft.ifft2( image )

# Get magitude of value from real of and imaginary component
def getMaginitude(pixel):
  return np.sqrt(np.real(pixel)**2 + np.imag(pixel)**2)

# Get half the size of image and if it was odd in x or y
def getSize(image):
  x = image.shape[0]
  y = image.shape[1]
  if x % 2 == 0:
    size_x = x/2
    val_x = (size_x, 0)
  else:
    size_x = x/2
    val_x = (size_x, 1)
  if y % 2 == 0:
    size_y = y/2
    val_y = (size_y, 0)
  else:
    size_y = y/2
    val_y = (size_y, 1)
    
  return (val_x,val_y)
  
# Get average of list
def avg(l):
    if l:
        return sum(l)/len(l)
    return 0
   

def get_avg_distance(lis):
  
  # returns average distance between points (tuples) in a list
  
  count = 0
  total = 0
  
  for i in range(len(lis)-2):
    # find distance using pythag-theory (a^2 + b^2 = c^2)
    val = np.sqrt((abs(lis[i][0]-lis[i+1][0]))**2 + (abs(lis[i][1]-lis[i+1][1]))**2)
    # excludes distances between points that are next to each other
    # as to not include points that are made out of multipul pixels
    if val > 1.5:
      count += 1
      total += val
  # avoid divide by zero errors
  if count == 0:
    return 0
  # return average distance
  return total/count



def get_replace_val(i,j):
  
  # get pixel values to fill in broken line
  # returns the brightest pixel in a 1-pixel neighbourhood, not on a gridline

  global gridImage, image
  height = image.shape[0]
  width  = image.shape[1] 
  
  # build 1-neighbourhood filter
  line_filter = [(i, j+1), (i+1, j),(i+1, j+1),(i-1, j-1),(i-1, j),(i, j-1),(i+1, j-1),(i-1, j+1)]
  # remove any pixels outside of image bounds
  line_filter = filter(lambda x: x[0] < height-2 and x[0] > 0 and x[1]< width-2 and x[1] > 0, line_filter)
  
  # get all values that are not on the grid
  vals = [image[r[0]][r[1]] for r in line_filter if gridImage[r[0]][r[1]] <= 0.1*255]
  
  # return 0 if no values found
  if len(vals) == 0:
    return 0 
    
  # return brightest pixel in list
  return np.max(vals)



# Set up the display and draw the current image


def display():

  # Clear window

  glClearColor ( 1, 1, 1, 0 )
  glClear( GL_COLOR_BUFFER_BIT )

  glMatrixMode( GL_PROJECTION )
  glLoadIdentity()

  glMatrixMode( GL_MODELVIEW )
  glLoadIdentity()
  glOrtho( 0, windowWidth, 0, windowHeight, 0, 1 )

  # Set up texturing

  global texID
  
  if texID == None:
    texID = glGenTextures(1)

  glPixelStorei( GL_UNPACK_ALIGNMENT, 1 )
  glBindTexture( GL_TEXTURE_2D, texID )

  glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE)
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
  glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, [1,0,0,1] );

  # Images to draw, in rows and columns

  toDraw, rows, cols, maxHeight, maxWidth, scale, horizSpacing, vertSpacing = getImagesInfo()

  for r in range(rows):
    for c in range(cols):
      if toDraw[r][c] is not None:

        if r == 0: # normal image in row 0 
          img = toDraw[r][c]
        else: # FT in column 1
          img = np.fft.fftshift( toDraw[r][c] ) # shift FT so that origin is in centre (just for display)

        height = scale * img.shape[0]
        width  = scale * img.shape[1]

        # Find lower-left corner

        baseX = (horizSpacing + maxWidth ) * c + horizSpacing
        baseY = (vertSpacing  + maxHeight) * (rows-1-r) + vertSpacing

        # Get pixels and draw

        if r == 0: # for images (in row 0), show the real part of each pixel
          show = np.real(img)
        else: # for FT (in column 1), show magnitude or phase
          ak =  2 * np.real(img)
          bk = -2 * np.imag(img)
          if showMagnitude:
            show = np.log( 1 + np.sqrt( ak*ak + bk*bk ) ) # take the log because there are a few very large values (e.g. the DC component)
          else:
            show = np.arctan2( -1 * bk, ak )

          if doHistoEq and c > 0:
            show = histoEq( show ) # optionally, perform histogram equalization on FT image (but this takes time!)

        # Put the image into a texture, then draw it

        max = show.max()
        min = show.min()
        if max == min:
          max = min+1
          
        imgData = np.array( (np.ravel(show) - min) / (max - min) * 255, np.uint8 )

        glTexImage2D( GL_TEXTURE_2D, 0, GL_INTENSITY, img.shape[1], img.shape[0], 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, imgData )

        # Include zoom and translate

        cx     = 0.5 - translate[0]/width
        cy     = 0.5 - translate[1]/height
        offset = 0.5 / zoom

        glEnable( GL_TEXTURE_2D )

        glBegin( GL_QUADS )
        glTexCoord2f( cx-offset, cy-offset )
        glVertex2f( baseX, baseY )
        glTexCoord2f( cx+offset, cy-offset )
        glVertex2f( baseX+width, baseY )
        glTexCoord2f( cx+offset, cy+offset )
        glVertex2f( baseX+width, baseY+height )
        glTexCoord2f( cx-offset, cy+offset )
        glVertex2f( baseX, baseY+height )
        glEnd()

        glDisable( GL_TEXTURE_2D )

        if zoom != 1 or translate != (0,0):
          glColor3f( 0.8, 0.8, 0.8 )
          glBegin( GL_LINE_LOOP )
          glVertex2f( baseX, baseY )
          glVertex2f( baseX+width, baseY )
          glVertex2f( baseX+width, baseY+height )
          glVertex2f( baseX, baseY+height )
          glEnd()

  # Draw image captions

  glColor3f( 0.2, 0.5, 0.7 )
 
  if image is not None:
    baseX = horizSpacing
    baseY = (vertSpacing  + maxHeight) * (rows) + 8
    drawText( baseX, baseY, imageFilename )

  if imageFT is not None:
    baseX = horizSpacing
    baseY = (vertSpacing  + maxHeight) * (rows-2) + vertSpacing - 18
    drawText( baseX, baseY, 'FT of %s' % imageFilename )

  if gridImage is not None:
    baseX = (horizSpacing + maxWidth) * 1 + horizSpacing
    baseY = (vertSpacing  + maxHeight) * rows + 8
    drawText( baseX, baseY, 'extracted grid' )

  if gridImageFT is not None:
    baseX = (horizSpacing + maxWidth) * 1 + horizSpacing
    baseY = (vertSpacing  + maxHeight) * (rows-2) + vertSpacing - 18
    drawText( baseX, baseY, 'FT of extracted grid' )

  if resultImage is not None:
    baseX = (horizSpacing + maxWidth) * 2 + horizSpacing
    baseY = (vertSpacing  + maxHeight) * (rows) + 8
    drawText( baseX, baseY, 'result' )

  # Draw mode information

  str = 'show %s' % ('magnitudes' if showMagnitude else 'phases')
  glColor3f( 0.5, 0.2, 0.4 )
  drawText( windowWidth-len(str)*8-8, 12, str )

  # Done

  glutSwapBuffers()

  

# Get information about how to place the images.
#
# toDraw                       2D array of complex images 
# rows, cols                   rows and columns in array
# maxHeight, maxWidth          max height and width of images
# scale                        amount by which to scale images
# horizSpacing, vertSpacing    spacing between images


def getImagesInfo():

  toDraw = [ [ image,   gridImage,   resultImage   ],
             [ imageFT, gridImageFT, None ] ]

  rows = len(toDraw)
  cols = len(toDraw[0])

  # Find max image dimensions

  maxHeight = 0
  maxWidth  = 0
  
  for row in toDraw:
    for img in row:
      if img is not None:
        if img.shape[0] > maxHeight:
          maxHeight = img.shape[0]
        if img.shape[1] > maxWidth:
          maxWidth = img.shape[1]

  # Scale everything to fit in the window

  minSpacing = 30 # minimum spacing between images

  scaleX = (windowWidth  - (cols+1)*minSpacing) / float(maxWidth  * cols)
  scaleY = (windowHeight - (rows+1)*minSpacing) / float(maxHeight * rows)

  if scaleX < scaleY:
    scale = scaleX
  else:
    scale = scaleY

  maxWidth  = scale * maxWidth
  maxHeight = scale * maxHeight

  # Draw each image

  horizSpacing = (windowWidth-cols*maxWidth)/(cols+1)
  vertSpacing  = (windowHeight-rows*maxHeight)/(rows+1)

  return toDraw, rows, cols, maxHeight, maxWidth, scale, horizSpacing, vertSpacing
  

  
# Equalize the image histogram

def histoEq( pixels ):

  # build histogram

  h = [0] * 256 # counts

  width  = pixels.shape[0]
  height = pixels.shape[1]

  min = pixels.min()
  max = pixels.max()
  if max == min:
    max = min+1

  for i in range(width):
    for j in range(height):
      y = int( (pixels[i,j] - min) / (max-min) * 255 )
      h[y] = h[y] + 1

  # Build T[r] = s

  k = 256.0 / float(width * height) # common factor applied to all entries

  T = [0] * 256 # lookup table
  
  sum = 0
  for i in range(256):
    sum = sum + h[i]
    T[i] = int( math.floor(k * sum) - 1 )
    if T[i] < 0:
      T[i] = 0

  # Apply T[r]

  result = np.empty( pixels.shape )

  for i in range(width):
    for j in range(height):
      y = int( (pixels[i,j] - min) / (max - min) * 255 )
      result[i,j] = T[y]

  return result
  

# Handle keyboard input

def keyboard( key, x, y ):

  global image, imageFT, gridImage, gridImageFT, resultImage, showMagnitude, doHistoEq, imageFilename, zoom, translate

  if key == '\033': # ESC = exit
    sys.exit(0)

  elif key == 'i':

    if sys.platform != 'darwin':
        imagePath = tkFileDialog.askopenfilename( initialdir = imageDir )
        if imagePath:
          image = loadImage( imagePath )
          imageFilename = os.path.basename( imagePath )
          imageFT = None
          gridImage = None
          gridImageFT = None
          resultImage = None

  elif key == 'm':
    showMagnitude = not showMagnitude

  elif key == 'h':
    doHistoEq = not doHistoEq

  elif key == 'z':
    zoom = 1
    translate = (0,0)

  elif key == 'c': # compute
    resultImage, lines = compute()
    print 'Grid lines:'
    for line in lines:
      print '  angle %.1f, distance %d' % (line[0],line[1])

  else:
    print '''keys:
           c  compute the solution
           m  toggle between magnitude and phase in the FT  
           h  toggle histogram equalization in the FT  
           i  load image
 right arrow  forward transform
  left arrow  inverse transform

              translate with left mouse dragging
              zoom with right mouse draggin up/down
           z  reset the translation and zoom'''

  glutPostRedisplay()


# Handle special key (e.g. arrows) input

def special( key, x, y ):

  if key == GLUT_KEY_DOWN:
    forwardFT_all()

  elif key == GLUT_KEY_UP:
    inverseFT_all()

  glutPostRedisplay()



# Do a forward FT to image


def forwardFT_all():

  global image, imageFT

  if image is not None:
    imageFT = forwardFT( image )



# Do an inverse FT to imageFT


def inverseFT_all():

  global image, imageFT

  if imageFT is not None: 
    image = inverseFT( imageFT )


    
# Load an image
#
# Return the image as a 2D numpy array of complex_ values.


def loadImage( path ):

  try:
    img = Image.open( path ).convert( 'L' ).transpose( Image.FLIP_TOP_BOTTOM )
  except:
    print 'Failed to load image %s' % path
    sys.exit(1)

  img = ImageOps.invert(img)

  return np.array( list( img.getdata() ), np.complex_ ).reshape( (img.size[1],img.size[0]) )



# Handle window reshape

def reshape( newWidth, newHeight ):

  global windowWidth, windowHeight

  windowWidth  = newWidth
  windowHeight = newHeight

  glViewport( 0, 0, windowWidth, windowHeight )

  glutPostRedisplay()



# Output an image
#
# The image has complex values, so output either the magnitudes or the
# phases, according to the 'outputMagnitudes' parameter.

def outputImage( image, filename, outputMagnitudes, isFT, invert ):

  if not isFT:
    show = np.real(image)
  else:
    ak =  2 * np.real(image)
    bk = -2 * np.imag(image)
    if outputMagnitudes:
      show = np.log( 1 + np.sqrt( ak*ak + bk*bk ) ) # take the log because there are a few very large values (e.g. the DC component)
    else:
      show = np.arctan2( -1 * bk, ak )
    show = np.fft.fftshift( show ) # shift FT so that origin is in centre

  min = show.min()
  max = show.max()

  img = Image.fromarray( np.uint8( (show - min) * (255 / (max-min)) ) ).transpose( Image.FLIP_TOP_BOTTOM )

  if invert:
    img = ImageOps.invert(img) 

  img.save( filename )




# Draw text in window

def drawText( x, y, text ):

  glRasterPos( x, y )
  for ch in text:
    glutBitmapCharacter( GLUT_BITMAP_8_BY_13, ord(ch) )

    

# Handle mouse click


currentButton = None
initX = 0
initY = 0
initZoom = 0
initTranslate = (0,0)

def mouse( button, state, x, y ):

  global currentButton, initX, initY, initZoom, initTranslate, translate, zoom

  if state == GLUT_DOWN:

    currentButton = button
    initX = x
    initY = y
    initZoom = zoom
    initTranslate = translate

  elif state == GLUT_UP:

    currentButton = None

    if button == GLUT_LEFT_BUTTON and initX == x and initY == y: # Process a left click (with no dragging)

      # Find which image the click is in

      toDraw, rows, cols, maxHeight, maxWidth, scale, horizSpacing, vertSpacing = getImagesInfo()

      row = (y-vertSpacing ) / float(maxHeight+vertSpacing)
      col = (x-horizSpacing) / float(maxWidth+horizSpacing)

      if row < 0 or row-math.floor(row) > maxHeight/(maxHeight+vertSpacing):
        return

      if col < 0 or col-math.floor(col) > maxWidth/(maxWidth+horizSpacing):
        return

      # Get the image

      image = toDraw[ int(row) ][ int(col) ]

      if image is None:
        return

      # Get bounds of visible image
      #
      # Bounds are [cx-offset,cx+offset] x [cy-offset,cy+offset]
      
      height = scale * image.shape[0]
      width  = scale * image.shape[1]

      cx     = 0.5 - translate[0]/width
      cy     = 0.5 - translate[1]/height
      offset = 0.5 / zoom

      # Find pixel position within the image array

      xFraction = (col-math.floor(col)) / (maxWidth /float(maxWidth +horizSpacing))
      yFraction = (row-math.floor(row)) / (maxHeight/float(maxHeight+vertSpacing ))

      pixelX = int( image.shape[1] * ((1-xFraction)*(cx-offset) + xFraction*(cx+offset)) )
      pixelY = int( image.shape[0] * ((1-yFraction)*(cy+offset) + yFraction*(cy-offset)) )
      
      # for the FT images, move the position half up and half right,
      # since the image is displayed with that shift, while the FT array
      # stores the unshifted values.

      isFT = (int(row) == 1)

      if isFT:

        pixelX = pixelX - image.shape[1]/2
        if pixelX < 0:
          pixelX = pixelX + image.shape[1]

        pixelY = pixelY - image.shape[0]/2
        if pixelY < 0:
          pixelY = pixelY + image.shape[0]

      # Perform the operation
      #
      # No operation is implemented here, but could be (e.g. image modification at the mouse position)

      # applyOperation( image, pixelX, pixelY, isFT )  

      print 'click at', pixelX, pixelY, '=', image[pixelY][pixelX], np.absolute(image[pixelY][pixelX])

      # Done

      glutPostRedisplay()



# Handle mouse dragging
#
# Zoom out/in with right button dragging up/down.
# Translate with left button dragging.


def mouseMotion( x, y ):

  global zoom, translate

  if currentButton == GLUT_RIGHT_BUTTON:

    # zoom

    factor = 1 # controls the zoom rate
    
    if y > initY: # zoom in
      zoom = initZoom * (1 + factor*(y-initY)/float(windowHeight))
    else: # zoom out
      zoom = initZoom / (1 + factor*(initY-y)/float(windowHeight))

  elif currentButton == GLUT_LEFT_BUTTON:

    # translate

    translate = ( initTranslate[0] + (x-initX)/zoom, initTranslate[1] + (initY-y)/zoom )

  glutPostRedisplay()


# For an image coordinate, if it's < 0 or >= max, wrap the coorindate
# around so that it's in the range [0,max-1].  This is useful dealing
# with FT images.

def wrap( val, max ):

  if val < 0:
    return val+max
  elif val >= max:
    return val-max
  else:
    return val



# Load initial data
#
# The command line (stored in sys.argv) could have:
#
#     main.py {image filename}

if len(sys.argv) > 1:
  imageFilename = sys.argv[1]
  imagePath = os.path.join( imageDir,  imageFilename  )

image  = loadImage(  imagePath  )


# If commands exist on the command line (i.e. there are more than two
# arguments), process each command, then exit.  Otherwise, go into
# interactive mode.

if len(sys.argv) > 2:

  outputMagnitudes = True

  # process commands

  cmds = sys.argv[2:]

  while len(cmds) > 0:
    cmd = cmds.pop(0)
    if cmd == 'f':
      forwardFT_all()
    elif cmd == 'i':
      inverseFT_all()
    elif cmd == 'm':
      outputMagnitudes = True
    elif cmd == 'p':
      outputMagnitudes = Falsea
    elif cmd == 'c':
      image, lines = compute()
      print lines
    elif cmd[0] == 'o': # image name follows in 'cmds'
      filename = cmds.pop(0)
      outputImage( resultImage, filename, False, False, True )
    else:
      print """command '%s' not understood.
command-line arguments:
  c - compute  
  f - apply forward FT
  i - apply inverse FT
  o - output the image
  m - for output, use magnitudes (default)
  p - for output, use phases""" % cmd

else:
      
  # Run OpenGL

  glutInit()
  glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGB )
  glutInitWindowSize( windowWidth, windowHeight )
  glutInitWindowPosition( 50, 50 )

  glutCreateWindow( 'imaging' )

  glutDisplayFunc( display )
  glutKeyboardFunc( keyboard )
  glutSpecialFunc( special )
  glutReshapeFunc( reshape )
  glutMouseFunc( mouse )
  glutMotionFunc( mouseMotion )

  glDisable( GL_DEPTH_TEST )

  glutMainLoop()
