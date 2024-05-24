import mrcnn
import mrcnn.config
import mrcnn.visualize
from mrcnn import model
import os
import numpy
import cv2
from PIL import Image, ImageDraw

CLASS_NAMES = ['BG', 'figure','formula']

class SimpleConfig(mrcnn.config.Config):
    NAME = "coco_inference"
    
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    NUM_CLASSES = len(CLASS_NAMES)

def load_model():
    model1 = model.MaskRCNN(mode="inference",
                                    config=SimpleConfig(),
                                    model_dir=os.getcwd())
    model1.load_weights(filepath="mask_custom.h5",
                       by_name=True)
    return model1


def merge_boxes(results_rois,results_masks):
    #line = len(results_rois)
    boxes = list()
    for box in results_rois:
        ymin = box[0]
        xmin = box[1]
        ymax = box[2]
        xmax = box[3]

        coors = [[xmin, ymin],[xmax,ymin], [xmax, ymax],[xmin,ymax]]

        boxes.append(coors)

    size = list(results_masks.shape[:2])
    size.append(3)

    stencil1 = numpy.zeros(size).astype(numpy.dtype("uint8"))
    stencil2= numpy.zeros(size).astype(numpy.dtype("uint8"))

    color = [255, 255, 255]

    for i in range(len(boxes)):
        stencil1 = numpy.zeros(size).astype(numpy.dtype("uint8"))

        contours = [numpy.array(boxes[i])]
        cv2.fillPoly(stencil1, contours, color)


        for j in range(i+1,len(boxes)):
            stencil2= numpy.zeros(size).astype(numpy.dtype("uint8"))
            contours = [numpy.array(boxes[j])]
            cv2.fillPoly(stencil2, contours, color)


            intersection = numpy.sum(numpy.logical_and(stencil1, stencil2))
        
            if intersection > 0:
                xmin = min(boxes[i][0][0],boxes[j][0][0])
                ymin = min(boxes[i][0][1],boxes[j][0][1])
                xmax = max(boxes[i][2][0],boxes[j][2][0])
                ymax = max(boxes[i][2][1],boxes[j][2][1])

                '''
                coors = [[xmin, ymin],[xmax,ymin], [xmax, ymax],[xmin,ymax]]
                '''
                #print(" {},{} INTERSECTION : {}".format(i,j,np.sum(intersection)))

                results_rois[i] = [ymin,xmin,ymax,xmax]
                arr = numpy.delete(results_rois,j,0)
                
                return merge_boxes(arr,results_masks)

    return results_rois

def extract_Figures(model,pil_image):
    
    image=numpy.array(pil_image)
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    r = model.detect([image], verbose=0)
    r = r[0]
    merged = merge_boxes(r['rois'],r['masks'])
    # print(merged)
    extract_imgs = list()
    bbox = []
    for i in merged:

        #  cropped_img = img[y: y + h, x: x + w]
        cropped_img = image[i[0]:i[2], i[1]: i[3]]
        extract_imgs.append(Image.fromarray(cropped_img))
    image = Image.fromarray(image)
    for i in merged:
        bbox.append((i[1], i[0], i[3], i[2]))
        shape = [(i[1], i[0]), (i[3],i[2])]
        
        img1 = ImageDraw.Draw(image)
        img1.rectangle(shape, fill ="#FFFFFF")
    
    return extract_imgs , image, bbox

