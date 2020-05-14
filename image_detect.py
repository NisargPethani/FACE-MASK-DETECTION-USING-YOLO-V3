from __future__ import division

from models import Darknet
from utils.utils import load_classes,non_max_suppression_output, non_max_suppression

import argparse

import os
import torch
import numpy as np
from torch.autograd import Variable

import cv2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file_path", type=str, default="testing/input/images", help="path to images directory")
    parser.add_argument("--output_path", type=str, default="testing/output/images", help="output image directory")
    parser.add_argument("--model_def", type=str, default="config/yolov3_mask.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="checkpoints/yolov3_ckpt_35.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/mask_dataset.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.3, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--frame_size", type=int, default=416, help="size of each image dimension")

    opt = parser.parse_args()
    print(opt)

    # Output directory
    os.makedirs(opt.output_path, exist_ok=True)

    # checking for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.frame_size).to(device)

    # loading weights
    if opt.weights_path.endswith(".weights"):
        model.load_darknet_weights(opt.weights_path)  # Load weights
    else:
        model.load_state_dict(torch.load(opt.weights_path))  # Load checkpoints

    # Set in evaluation mode
    model.eval()

    # Extracts class labels from file
    classes = load_classes(opt.class_path)

    # ckecking for GPU for Tensor
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    print("\nPerforming object detection:")

    # for text in output
    t_size = cv2.getTextSize(" ", cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]

    for imagename in os.listdir(opt.input_file_path):

        print("\n"+imagename+"_______")
        image_path = os.path.join(opt.input_file_path, imagename)

        # frame extraction
        org_img = cv2.imread(image_path)

        # Original image width and height
        i_height, i_width = org_img.shape[:2]

        # resizing => [BGR -> RGB] => [[0...255] -> [0...1]] => [[3, 416, 416] -> [416, 416, 3]]
        #                       => [[416, 416, 3] => [416, 416, 3, 1]] => [np_array -> tensor] => [tensor -> variable]

        # resizing to [416 x 416]

        # Create a black image
        x = y = i_height if i_height > i_width else i_width

        # Black image
        img = np.zeros((x, y, 3), np.uint8)

        # Putting original image into black image
        start_new_i_height = int((y - i_height) / 2)
        start_new_i_width = int((x - i_width) / 2)

        img[start_new_i_height: (start_new_i_height + i_height) ,start_new_i_width: (start_new_i_width + i_width) ] = org_img

        #resizing to [416x 416]
        img = cv2.resize(img, (opt.frame_size, opt.frame_size))

        # [BGR -> RGB]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # [[0...255] -> [0...1]]
        img = np.asarray(img) / 255
        # [[3, 416, 416] -> [416, 416, 3]]
        img = np.transpose(img, [2, 0, 1])
        # [[416, 416, 3] => [416, 416, 3, 1]]
        img = np.expand_dims(img, axis=0)
        # [np_array -> tensor]
        img = torch.Tensor(img)

        # plt.imshow(img[0].permute(1, 2, 0))
        # plt.show()

        # [tensor -> variable]
        img = Variable(img.type(Tensor))

        # Get detections
        with torch.no_grad():
            detections = model(img)

        detections = non_max_suppression_output(detections, opt.conf_thres, opt.nms_thres)

        # print(detections)

        # For accommodate results in original frame
        mul_constant = x / opt.frame_size

        # For each detection in detections
        for detection in detections:
            if detection is not None:

                print("{0} Detection found".format(len(detection)))
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detection:

                    # Accommodate bounding box in original frame
                    x1 = int(x1 * mul_constant - start_new_i_width)
                    y1 = int(y1 * mul_constant - start_new_i_height)
                    x2 = int(x2 * mul_constant - start_new_i_width)
                    y2 = int(y2 * mul_constant - start_new_i_height)

                    # Bounding box making and setting Bounding box title
                    if (int(cls_pred) == 0):
                        # WITH_MASK
                        cv2.rectangle(org_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    else:
                        #WITHOUT_MASK
                        cv2.rectangle(org_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

                    cv2.putText(org_img, classes[int(cls_pred)]+": %.2f" %conf, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1,
                                [225, 255, 255], 2)



        out_filepath = os.path.join(opt.output_path, imagename)
        cv2.imwrite(out_filepath,org_img)

        print("Done....")

    cv2.destroyAllWindows()
