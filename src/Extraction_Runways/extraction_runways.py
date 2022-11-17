""" Extraction of Runways Script"""

import math
from ast import literal_eval
import time
import cv2
import pandas as pd
import numpy as np

import logging
logger = logging.getLogger('main_logger')


class Extractrunways:
    """Create a class to extract "runways from cropped airports."""

    def __init__(self, conf):
        self.conf_path = conf["paths"]

    def get_labeled_runways(self):
        """Read and prepare the extracted output from Label Studio.
        Args:
            config_path: the input paths set in the config.
        Returns:
           df_runways: A dataframe with coordinates of runways in each row.
        """
        path = self.conf_path["runway_input_path"] + self.conf_path["runway_input_file"]
        df_runways = pd.read_csv(path)
        df_runways["image"] = df_runways["image"].apply(lambda x: "cropped" + x.split("cropped")[1])
        df_runways['label'] = df_runways['label'].apply(literal_eval)
        df_runways = df_runways.explode('label').reset_index(drop=True)
        return df_runways

    def calculate_coordinates(self, label):
        """Get coordinates from the x, y, width, height and rotation.
        Args:
            label: dict containing the x, y, width, height and rotation info.
        Returns:
           cnt: array of rectangular coordinates of the runway.
        """
        pixel_x = label["x"] / 100.0 * label["original_width"]
        pixel_y = label["y"] / 100.0 * label["original_height"]
        pixel_width = label["width"] / 100.0 * label["original_width"]
        pixel_height = label["height"] / 100.0 * label["original_height"]
        angle = label["rotation"]

        # Calculate added distances
        alpha = math.atan(pixel_height / pixel_width)
        # Label studio defines the angle towards the vertical axis
        beta = math.pi * (angle/ 180)

        radius = math.sqrt((pixel_width/2) ** 2 + (pixel_height/2) ** 2)

        # Label studio saves the position of top left corner after rotation
        x_0 = pixel_x - radius * (math.cos(math.pi - alpha - beta) - math.cos(math.pi - alpha)) + pixel_width / 2
        y_0 = pixel_y + radius * (math.sin(math.pi - alpha - beta) - math.sin(math.pi - alpha)) + pixel_height / 2

        theta_1 = alpha + beta
        theta_2 = math.pi - alpha + beta
        theta_3 = math.pi + alpha + beta
        theta_4 = 2 * math.pi - alpha + beta

        x_coord = [
            x_0 + radius * math.cos(theta_1),
            x_0 + radius * math.cos(theta_2),
            x_0 + radius * math.cos(theta_3),
            x_0 + radius * math.cos(theta_4),
        ]
        y_coord = [
            y_0 + radius * math.sin(theta_1),
            y_0 + radius * math.sin(theta_2),
            y_0 + radius * math.sin(theta_3),
            y_0 + radius * math.sin(theta_4),
        ]
        cnt = np.array([
                    [[int(x_coord[0]), int(y_coord[0])]],
                    [[int(x_coord[1]), int(y_coord[1])]],
                    [[int(x_coord[2]), int(y_coord[2])]],
                    [[int(x_coord[3]), int(y_coord[3])]],
                ])
        return cnt

    def crop_save_runway(self, rect, image_name, path_output, index):
        start = time.time()
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        time_1 = time.time()
        logger.info(f"Prepare boxPoints. This took {time_1 - start}")
        path = self.conf_path["Outputs_path"]
        path_in = path + self.conf_path["folder_extraction_airports"]
        img = cv2.imread(path_in + image_name)
        time_2 = time.time()
        logger.info(f"Read image. This took {time_2-time_1}")
        if img is None:
            logger.info(f"The image does not exist in the airports cropped: {image_name}")
        else:
            cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

            # get width and height of the detected rectangle
            width = int(rect[1][0])
            height = int(rect[1][1])

            src_pts = box.astype("float32")

            # coordinate of the points in box points after the rectangle has been
            # straightened
            dst_pts = np.array([[0, height-1],
                                    [0, 0],
                                    [width-1, 0],
                                    [width-1, height-1]], dtype="float32")
            # the perspective transformation matrix
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)

            # directly warp the rotated rectangle to get the straightened rectangle
            warped = cv2.warpPerspective(img, M, (width, height))
            time_3 = time.time()
            logger.info(f"Crop and rotate rectangle. This took {time_3-time_2}")
            path_im = image_name.split("cropped")[1]
            shape_warped = warped.shape
            # check if image is horizontal and change to vertical
            if shape_warped[1] > shape_warped[0]:
                warped = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imwrite(path_output + "runway_" + str(index) + path_im, warped)
            time_4 = time.time()
            logger.info(f"Save Runway. This took {time_4-time_3}")
        return None


    def detect_save_runway(self):
        """Extract and save runways.
        Args:
            None.
        Returns:
           Saves runways.
        """
        df_runways = self.get_labeled_runways()
        for index in df_runways.index:
            label = df_runways.loc[index,"label"]
            image_name = df_runways.loc[index,"image"]
            logger.info(f"Start cropping runway process for image: {image_name}")
            cnt = self.calculate_coordinates(label)
            rect = cv2.minAreaRect(cnt)
            path = self.conf_path["Outputs_path"]
            path_out = path + self.conf_path["folder_extraction_runways"]
            self.crop_save_runway(rect, image_name, path_out, index)
        return None