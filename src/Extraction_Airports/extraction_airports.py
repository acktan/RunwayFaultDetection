""" Extraction of Airports Script"""

import geopandas
import pandas as pd
import numpy as np
import cv2
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from shapely.geometry import Polygon, mapping
import rasterio
from rasterio.windows import Window
import warnings

import logging
logger = logging.getLogger('main_logger')


class Extractionairports:
    """Create a class to extract airports from orthophotos."""

    def __init__(self, conf, img_shape):
        self.conf_path = conf["paths"]
        self.img_shape = img_shape

    def read_create_input(self):
        """Read and create input.
        Args:
            config_path: the input paths set in the config.
        Returns:
           df: A dataframe intersecting the BD_ORTHO and BD_CARTA data,
           with their intersection geometry.
           df_multi_photos: A dataframe intersecting the BD_ORTHO and BD_CARTA data,
           with their intersection geometry for airports on multiple photos.
           dup_aero: dataframe of airports on multiple photos.
        """
        logger.info('Reading input for extraction of airports...')
        path_ortho = self.conf_path["bdortho_geom_input_path"]
        path_aero = self.conf_path["bdcarta_input_path"]
        print(path_ortho)
        print(path_aero)
        bd_ortho_data = geopandas.read_file(path_ortho)
        bd_aero = geopandas.read_file(path_aero)
        bd_ortho_data["area"] = bd_ortho_data.area

        # Create column for geometry of orthophotos and aeroports to be used later
        bd_ortho_data['geometry_photos'] = bd_ortho_data['geometry']
        bd_aero['geometry_aero'] = bd_aero['geometry']

        # Join both databases bd
        df = bd_aero.sjoin(bd_ortho_data, how="inner", predicate='intersects')
        df.drop(['NATURE', 'DESSERTE', 'area'], axis=1, inplace=True)
        df = df.sort_values(by='ID').reset_index(drop=True)

        # Create geometry intersection
        df["geometry_aero"] = geopandas.GeoSeries(df["geometry_aero"])
        df['geometry_intersection'] = df['geometry_aero'].intersection(df['geometry_photos'])

        # remove lines with no TOPONYME
        dup_aero = df[df['TOPONYME'].duplicated()]['TOPONYME'].unique()
        df_multi_photos = df[df['TOPONYME'].isin(dup_aero)].reset_index(drop=True)
        df = df[~df['TOPONYME'].isin(dup_aero)].reset_index(drop=True)
        return df, df_multi_photos, dup_aero

    def normalize_val(self, val, min_val, max_val, img_shape):
        """Normalizing function converting geometry to picture scale.
        Args:
            val: geometry to normalize.
            min_val: lower boundary of the geometry.
            max_val: upper boundary of the geometry.
            img_shape: the shape of the image.
        Returns:
           The normalized value.
        """
        val = int(((val-min_val)/(max_val-min_val))*img_shape)

        return val

    def get_edges_photos(self, poly_boundaries):
        """Get the edges of a picture.
        Args:
            poly_boundaries: the geometry polygon of the photo.
        Returns:
           The boundaries of the photo in latitude and longitude.
        """
        logger.info('Getting edges for photos...')
        poly_mapped_bound = mapping(poly_boundaries)
        poly_coordinates_bound = poly_mapped_bound['coordinates'][0]
        poly_bound = pd.DataFrame([
            {'lat_bound': int(coords[1]),
             'lon_bound': int(coords[0])}
            for coords in poly_coordinates_bound])

        # Get the top, bottom, right and left limits of x and y for normaliz.
        min_lat = poly_bound['lat_bound'].min()
        max_lat = poly_bound['lat_bound'].max()
        min_lon = poly_bound['lon_bound'].min()
        max_lon = poly_bound['lon_bound'].max()
        return min_lat, max_lat, min_lon, max_lon

    def get_edges_intersection(self, poly_intersection, poly_boundaries):
        """Get the edges of the intersection between two geometries.
        Args:
            poly_intersections: the geometry polygon of the intersection.
            poly_boundaries: the geometry polygon of the photo.
            img_shape: the shape of the images.
        Returns:
           The boundaries of the intersection in latitude and longitude.
        """
        logger.info('Getting edges for intersection...')
        min_lat, max_lat, min_lon, max_lon = self.get_edges_photos(poly_boundaries)
        poly_mapped = mapping(poly_intersection)
        poly_coordinates = poly_mapped['coordinates'][0]

        # update coordinates of intersection from the map to the picture reference
        poly_int = pd.DataFrame([{'nor_lat_int':
                                  self.normalize_val(int(coords[1]), min_lat,
                                                max_lat, self.img_shape[0]),
                                'nor_lon_int':
                                  self.normalize_val(int(coords[0]), min_lon,
                                                     max_lon, self.img_shape[0])}
                                 for coords in poly_coordinates])

        poly_crop = [(self.normalize_val(int(coords[1]), min_lat,
                                         max_lat, self.img_shape[0]),
                      self.normalize_val(int(coords[0]), min_lon,
                                         max_lon, self.img_shape[0]))
                     for coords in poly_coordinates]

        max_1, max_2 = poly_int.max()['nor_lat_int'], poly_int.max()['nor_lon_int']
        min_1, min_2 = poly_int.min()['nor_lat_int'], poly_int.min()['nor_lon_int']
        return max_1, max_2, min_1, min_2, poly_crop

    def get_cropping_coordinates(self, poly_bound, poly_inter):
        """Get the edges of the intersection between two geometries.
        Args:
            poly_intersections: the geometry polygon of the intersection.
            poly_boundaries: the geometry polygon of the photo.
            img_shape: the shape of the images.
        Returns:
           The boundaries of the intersection in latitude and longitude.
        """
        logger.info('Getting cropping coordinates...')
        # Get the x and y values of the edges of the photos
        poly_boundaries = poly_bound

        # poly_intersection = x[1]
        poly_intersection = poly_inter

        if poly_intersection.geom_type == 'Polygon':

            max_1, max_2, min_1, min_2, poly_crop = self.get_edges_intersection(
                poly_intersection, poly_boundaries
            )

            f_min_1 = (self.img_shape[0]-max_1)
            f_max_1 = (self.img_shape[1]-min_1)

            # Update cropping coordinates
            up_poly_crop = [
                (self.normalize_val(coords[1], min_2,
                                    max_2, (max_2-min_2)),
                 self.normalize_val((self.img_shape[0]-coords[0]),
                                    f_min_1, f_max_1, (f_max_1-f_min_1)))
                for coords in poly_crop]

        else:
            up_poly_crop = []
            f_max_1 = []
            max_2 = []
            f_min_1 = []
            min_2 = []
            lst_multi = list(poly_intersection)
            for j in range(len(lst_multi)):
                poly_intersection = lst_multi[j]
                max_1_k, max_2_k, min_1_k, min_2_k, poly_crop = self.get_edges_intersection(
                    poly_intersection, poly_boundaries)

                f_min_1_k = (self.img_shape[0]-max_1_k)
                f_max_1_k = (self.img_shape[1]-min_1_k)

                # Update cropping coordinates
                up_poly_crop_int = [
                    (self.normalize_val(coords[1], min_2_k,
                                        max_2_k, (max_2_k-min_2_k)),
                     self.normalize_val((self.img_shape[0]-coords[0]),
                                        f_min_1_k, f_max_1_k, (f_max_1_k-f_min_1_k)))
                    for coords in poly_crop]
                up_poly_crop.append(up_poly_crop_int)
                f_max_1.append(f_max_1_k)
                max_2.append(max_2_k)
                f_min_1.append(f_min_1_k)
                min_2.append(min_2_k)

        return up_poly_crop, f_min_1, f_max_1, min_2, max_2

    def get_edges_photos_multi(self, poly_boundaries):
        """Get the edges of multiple photos containing one airort.
        Args:
            poly_boundaries: the geometry polygon of the photos.
        Returns:
           The boundaries of the photos.
        """
        logger.info('Getting edges for airports in multiple photos...')
        poly_mapped_bound = mapping(poly_boundaries)
        poly_coordinates_bound = poly_mapped_bound['coordinates']

        if poly_boundaries.geom_type == 'Polygon':
            poly_use = poly_coordinates_bound[0]
            min_lat = min([poly_use[c][1] for c in range(len(poly_use))])
            max_lat = max([poly_use[c][1] for c in range(len(poly_use))])
            min_lon = min([poly_use[c][0] for c in range(len(poly_use))])
            max_lon = max([poly_use[c][0] for c in range(len(poly_use))])
        else:
            min_lat_l = []
            max_lat_l = []
            min_lon_l = []
            max_lon_l = []
            for j in range(len(poly_coordinates_bound)):
                poly_use = poly_coordinates_bound[j][0]

                # Get the top, bottom, right and left limits of x and y for normaliz.
                min_lat_l.append(min([poly_use[c][1]
                                      for c in range(len(poly_use))]))
                max_lat_l.append(max([poly_use[c][1]
                                      for c in range(len(poly_use))]))
                min_lon_l.append(min([poly_use[c][0]
                                      for c in range(len(poly_use))]))
                max_lon_l.append(max([poly_use[c][0]
                                      for c in range(len(poly_use))]))

            min_lat = min(min_lat_l)
            max_lat = max(max_lat_l)
            min_lon = min(min_lon_l)
            max_lon = max(max_lon_l)
        return min_lat, max_lat, min_lon, max_lon

    def crop_save(self, image_adj, poly_crop_adj,
                  path, path_im, q='', multi=False):
        """Crop airports from images and save.
        Args:
            img_adj: image reduced in size.
            poly_crop_adj: cropping coordinates.
            path: path of output.
            path_im: image name to save.
        Returns:
           Save a cropped airport image.
        """
        print("GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG")
        pts = np.array(poly_crop_adj)
        # (1) Crop the bounding rect
        if not multi:
            rect = cv2.boundingRect(pts)
            x, y, w, h = rect
            image_adj = image_adj[y:y+h, x:x+w].copy()
        # (2) make mask
        pts = pts - pts.min(axis=0)
        mask = np.zeros(image_adj.shape[:2], np.uint8)
        cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
        # (4) add the white background
        bg = np.ones_like(image_adj, np.uint8)*255
        cv2.bitwise_not(bg, bg, mask=mask)
        bg = bg + cv2.bitwise_and(image_adj, image_adj, mask=mask)
        # save the result
        cv2.imwrite(path + '/cropped_cv2_' +
                    path_im[:-4] + '_' + q + '.jpg', bg)
        logger.info('Image cropped and saved :)')
        return None

    def extract_airports(self, df, path_input, path_output):
        """Loop to save cropped images of airports contained in one photo.
        Args:
            df: dataframe with geometries of photos and airports.
        Returns:
           Saves cropped images of airports.
        """
        for i in range(len(df)):
            logger.info('iteration '+str(i+1)+' out of '+str(len(df)))
            logger.info('aeroport '+df.iloc[i]['TOPONYME'])
            logger.info('Getting cropping coordinates...')
            up_poly_crop, f_min_1, f_max_1, min_2, max_2 = self.get_cropping_coordinates(
                df.iloc[i]["geometry_photos"],
                df.iloc[i]["geometry_intersection"])
            path_im = df.iloc[i]['NOM'][2:]

            logger.info('Cropping and saving...')
            if df.iloc[i]["geometry_intersection"].geom_type == 'Polygon':
                logger.info('Reading photo...')
                with rasterio.open(path_input+path_im) as src:
                    im_crop = src.read(window=Window(min_2, f_min_1,
                                                     (max_2-min_2),
                                                     (f_max_1-f_min_1)))
                    im_crop = np.swapaxes(im_crop, 0, 2)
                    im_crop = np.swapaxes(im_crop, 0, 1)
                logger.info('Cropping photo...')
                self.crop_save(image_adj=im_crop, poly_crop_adj=up_poly_crop,
                               path=path_output, path_im=path_im)
            else:
                logger.info('This is a MultiPolygon!')
                for k in range(len(up_poly_crop)):
                    logger.info('Reading photo...')
                    with rasterio.open(path_input+path_im) as src:
                        img_int = src.read(window=Window(min_2[k], f_min_1[k],
                                                         (max_2[k]-min_2[k]),
                                                         (f_max_1[k]-f_min_1[k])
                                                        )
                                          )
                        img_swap = np.swapaxes(img_int, 0, 2)
                        im_crop = np.swapaxes(img_swap, 0, 1)
                    logger.info('Cropping photo...')
                    self.crop_save(image_adj=im_crop,
                                   poly_crop_adj=up_poly_crop[k],
                                   path=path_output,
                                   path_im=path_im, q=str(k))
            logger.info('Cropped and saved!')
        return None

    def extract_airports_multi(self, df, dup_aero, path_input, path_output):
        """Loop to save cropped images of airports contained in multiple photos.
        Args:
            df: dataframe with geometries of multi photos merged and airports.
        Returns:
           Saves cropped images of airports.
        """
        print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        for i in range(len(dup_aero)):
            print("BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB")
            logger.info("Iteration " + str(i+1) + ' out of ' + str(len(dup_aero)))
            df_aero_i = df[df["TOPONYME"]==dup_aero[i]].reset_index(drop=True)
            min_mi_lat_l = []
            max_ma_lat_l = []
            min_mi_lon_l = []
            max_ma_lon_l = []
            for k in range(len(df_aero_i)):
                print("CCCCCCCCCCCCCCCCCCCCCCCCCCCCCC")
                mi_lon, ma_lon, mi_lat, ma_lat = self.get_edges_photos_multi(
                    df_aero_i.iloc[k]["geometry_intersection"])
                if isinstance(mi_lat, list):
                    min_mi_lat_l.append(min(mi_lat))
                    max_ma_lat_l.append(max(ma_lat))
                    min_mi_lon_l.append(min(mi_lon))
                    max_ma_lon_l.append(max(ma_lon))
                else:
                    min_mi_lat_l.append(int(mi_lat))
                    max_ma_lat_l.append(int(ma_lat))
                    min_mi_lon_l.append(int(mi_lon))
                    max_ma_lon_l.append(int(ma_lon))
            min_mi_lat = min(min_mi_lat_l)
            max_ma_lat = max(max_ma_lat_l)
            min_mi_lon = min(min_mi_lon_l)
            max_ma_lon = max(max_ma_lon_l)
            height = int((max_ma_lon - min_mi_lon)*25000/5000)
            width = int((max_ma_lat - min_mi_lat)*25000/5000)
            img_fin = np.zeros((height, width, 3))
            del mi_lon, ma_lon, mi_lat, ma_lat
            del min_mi_lon_l, max_ma_lon_l
            logger.info("Cropping and saving...")
            for k in range(len(df_aero_i)):
                print("DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD")
                logger.info("Iteration "+str(k+1)+" out of "+str(len(df_aero_i)))
                logger.info("aeroport "+df_aero_i.iloc[k]["TOPONYME"])
                logger.info("Getting cropping coordinates...")
                up_poly_crop, f_min_1, f_max_1, min_2, max_2 = self.get_cropping_coordinates(
                    df_aero_i.iloc[k]["geometry_photos"],
                    df_aero_i.iloc[k]["geometry_intersection"])
                path_im = df_aero_i.iloc[k]["NOM"][2:]
                logger.info("Iteration k " + str(k+1) + " out of " + str(len(df_aero_i)))
                if df_aero_i.iloc[k]["geometry_intersection"].geom_type == "Polygon":
                    logger.info("Reading photo...")
                    with rasterio.open(path_input+path_im) as src:
                        print("EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
                        im_crop = src.read(window=Window(min_2, f_min_1, (max_2-min_2), (f_max_1-f_min_1)))
                        im_crop = np.swapaxes(im_crop, 0, 2)
                        im_crop = np.swapaxes(im_crop, 0, 1)
                    logger.info("Cropping photo...")
                    poly_mapped_int_coords = mapping(df_aero_i.iloc[k]["geometry_photos"])
                    poly_mapped_int_coords = poly_mapped_int_coords["coordinates"][0]
                    lat_new_orig = (min(poly_mapped_int_coords)[0]+(min_2*5000/25000) - min_mi_lat)*25000/5000
                    lon_new_orig = (max_ma_lon - max(poly_mapped_int_coords)[1] + (f_min_1*5000/25000))*25000/5000
                    if k == 0:
                        poly_last = df_aero_i.iloc[k]["geometry_intersection"]
                    else:
                        poly_last = poly_last.union(df_aero_i.iloc[k]["geometry_intersection"])
                    img_fin[int(lon_new_orig):int((lon_new_orig+f_max_1-f_min_1)),
                            int(lat_new_orig):int(lat_new_orig+max_2-min_2),:] = im_crop[:,:,:]
                    # UNION OF POLYGONS
                else:
                    logger.info("This is a MultiPolygon!")
                    for q in range(len(up_poly_crop)):
                        print("IIIIIIIIIIIIIIIIIIIIIIIIIIIIIII")
                        logger.info("Iteration q "+str(q+1) + " out of " + str(len(up_poly_crop)))
                        logger.info("Reading photo...")
                        with rasterio.open(path_input+path_im) as src:
                            im_crop = src.read(window=Window(min_2[q], (f_min_1[q]), (max_2[q]-min_2[q]), (f_max_1[q]-f_min_1[q])))
                            im_crop = np.swapaxes(im_crop, 0, 2)
                            im_crop = np.swapaxes(im_crop, 0, 1)
                        logger.info("Cropping photo...")
                        poly_mapped_int_coords = mapping(df_aero_i.iloc[k]["geometry_photos"])
                        poly_mapped_int_coords = poly_mapped_int_coords["coordinates"][0]
                        lat_new_orig = (min(poly_mapped_int_coords)[0]+(min_2[q]*5000/25000) - min_mi_lat)*25000/5000
                        lon_new_orig = (max_ma_lon - max(poly_mapped_int_coords)[1] + (f_min_1[q]*5000/25000))*25000/5000
                        if ((k == 0) and (q == 0)):
                            poly_last = df_aero_i.iloc[k]["geometry_intersection"]
                        else:
                            poly_last = poly_last.union(df_aero_i.iloc[k]["geometry_intersection"])
                        print("OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
                        for llo in range(2):
                            for lla in range(2):
                                img_fin[int(lon_new_orig+llo*((f_max_1[q]-f_min_1[q])//2)):int(lon_new_orig+(llo+1)*((f_max_1[q]-f_min_1[q])//2)),
                                        int(lat_new_orig+lla*((max_2[q]-min_2[q])//2)):int(lat_new_orig+(lla+1)*((max_2[q]-min_2[q])//2)),:] = im_crop[llo*((f_max_1[q]-f_min_1[q])//2):(llo+1)*((f_max_1[q]-f_min_1[q])//2),lla*((max_2[q]-min_2[q])//2):(lla+1)*((max_2[q]-min_2[q])//2),:]
            print("FFFFGGGGGGGGGGGGGGGGGG")
            del up_poly_crop
            del f_min_1, f_max_1, min_2, max_2, lat_new_orig
            del lon_new_orig, poly_mapped_int_coords
            del im_crop
            map_last = mapping(poly_last)["coordinates"]
            for p in range(len(map_last)):
                print("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF")
                if p == 0:
                    pop_lst = [(pp[0], min_mi_lat, height, pp[1], max_ma_lon, width) for pp in map_last[p]]
                    new_lst_1 = ([
                        (int(((pp[0] - min_mi_lat)*25000/5000)),
                         int((max_ma_lon - pp[1])*25000/5000))
                        for pp in map_last[p]])
                    last_polygon_multi = [*new_lst_1]
                else:
                    new_lst_2 = ([
                        (int(((pp[0] - min_mi_lat)*25000/5000)),
                         int((max_ma_lon - pp[1])*25000/5000))
                        for pp in map_last[p]])
                    last_polygon_multi = last_polygon_multi + new_lst_2
                    pop_lst = pop_lst + ([(
                        int(((pp[0] - min_mi_lat)*25000/5000)),
                        int((max_ma_lon - pp[1])*25000/5000))
                        for pp in map_last[p]])
            self.crop_save(image_adj=img_fin,
                           poly_crop_adj=last_polygon_multi,
                           path=path_output,
                           path_im=path_im, multi=True)
            logger.info("Cropped and saved!")
        return None
    
    def extract_all_airports(self):
        """Loop to save all cropped images of airports.
        Args:
            None
        Returns:
           Saves all cropped images of airports.
        """
        df, df_multi_photos, dup_aero = self.read_create_input()
        print(len(df), len(df_multi_photos), len(dup_aero))
        path_output = self.conf_path["Outputs_path"] + self.conf_path["folder_extraction_airports"]
        path_input = self.conf_path["bdortho_input_path"]
        self.extract_airports(df, path_input, path_output)
        self.extract_airports_multi(df_multi_photos, dup_aero, path_input, path_output)
        return None