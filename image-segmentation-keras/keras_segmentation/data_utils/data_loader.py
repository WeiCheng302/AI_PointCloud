import itertools
import os
import random
import six
import numpy as np
import cv2
import tifffile
from multiprocessing import cpu_count, Pool
import threading

try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence

try:
    from tqdm import tqdm
except ImportError:
    print("tqdm not found, disabling progress bars")

    def tqdm(iter):
        return iter


from ..models.config import IMAGE_ORDERING
from ..data_utils.augmentation import augment_seg, custom_augment_seg

DATA_LOADER_SEED = 0

random.seed(DATA_LOADER_SEED)
class_colors = [(random.randint(0, 255), random.randint(
    0, 255), random.randint(0, 255)) for _ in range(5000)]


ACCEPTABLE_IMAGE_FORMATS = [".jpg", ".jpeg", ".png", ".bmp", ".tif"]
ACCEPTABLE_SEGMENTATION_FORMATS = [".png", ".bmp", ".tif"]


class DataLoaderError(Exception):
    pass

def get_image_list_from_path(images_path ):
    image_files = []
    for dir_entry in os.listdir(images_path):
            if os.path.isfile(os.path.join(images_path, dir_entry)) and \
                    os.path.splitext(dir_entry)[1] in ACCEPTABLE_IMAGE_FORMATS:
                file_name, file_extension = os.path.splitext(dir_entry)
                image_files.append(os.path.join(images_path, dir_entry))
    return image_files


def get_pairs_from_paths(images_path, segs_path, ignore_non_matching=False, other_inputs_paths=None):
    """ Find all the images from the images_path directory and
        the segmentation images from the segs_path directory
        while checking integrity of data """

    image_files = []
    segmentation_files = {}
    for dir_entry in os.listdir(images_path):
        if os.path.isfile(os.path.join(images_path, dir_entry)) and \
                os.path.splitext(dir_entry)[1] in ACCEPTABLE_IMAGE_FORMATS:
            file_name, file_extension = os.path.splitext(dir_entry)
            image_files.append((file_name, file_extension,
                                os.path.join(images_path, dir_entry)))

    if other_inputs_paths is not None:
        other_inputs_files = []

        for i, other_inputs_path in enumerate(other_inputs_paths):
            temp = []

            for y, dir_entry in enumerate(os.listdir(other_inputs_path)):
                if os.path.isfile(os.path.join(other_inputs_path, dir_entry)) and \
                        os.path.splitext(dir_entry)[1] in ACCEPTABLE_IMAGE_FORMATS:
                    file_name, file_extension = os.path.splitext(dir_entry)

                    temp.append((file_name, file_extension,
                                 os.path.join(other_inputs_path, dir_entry)))

            other_inputs_files.append(temp)

    for dir_entry in os.listdir(segs_path):
        if os.path.isfile(os.path.join(segs_path, dir_entry)) and \
           os.path.splitext(dir_entry)[1] in ACCEPTABLE_SEGMENTATION_FORMATS:
            file_name, file_extension = os.path.splitext(dir_entry)
            full_dir_entry = os.path.join(segs_path, dir_entry)
            if file_name in segmentation_files:
                raise DataLoaderError("Segmentation file with filename {0}"
                                      " already exists and is ambiguous to"
                                      " resolve with path {1}."
                                      " Please remove or rename the latter."
                                      .format(file_name, full_dir_entry))

            segmentation_files[file_name] = (file_extension, full_dir_entry)

    return_value = []
    # Match the images and segmentations
    for image_file, _, image_full_path in image_files:
        if image_file in segmentation_files:
            if other_inputs_paths is not None:
                other_inputs = []
                for file_paths in other_inputs_files:
                    success = False

                    for (other_file, _, other_full_path) in file_paths:
                        if image_file == other_file:
                            other_inputs.append(other_full_path)
                            success = True
                            break

                    if not success:
                        raise ValueError("There was no matching other input to", image_file, "in directory")

                return_value.append((image_full_path,
                                     segmentation_files[image_file][1], other_inputs))
            else:
                return_value.append((image_full_path,
                                     segmentation_files[image_file][1]))
        elif ignore_non_matching:
            continue
        else:
            # Error out
            raise DataLoaderError("No corresponding segmentation "
                                  "found for image {0}."
                                  .format(image_full_path))

    return return_value


def get_image_array(image_input,
                    width, height,
                    imgNorm="None", ordering='channels_first', read_image_type=1):
    """ Load image array from input """
    if type(image_input) is np.ndarray:
        # It is already an array, use it as it is
        img = image_input
    elif isinstance(image_input, six.string_types):
        if not os.path.isfile(image_input):
            raise DataLoaderError("get_image_array: path {0} doesn't exist"
                                  .format(image_input))
        if os.path.splitext(image_input)[1] is ".tif":
            img = tifffile.imread(image_input)
        else:
            img = cv2.imread(image_input, read_image_type)
            #print("IMG")
    else:
        raise DataLoaderError("get_image_array: Can't process input type {0}"
                              .format(str(type(image_input))))

    if imgNorm == "sub_and_divide":
        img = np.float32(cv2.resize(img, (width, height))) / 127.5 - 1
    elif imgNorm == "sub_mean":
        img = cv2.resize(img, (width, height))
        img = img.astype(np.float32)
        img = np.atleast_3d(img)

        means = [103.939, 116.779, 123.68]

        for i in range(min(img.shape[2], len(means))):
            img[:, :, i] -= means[i]

        img = img[:, :, ::-1]
    elif imgNorm == "divide":
        img = cv2.resize(img, (width, height))
        img = img.astype(np.float32)
        img = img/255.0
    elif imgNorm == "None":
        img = cv2.resize(img, (width, height))

    if ordering == 'channels_first':
        img = np.rollaxis(img, 2, 0)
    return img

def get_segmentation_array_float(image_input, nClasses,
                           width, height, no_reshape=False, read_image_type=1):

    seg_labels = np.zeros((height, width, nClasses))
    img = image_input
    class_list = np.array([0, 0.5, 1])
    
    for c in range(nClasses):
        seg_labels[:, :, 0] = class_list[c]
        # seg_labels[:, :, c] = (img == class_list[c]).astype(np.float32)

    if not no_reshape:
        seg_labels = np.reshape(seg_labels, (width*height, nClasses))

    return seg_labels


def get_segmentation_array(image_input, nClasses,
                           width, height, no_reshape=False, read_image_type=1):
    """ Load segmentation array from input """

    seg_labels = np.zeros((height, width, nClasses))

    if type(image_input) is np.ndarray:
        # It is already an array, use it as it is
        img = image_input
    elif isinstance(image_input, six.string_types):
        if not os.path.isfile(image_input):
            raise DataLoaderError("get_segmentation_array: "
                                  "path {0} doesn't exist".format(image_input))
        if os.path.splitext(image_input)[1] is ".tif":
            img = tifffile.imread(image_input)
        else:
            img = cv2.imread(image_input, read_image_type)
    else:
        raise DataLoaderError("get_segmentation_array: "
                              "Can't process input type {0}"
                              .format(str(type(image_input))))

    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)
    ##########
    # Fill in disappeared during augmentation ?
    # Consider to use the Upper functions
    ##########
    img = img[:, :, 0] 
    #class_list = np.array([0, 0.5, 1])

    for c in range(nClasses):
        seg_labels[:, :, c] = (img == c).astype(int)

    if not no_reshape:
        seg_labels = np.reshape(seg_labels, (width*height, nClasses))

    return seg_labels


def verify_segmentation_dataset(images_path, segs_path,
                                n_classes, show_all_errors=False):
    try:
        img_seg_pairs = get_pairs_from_paths(images_path, segs_path)
        if not len(img_seg_pairs):
            print("Couldn't load any data from images_path: "
                  "{0} and segmentations path: {1}"
                  .format(images_path, segs_path))
            return False

        return_value = True
        for im_fn, seg_fn in tqdm(img_seg_pairs):
            if os.path.splitext(im_fn)[1] == ".tif":
                img = tifffile.imread(im_fn)
            else:
                img = cv2.imread(im_fn)
            seg = cv2.imread(seg_fn)
            # Check dimensions match
            if not img.shape == seg.shape:
                return_value = False
                print("The size of image {0} and its segmentation {1} "
                      "doesn't match (possibly the files are corrupt)."
                      .format(im_fn, seg_fn))
                if not show_all_errors:
                    break
            else:
                max_pixel_value = np.max(seg[:, :, 0])
                if max_pixel_value >= n_classes:
                    return_value = False
                    print("The pixel values of the segmentation image {0} "
                          "violating range [0, {1}]. "
                          "Found maximum pixel value {2}"
                          .format(seg_fn, str(n_classes - 1), max_pixel_value))
                    if not show_all_errors:
                        break
        if return_value:
            print("Dataset verified! ")
        else:
            print("Dataset not verified!")
        return return_value
    except DataLoaderError as e:
        print("Found error during data loading\n{0}".format(str(e)))
        return False

def image_segmentation_generator(images_path, segs_path, batch_size,
                                 n_classes, input_height, input_width,
                                 output_height, output_width,
                                 do_augment=False,
                                 augmentation_name="aug_all",
                                 custom_augmentation=None,
                                 other_inputs_paths=None, preprocessing=None,
                                 read_image_type=cv2.IMREAD_COLOR , ignore_segs=False ):
    

    if not ignore_segs:
        img_seg_pairs = get_pairs_from_paths(images_path, segs_path, other_inputs_paths=other_inputs_paths)
        random.shuffle(img_seg_pairs)
        zipped = itertools.cycle(img_seg_pairs)
    else:
        img_list = get_image_list_from_path( images_path )
        random.shuffle( img_list )
        img_list_gen = itertools.cycle( img_list )


    while True:
        X = []
        Y = []
        for _ in range(batch_size):
            if other_inputs_paths is None:

                if ignore_segs:
                    im = next( img_list_gen )
                    seg = None 
                else:
                    im, seg = next(zipped)
                    seg = cv2.imread(seg, 1)

                if os.path.splitext(im)[1] == ".tif":
                    im = tifffile.imread(im)
                    #print("TIF")
                else:
                    im = cv2.imread(im, read_image_type)

                if do_augment:

                    assert ignore_segs == False , "Not supported yet"

                    if custom_augmentation is None:
                        im, seg[:, :, 0] = augment_seg(im, seg[:, :, 0],
                                                       augmentation_name)
                    else:
                        im, seg[:, :, 0] = custom_augment_seg(im, seg[:, :, 0],
                                                              custom_augmentation)

                if preprocessing is not None:
                    im = preprocessing(im)

                X.append(get_image_array(im, input_width,
                                         input_height, ordering=IMAGE_ORDERING))
            else:

                assert ignore_segs == False , "Not supported yet"

                im, seg, others = next(zipped)

                #im = cv2.imread(im, read_image_type)
                if os.path.splitext(im)[1] is ".tif":
                    im = tifffile.imread(im)
                else:
                    im = cv2.imread(im, read_image_type)
                seg = cv2.imread(seg, 1)

                oth = []
                for f in others:
                    oth.append(cv2.imread(f, read_image_type))

                if do_augment:
                    if custom_augmentation is None:
                        ims, seg[:, :, 0] = augment_seg(im, seg[:, :, 0],
                                                        augmentation_name, other_imgs=oth)
                    else:
                        ims, seg[:, :, 0] = custom_augment_seg(im, seg[:, :, 0],
                                                               custom_augmentation, other_imgs=oth)
                else:
                    ims = [im]
                    ims.extend(oth)

                oth = []
                for i, image in enumerate(ims):
                    oth_im = get_image_array(image, input_width,
                                             input_height, ordering=IMAGE_ORDERING)

                    if preprocessing is not None:
                        if isinstance(preprocessing, Sequence):
                            oth_im = preprocessing[i](oth_im)
                        else:
                            oth_im = preprocessing(oth_im)

                    oth.append(oth_im)

                X.append(oth)

            if not ignore_segs:
                Y.append(get_segmentation_array(
                    seg, n_classes, output_width, output_height))

        if ignore_segs:
            yield np.array(X)
        else:
            yield np.array(X), np.array(Y)

def read_all_imgs(images_path, segs_path, read_image_type=cv2.IMREAD_COLOR):

    img_seg_pairs = get_pairs_from_paths(images_path, segs_path)
    zipped = itertools.cycle(img_seg_pairs)

    img_list = []
    seg_list = []

    for i in range(len(img_seg_pairs)):

        im, seg = next(zipped)

        if os.path.splitext(seg)[1] == ".tif":
            seg = tifffile.imread(seg)
        else:
            seg = cv2.imread(seg, 1)

        if os.path.splitext(im)[1] == ".tif":
            im = tifffile.imread(im)
        else:
            im = cv2.imread(im, read_image_type)
        img_list.append(im)
        #img_list.append(im[:, :, 0:3])
        seg_list.append(seg[: , : , 0])

    return img_list, seg_list

def Clip_and_Aug(im, seg, X, Y, h, w, input_height, input_width, n_classes, output_height, output_width, do_augment, augmentation_name):

    imClip = im[h : h + input_height, w : w + input_width]
    segClip = seg[h : h + input_height, w : w + input_width]

    if do_augment: 
        imClip, segClip = augment_seg(imClip, segClip, augmentation_name)

    X.append(get_image_array(imClip, input_width,
                                     input_height, imgNorm="None", ordering=IMAGE_ORDERING))
    Y.append(get_segmentation_array_float(
                    segClip, n_classes, output_width, output_height))

    return X, Y

def random_img_seg_generator_pre_read_multi(img_list, seg_list, batch_size, 
                             n_classes, input_height, input_width,
                             output_height, output_width, do_augment = False, 
                             augmentation_name="aug_flip_and_rot", multi = False):

    img_index = np.arange(len(img_list))
    random.shuffle(img_index)
    zipped = itertools.cycle(img_index)

    while True:
        X = []
        Y = []

        idx = next(zipped)

        im = img_list[idx]
        seg = seg_list[idx]

        # Make sure the ratio between clipped img and large img

        threads = []
        for i in range(int(batch_size)):            

            h = random.randint(0, im.shape[0] - input_height)
            w = random.randint(0, im.shape[1] - input_width)

            threads.append(threading.Thread(target= Clip_and_Aug, 
                                            args = ((im, seg, X, Y, h, w, input_height, input_width, n_classes, output_height, output_width, do_augment, augmentation_name))))
            threads[i].start()

        for i in range(int(batch_size)):
            threads[i].join()            

        yield np.array(X), np.array(Y)

def random_img_seg_generator_pre_read(img_list, seg_list, batch_size, 
                             n_classes, input_height, input_width,
                             output_height, output_width, do_augment = False, 
                             augmentation_name="aug_flip_and_rot"):


    while True:
        X = []
        Y = []

        # Make sure the ratio between clipped img and large img
        for _ in range(batch_size):

            img_index = np.arange(len(img_list))
            random.shuffle(img_index)
            zipped = itertools.cycle(img_index)

            idx = next(zipped)

            im = img_list[idx]
            seg = seg_list[idx]

            h = random.randint(0, im.shape[0] - input_height)
            w = random.randint(0, im.shape[1] - input_width)

            imClip = im[h : h + input_height, w : w + input_width]
            segClip = seg[h : h + input_height, w : w + input_width]

            #seg_labels = np.zeros((input_height, input_width, 3))
            #for c in range(3):
            #    seg_labels[:, :, c] = (segClip == c).astype(int)

            #segClip = seg_labels
            segClip = segClip/2 # Read the Labels

            if do_augment: 
                #imClip, segClip[:, :, 0] = augment_seg(imClip, segClip[:, :, 0], augmentation_name)
                imClip, segClip = augment_seg(imClip, segClip, augmentation_name)

            X.append(get_image_array(imClip, input_width,
                                     input_height, imgNorm="None", ordering=IMAGE_ORDERING))  #[:, :, 0:3]
            
            #CE
            #segClip = np.reshape(segClip, (input_width*input_height, 3))
            #Y.append(segClip)

            #BCE
            Y.append(segClip.flatten()) #<===== Be Careful
            #Y.append(get_segmentation_array_float(
                    #segClip, n_classes, output_width, output_height))

        yield np.array(X), np.array(Y)

def random_img_seg_generator(images_path, segs_path, batch_size, 
                             n_classes, input_height, input_width,
                             output_height, output_width, do_augment = False, 
                             augmentation_name="aug_flip_and_rot", 
                             read_image_type=cv2.IMREAD_COLOR):

    img_seg_pairs = get_pairs_from_paths(images_path, segs_path)
    random.shuffle(img_seg_pairs)
    zipped = itertools.cycle(img_seg_pairs)

    while True:
        X = []
        Y = []

        im, seg = next(zipped)

        if os.path.splitext(seg)[1] == ".tif":
            seg = tifffile.imread(seg)
        else:
            seg = cv2.imread(seg, 1)

        if os.path.splitext(im)[1] == ".tif":
            im = tifffile.imread(im)
        else:
            im = cv2.imread(im, read_image_type)

        # Make sure the ratio between clipped img and large img
        count = -1
        for _ in range(batch_size):

            count += 1
            
            if count >= batch_size/2 :

                count = -1
                
                im, seg = next(zipped)                

                if os.path.splitext(seg)[1] == ".tif":
                    seg = tifffile.imread(seg)
                    #print(seg)
                else:
                    seg = cv2.imread(seg, 1)
                    #print(seg)

                if os.path.splitext(im)[1] == ".tif":
                    im = tifffile.imread(im)
                else:
                    im = cv2.imread(im, read_image_type)

            h = random.randint(0, im.shape[0] - input_height)
            w = random.randint(0, im.shape[1] - input_width)

            imClip = im[h : h + input_height, w : w + input_width]
            segClip = seg[h : h + input_height, w : w + input_width]
            #segClip = segClip/2

            if do_augment: 
                #imClip, segClip[:, :, 0] = augment_seg(imClip, segClip[:, :, 0], augmentation_name)
                imClip, segClip = augment_seg(imClip, segClip, augmentation_name)


                X.append(get_image_array(imClip, input_width,
                                         input_height, ordering=IMAGE_ORDERING))
                Y.append(get_segmentation_array_float(
                    segClip, n_classes, output_width, output_height))

        yield np.array(X), np.array(Y)