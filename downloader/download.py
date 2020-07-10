import urllib.request
import os
import argparse
import errno
import pandas as pd
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from tqdm import tqdm
from multiprocessing.pool import ThreadPool

argparser = argparse.ArgumentParser(description='Download specific objects from Open-Images dataset')
argparser.add_argument('-a', '--annots', default="/home/malnakli/scratch/datasets/openimages/validation-annotations-bbox.csv",
                       help='path to annotations file (.csv)')
argparser.add_argument('-o', '--objects', nargs='+',
                       help='download images of these objects')
argparser.add_argument('-d', '--dir', default="/home/malnakli/scratch/datasets/openimages/images/validation",
                       help='path to output directory')
argparser.add_argument('-l', '--labelmap', default="/home/malnakli/scratch/datasets/openimages/class-descriptions-boxable.csv",
                       help='path to labelmap (.csv)')
argparser.add_argument('-s3b', '--s3_bucket', default="open-images-dataset",
                       help="https://github.com/cvdfoundation/open-images-dataset#download-images-with-bounding-boxes-annotations")

argparser.add_argument('-s3o', '--s3_object', default="validation",
                       help="https://github.com/cvdfoundation/open-images-dataset#download-images-with-bounding-boxes-annotations")

args = argparser.parse_args()

# # parse arguments
ANNOTATIONS = args.annots
OUTPUT_DIR = args.dir
OBJECTS = args.objects
LABELMAP = args.labelmap

s3 = boto3.client('s3',config=Config(signature_version=UNSIGNED))
s3_BUCKET_NAME = args.s3_bucket
s3_OBJECT_NAME = args.s3_object

# make OUTPUT_DIR if not present
if not os.path.isdir(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print("\nCreated {} directory\n".format(OUTPUT_DIR))

# check if input files are valid, raise FileNotFoundError if not found
if not os.path.exists(ANNOTATIONS):
    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), ANNOTATIONS)
elif not os.path.exists(LABELMAP):
    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), LABELMAP)


def get_ooi_labelmap(labelmap):
    '''
    Given labelmap of all objects in Open Images dataset, get labelmap of objects of interest

    :param labelmap: dataframe containing object labels with respective label codes
    :return: dictionary containing object labels and codes of
                          user-inputted objects
    '''

    object_codes = {}
    for idx, row in labelmap.iterrows():
        if any(obj.lower() == row[1].lower() for obj in OBJECTS):
            object_codes[row[1].lower()] = row[0]

    return object_codes


def generate_download_list(annotations, labelmap):
    '''
    Parse through input annotations dataframe, find ImageID's of objects of interest,
    and get download urls for the corresponding images

    :param annotations: annotations dataframe
    :param labelmap: dictionary of object labels and codes
    :param base_url: basename of url
    :return: list of urls to download
    '''
    # create an empty dataframe
    df_download = pd.DataFrame(columns=['ImageID', 'LabelName'])

    # append dataframes to empty df according to conditions
    for key, value in labelmap.items():
        # find ImageID's in original annots dataframe corresponding to ooi's codes
        df_download = df_download.append(annotations.loc[annotations['LabelName'] == value, ['ImageID', 'LabelName']])

    ######################
    df_download.drop_duplicates()
    url_download_list = []

    for idx, row in df_download.iterrows():
        # get name of the image
        image_name = row['ImageID'] + ".jpg"

        # check if the image exists in directory
        if not os.path.exists(os.path.join(OUTPUT_DIR, image_name)):
            url_download_list.append(image_name)

    return url_download_list


def download_objects_of_interest(download_list):        

    for image in tqdm(download_list, desc="Download %: "):
        try:
            with open(f"{OUTPUT_DIR}/{image}", "wb") as dest_file:
                s3.download_fileobj(s3_BUCKET_NAME, f"{s3_OBJECT_NAME}/{image}", dest_file)
        
        except Exception as e:
            print("error fetching {}: {}".format(image, e))
        


def main():
    # read labelmap
    df_oid_labelmap = pd.read_csv(LABELMAP)  # open images dataset (oid) labelmap
    ooi_labelmap = get_ooi_labelmap(df_oid_labelmap)  # objects of interest (ooi) labelmap

    # read annotations
    df_annotations = pd.read_csv(ANNOTATIONS)

    print("\nGenerating download list for the following objects: ", [k for k, v in ooi_labelmap.items()])

    # get url list to download
    download_list = generate_download_list(annotations=df_annotations,
                                           labelmap=ooi_labelmap)
    print("\n# Images to dwonload",len(download_list))
    # download objects of interest
    download_objects_of_interest(download_list[:2])
    #s3.download_file(s3_BUCKET_NAME, s3_OBJECT_NAME, download_list[0])
    print("\nFinished downloads.")


if __name__ == '__main__':
    main()
