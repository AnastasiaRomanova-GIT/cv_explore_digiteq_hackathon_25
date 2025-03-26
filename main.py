import cv2
import os
import pandas as pd
import argparse
from utls import create_template_from_labeled_image


def main(show=False):
    # === Paths ===
    data_folder = 'data'
    char_data_folder = 'basic'
    labels_file = os.path.join('.', data_folder, char_data_folder, 'labels.csv')
    image_folder = os.path.join('.', data_folder, char_data_folder, 'dataset')

    test_image = 'emoji_95.jpg'  # any image from the dataset

    # === Load Labels ===
    df = pd.read_csv(labels_file, sep=';')
    df.columns = ['id', 'file_name', 'label', 'x', 'y']

    # === Convert strings to usable Python types ===
    df['label'] = df['label'].apply(eval)
    df['x'] = df['x'].apply(lambda x: int(eval(x)[0]))
    df['y'] = df['y'].apply(lambda x: int(eval(x)[0]))

    # === Load Images ===
    image = cv2.imread(os.path.join(image_folder, test_image), cv2.IMREAD_GRAYSCALE)
    template_row = df.iloc[0]
    template, emoji_h, emoji_w = create_template_from_labeled_image(template_row, image_folder)

    if image is None or template is None:
        print("❌ Failed to load image or template.")
        exit()

    # === Initialize ORB Detector ===
    orb = cv2.ORB_create(nfeatures=500)

    # === Find Keypoints and Descriptors ===
    kp1, des1 = orb.detectAndCompute(template, None)
    kp2, des2 = orb.detectAndCompute(image, None)

    # === Match Descriptors ===
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # === Draw Top N Matches ===
    matched_img = cv2.drawMatches(template, kp1, image, kp2, matches[:20], None, flags=2)

    # === Show Result ===
    cv2.imshow("ORB Feature Matches", matched_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Emoji template matching with optional visualization.")
    parser.add_argument('--show', action='store_true', help="Show matched image with rectangle.")
    args = parser.parse_args()

    main(show=args.show)
