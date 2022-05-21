from cnstd import CnStd
import numpy as np
import cv2
import pickle
from DANN import dann_model, dann_scenario_dict, save_check_pt
from CCSA import ccsa_model, save_CCSA, ccsa_scenario_dict


std = CnStd(rotated_bbox=False)
label2index = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "A": 10,
    "B": 11,
    "C": 12,
    "D": 13,
    "E": 14,
    "F": 15,
    "G": 16,
    "H": 17,
    "J": 18,
    "K": 19,
    "L": 20,
    "M": 21,
    "N": 22,
    "P": 23,
    "Q": 24,
    "R": 25,
    "S": 26,
    "T": 27,
    "U": 28,
    "V": 29,
    "W": 30,
    "X": 31,
    "Y": 32,
    "Z": 33,
}
index2label = dict(zip(label2index.values(), label2index.keys()))

example2file = {
    "Example 1": "1.png",
    "Example 2": "2.png",
    "Example 3": "3.png",
    "Example 4": "4.png",
    "Example 5": "5.jpg",
    "Example 6": "6.jpg",
    "Example 7": "7.jpg",
    "Example 8": "8.jpg",
    "Example 9": "9.jpg",
}


def load_model(model_name="Logistic Regression", scenario_select=None):
    """
    Currently available models:
        "Logistic Regression": Logistic regression model
        "SVM": Support vector machine model
    """
    model = None
    if model_name == "Logistic Regression":
        with open("models/lgr.pkl", "rb") as f:
            model = pickle.load(f)
    elif model_name == "SVM":
        with open("models/svm.pkl", "rb") as f:
            model = pickle.load(f)
    elif model_name == "Domain-Adversarial Neural Networks":
        assert scenario_select in dann_scenario_dict
        dann_model.load_weights(save_check_pt + dann_scenario_dict[scenario_select])
        model = dann_model
    elif model_name == "Classification and Contrastive Semantic Alignment":
        assert scenario_select in ccsa_scenario_dict
        ccsa_model.load_weights(save_CCSA + ccsa_scenario_dict[scenario_select])
    return model


def do_predict(model, crop_list, model_name=None):
    X = np.vstack(crop_list)
    if model_name == "Domain-Adversarial Neural Networks" or model_name == "Classification and Contrastive Semantic Alignment":
        X = X.reshape([-1, 32, 16, 3])
        Y = model.predict(X)
        Y = np.argmax(Y, axis=1)
    else:
        Y = model.predict(X)
    ret = []
    for each in Y:
        ret.append(index2label[each])
    return ret


def split_character(img_input, show_image=False, color_map="gray"):
    """
    img_input : original image
    """

    box_infos = std.detect(img_input, resized_shape=(320, 960))
    img = box_infos["detected_texts"][0]["cropped_img"]

    # Algorithm source: https://blog.csdn.net/twilight737/article/details/118190551
    # img1 : resize image
    img1 = cv2.resize(img, (320, 100), interpolation=cv2.INTER_AREA)
    # img2 : gray image
    img2 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    # img3 : noise reduction image
    img3 = cv2.bilateralFilter(img2, 11, 17, 17)
    # img4 : texture information image
    img4 = cv2.Canny(img3, 50, 150)
    # img5 : image removal image
    img5 = img4[10:90, 10:310]
    crop_img = img1[10:90, 10:310, :]
    contours, hierarchy = cv2.findContours(img5, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidate = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h < 500:
            continue
        if w * h > 4000:
            continue
        if h < 20:
            continue
        if w > 80:
            continue
        candidate.append([x, (x + w)])
    loc = np.zeros(300)

    for j in range(len(candidate)):
        x1 = candidate[j][0]
        x2 = candidate[j][1]
        loc[x1:x2] = 1
    start = []
    end = []

    if loc[0] == 1:
        start.append(0)
    for j in range(300 - 1):
        if loc[j] == 0 and loc[j + 1] == 1:
            start.append(j)
        if loc[j] == 1 and loc[j + 1] == 0:
            end.append(j)

    if loc[299] == 1:
        end.append(299)

    # print('Character start coordination : ', start, len(start))
    # print('Character end coordination : ', end, len(start))

    crop_list = []

    if len(start) == 7 and len(end) == 7:
        print("The Segmentation looks well ! ")
        cv2.rectangle(crop_img, (0, 0), (start[1] - 5, 80), (0, 0, 255), 2)
        for j in range(1, 7):
            x1 = start[j]
            x2 = end[j]
            y1 = 0
            y2 = 80
            cv2.rectangle(crop_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            acrop = cv2.resize(crop_img[y1:y2, min(x1, x2) : max(x1, x2)], (16, 32), interpolation=cv2.INTER_AREA)
            if color_map == "gray":
                acrop = cv2.cvtColor(acrop, cv2.COLOR_RGB2GRAY)
                crop_list.append(acrop.reshape([32 * 16]))
            elif color_map == "rgb":
                crop_list.append(acrop.reshape([32 * 16 * 3]))

    if show_image:
        cv2.namedWindow("final_crop_img")
        cv2.imshow("final_crop_img", crop_img)
        cv2.waitKey(0)
    return crop_list
