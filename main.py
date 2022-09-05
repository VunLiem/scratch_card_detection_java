import csv
import torch
import cv2
import numpy as np
import easyocr
from helper.general_utils import save_results

# DEFINING GLOBAL VARIABLE
EASY_OCR = easyocr.Reader(['en'],gpu=False)
OCR_TH = 0.2

# -------------------------------------- Function to run detection ---------------------------------------------------------
def detectx(frame, model):
    frame = [frame]
    print(f"[INFO] Detecting. . . ")
    results = model(frame)
    # results.show()
    # print(results.xyxyn[0])
    # print(results.xyxyn[0][:, -1])
    # print(results.xyxyn[0][:, :-1])

    labels, cordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

    return labels, cordinates


# ------------------------------------ To plot the BBox and results --------------------------------------------
def plot_boxes(results, frame, classes):
    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    print(f"[INFO] Total {n} detections. . . ")
    print(f"[INFO] Looping through all detections. . . ")

# -------------------------------------Looping through the detections------------------------------------
    for i in range(n):
        row = cord[i]
        if row[4] >= 0.55:  ### threshold value for detection. We are discarding everything below this value
            print(f"[INFO] Extracting BBox coordinates. . . ")
            # Bounding box coordinates
            x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), \
                             int( row[3] * y_shape)
            text_d = classes[int(labels[i])]
            #cv2.imwrite("dp.jpg",frame[int(y1):int(y2), int(x1):int(x2)])

            coords = [x1, y1, x2, y2]

            scratch_card_num = recognize_scratch_card_easyocr(img=frame, coords=coords, reader=EASY_OCR, region_threshold=OCR_TH)

            # if text_d == 'mask':
            cv2.rectangle(frame, (x1, y1), (x2, y2), (193, 0, 255), 2)  ## BBox
        # Open this when you show the result to the screen
            #cv2.rectangle(frame, (x1, y1 - 20), (x2, y1), (193, 0,255), -1)  ## for text label background
            #cv2.putText(frame, f"{scratch_card_num}", (x1, y1-3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,225), 2)

            #cv2.imwrite("np.jpg",frame[int(y1)-25:int(y2)+25, int(x1)-25:int(x2)+25])

    return frame

# ---------------------------- Function to recognize scratch card --------------------------------------

# Function to recognize scratch card numbers using easyOCR
def recognize_scratch_card_easyocr(img, coords, reader, region_threshold):
    # Separate coordinates from box
    xmin, ymin, xmax, ymax = coords

    # Get the subimage that makes up the bounded region and take an additional 5 pixels on each side
    # nplate = img[int(ymin)-5:int(ymax)+5, int(xmin)-5:int(xmax)+5]
    nplate = img[int(ymin):int(ymax), int(xmin):int(xmax)]  ### Cropping the scratch number from the whole image
    # Reading OCR inside the image from nplate image
    ocr_result = reader.readtext(nplate)

    text = filter_text(region=nplate, ocr_result=ocr_result, region_threshold=region_threshold)
    if len(text) == 1:
        text = text[0].upper()
    return text

# To filter out wrong detections
def filter_text(region, ocr_result, region_threshold):
    rectangle_size = region.shape[0] * region.shape[1]

    scratch_card = []
    # Select number only
    num = ""
    for c in (ocr_result[0][-2]):
        if c.isdigit():
            num = num + c
    # print(num)

#------Save result to .csv-------
    # header = ['scratch numbers']
    # data = [ocr_result[0][-2],ocr_result[0][-1]]
    # with open('scratch_card.csv','w',encoding='UTF8') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(header)
    #     writer.writerow(data)
#-----------------------------------

    # Saving text only into file ocr_detection.csv
    save_results(num, 'ocr_detection.csv', 'Detection_Image')
    # Saving text and coordinates of the 4 corners into file ocr_results.csv
    # save_results(ocr_result[-1],'ocr_results.csv','Detection_Image')
    print("Results of OCR detection inside the bounding box: " + num +'\n'+"Accurate detector value: " + str(ocr_result[0][-1]))
    # print("Accurate detector value: " + str(ocr_result[0][-1]))


    for result in ocr_result:
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))

        if length * height / rectangle_size > region_threshold:
            scratch_card.append(result[1])
    return scratch_card


# ---------------------------------------------- Main function -----------------------------------------------------

def main(img_path=None):
    print(f"[INFO] Loading model... ")
    ## loading the custom trained model
    # model =  torch.hub.load('ultralytics/yolov5', 'custom', path='last.pt',force_reload=True) ## if you want to download the git repo and then run the detection
    model = torch.hub.load('yolov5-master', 'custom', source='local', path='best_data.pt',
                           force_reload=True)  ### The repo is stored locally

    classes = model.names  ### class names in string format

# --------------- for detection on image --------------------
    if img_path != None:

        print(f"[INFO] Working with image: {img_path}")

        img_out_name = f"./output/result_{img_path.split('/')[-1]}"

        frame = cv2.imread(img_path)  ### reading the image
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detectx(frame, model=model)  ### DETECTION HAPPENING HERE
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = plot_boxes(results, frame, classes=classes)
        while True:
            #frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            # cv2.imwrite("img2.jpg", frame)
            # cv2.imshow("img" ,frame)
            if cv2.waitKey() & 0xFF == ord('q'):
                print(f"[INFO] Exiting. . . ")
                cv2.imwrite(f"{img_out_name}", frame)  ## if you want to save he output result
            break
        print(f"[INFO] Clening up. . . ")

# -------------------  calling the main function-------------------------------
main(img_path="32.jpg")  ## for local image



