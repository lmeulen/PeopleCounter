#
# Usage
# python detector.py --config config.ini
#
# It uses a pre-trained YOLO v3 network for object detection, trained on the COCO dataset
# Yolo v3: https://arxiv.org/abs/1804.02767
import argparse
import time
import os
import configparser
import cv2
import numpy as np
import pandas as pd
import csv
import datetime
import matplotlib.pyplot as plt

def define_args():
    """
    Specify the arguments of the application
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config",  required=True, help="Configuration file")
    return vars(ap.parse_args())


def read_config(filename):
    """
    Read the configuration file
    :param filename: Filename of the configuration file
    :return: configuration object
    """
    print("[INFO] Reading config: {}".format(filename))
    cfg = configparser.ConfigParser()
    cfg.read(filename)
    return cfg


def save_count(filename, n):
    """
    Save the specified value to a file.
    Value is appended to the end of the file
    Format: <timestamp> , <value>
    :param filename: filename of targetfile
    :param n: value to store
    :return:
    """
    timestamp = str(datetime.datetime.now().strftime("%Y%m%d_%H-%M-%S"))
    f = open(filename, "a")
    line = "{} , {}\n".format(timestamp, n)
    f.write(line)
    f.close()


def read_existing_data(filename):
    """
    Read existing data from file
    :param filename: filename of targetfile
    :return: timestamps, measurements
    """
    times = []
    values = []
    if os.path.isfile(filename):
        with open(filename) as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            for row in csv_reader:
                times.append(datetime.datetime.strptime(row[0], "%Y%m%d_%H-%M-%S "))
                values.append(int(row[1]))
    dataframe = pd.DataFrame()
    dataframe['timestamp'] = pd.Series(dtype='datetime64[ns]')
    dataframe['value'] = pd.Series(dtype=np.int64)
    dataframe['timestamp'] = times
    dataframe['value'] = values
    dataframe.set_index('timestamp', inplace=True)
    return dataframe


def blur_area(image, top_x, top_y, w, h):
    """
     Blur the specified area of the frame.
     BLurred area = <x,y> - <x+w, y+h>
     :type image: RGB array
     :type top_x: int
     :type top_y: int
     :type w: int
     :type h: int
    """
    # get the rectangle img around all the faces and apply blur
    sub_frame = image[top_y:top_y+h, top_x:top_x+w]
    sub_frame = cv2.GaussianBlur(sub_frame, (31, 31), 30)
    # merge back into the frame
    image[top_y:top_y+sub_frame.shape[0], top_x:top_x+sub_frame.shape[1]] = sub_frame
    return image


def execute_network(image, network, layernames):
    """
    Pull frame through the network
    :type image: RGB array
    :type network: object containing Yolo network
    :type layernames: array of layer names
    """
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    start2 = time.time()
    network.setInput(blob)
    outputs = network.forward(layernames)
    end2 = time.time()
    print("[INFO] YOLO  took      : %2.1f sec" % (end2-start2))
    return outputs


def load_network(networkpath):
    """
    Load the Yolo network from disk.
    Path specified as application argument
    """
    # load the COCO class labels our YOLO model was trained on
    labelspath = os.path.sep.join([networkpath, "coco.names"])
    labels = open(labelspath).read().strip().split("\n")

    # derive the paths to the YOLO weights and model configuration
    weightspath = os.path.sep.join([networkpath, "yolov3.weights"])
    configpath = os.path.sep.join([networkpath, "yolov3.cfg"])

    # load YOLO object detector trained on COCO dataset (80 classes)
    # and determine only the *output* layer names that we need from YOLO
    # Network storend in Darknet format
    print("[INFO] loading YOLO from disk...")
    network = cv2.dnn.readNetFromDarknet(configpath, weightspath)
    names = network.getLayerNames()
    names = [names[i[0] - 1] for i in network.getUnconnectedOutLayers()]
    return network, names, labels


def get_detected_items(layeroutputs, confidence_level, threshold):
    """
    Determine the objects as found by the network. Found objects are filtered
    on confidence leven and threshold.
    """
    
    # initialize our lists of detected bounding boxes, confidences, and class IDs
    boxes = []
    confidences = []
    classids = []

    for output in layeroutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of the current object detection
            scores = detection[5:]
            classid = np.argmax(scores)
            confidence = scores[classid]

            # filter out weak predictions by ensuring the detected probability is greater than the minimum probability
            if confidence > confidence_level:
                # scale the bounding box coordinates back relative to the size of the image
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top left corner of the bounding box
                top_x = int(centerX - (width / 2))
                top_y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences, and class IDs
                boxes.append([top_x, top_y, int(width), int(height)])
                confidences.append(float(confidence))
                classids.append(classid)

    # apply non-maxima suppression to suppress weak, overlapping bounding boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_level, threshold)

    return indexes, classids, boxes, confidences


def get_videowriter(outputfile, width, height, frames_per_sec=30):
    """
    Create a writer for the output video
    """
    # Initialise the writer
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    video_writer = cv2.VideoWriter(outputfile, fourcc, frames_per_sec, (width, height), True)
    return video_writer, frames_per_sec


def get_webcamesource(webcam_id, width=640, height=480):
    """
    Create a reader for the input video. Input can be a webcam
    or a videofile
    """
    print("[INFO] initialising video source...")
    vc = cv2.VideoCapture(webcam_id)
    vc.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    vc.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    (success, videoframe) = vc.read()
    (height, width) = videoframe.shape[:2]
    print("[INFO] Frame W x H: {} x {}".format(width, height))
    return vc, width, height


def get_filesource(filename):
    """
    Create a reader for the input video
    """
    print("[INFO] initialising video source : {}".format(filename))
    vs = cv2.VideoCapture(filename)
    width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("[INFO] Frame W x H: {} x {}".format(width, height))
    return vs, width, height


def count_people(indexes, class_ids):
    """
    Count the number of people found in the video
    """
    count = 0
    # ensure at least one detection exists
    if len(indexes) > 0:
        for i in indexes.flatten():
            if class_ids[i] == 0:
                count += 1
    return count


def update_frame(frame, idxs, class_ids, boxes, confidences, colors, labels, peoplecount,
                 showpeopleboxes, blurpeople, showallboxes):
    """
    Add bounding boxes and counted number of people to the frame
    """
    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y, w, h) = (boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3])

            # Blur, if required, people in the image
            if blurpeople and classIDs[i] == 0:
                frame = blur_area(frame, max(x, 0), max(y, 0), w, h)

            # draw a bounding box rectangle and label on the frame
            if (showpeopleboxes and classIDs[i] == 0) or showallboxes:
                color = [int(c) for c in colors[class_ids[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(labels[classIDs[i]], confidences[i])
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # write number of people in bottom corner
    text = "Persons: {}".format(peoplecount)
    cv2.putText(frame, text, (10, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return frame


def show_plots(data):
    df_1w = data[data.index >= pd.datetime.now() - pd.Timedelta('7D')]
    df_1d = df_1w[df_1w.index >= pd.datetime.now() - pd.Timedelta('24H')]
    df_8h = df_1d[df_1d.index >= pd.datetime.now() - pd.Timedelta('8H')]
    df_2h = df_8h[df_8h.index >= pd.datetime.now() - pd.Timedelta('2H')]
    # Resample to smooth the long running graphs
    df_1w = df_1w.resample('1H').max()
    df_1d = df_1d.resample('15min').max()

    plt.gcf().clear()

    plt.subplot(2, 2, 1)
    plt.plot(df_1w.index.tolist(), df_1w['value'].tolist())
    plt.title("Laatste week")
    plt.ylabel("Personen")
    plt.xlabel("Tijdstip")

    plt.subplot(2, 2, 2)
    plt.plot(df_1d.index.tolist(), df_1d['value'].tolist())
    plt.title("Afgelopen 24 uur")
    plt.ylabel("Personen")
    plt.xlabel("Tijdstip")

    plt.subplot(2, 2, 3)
    plt.plot(df_8h.index.tolist(), df_8h['value'].tolist())
    plt.title("Afgelopen 8 uur")
    plt.ylabel("Personen")
    plt.xlabel("Tijdstip")

    plt.subplot(2, 2, 4)
    plt.plot(df_2h.index.tolist(), df_2h['value'].tolist())
    plt.title("Afgelopen 2 uur")
    plt.ylabel("Personen")
    plt.xlabel("Tijdstip")

    plt.gcf().autofmt_xdate()
    plt.show()


if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    args = define_args()
    config = read_config(args["config"])
    
    # Load the trained network
    (net, ln, LABELS) = load_network(config['NETWORK']['Path'])
    
    # Initialise video source
    webcam = (config['READER']['Webcam'] == "yes")
    if webcam:
        cam_id = int(config['READER']['WebcamID'])
        cam_width = int(config['READER']['Width'])
        cam_height = int(config['READER']['Height'])
        (cam, W, H) = get_webcamesource(cam_id, cam_width, cam_height)
    else:
        (cam, W, H) = get_filesource(config['READER']['Filename'])
    
    # determine if we need to show the enclosing boxes
    showpeopleboxes = (config['OUTPUT']['ShowPeopleBoxes'] == "yes")
    showallboxes = (config['OUTPUT']['ShowAllBoxes'] == "yes")
    blurpeople = (config['OUTPUT']['BlurPeople'] == "yes")
    realspeed = (config['OUTPUT']['RealSpeed'] == "yes")
    nw_confidence = float(config['NETWORK']['Confidence'])
    nw_threshold = float(config['NETWORK']['Threshold'])
    print("[INFO] Confidence: {}, Threshold: {} ".format(nw_confidence, nw_threshold))
    countfile = config['OUTPUT']['Countfile']
    save_video = (config['OUTPUT']['SaveVideo'] == "yes")
    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

    # Initialise video ouptut writer
    if save_video:
        (writer, fps) = get_videowriter(config['OUTPUT']['Filename'], W, H, int(config['OUTPUT']['FPS']))

    # Create output windows
    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Video', 600, 600)
    cv2.moveWindow('Video', 0, 0)
    # Create plot
    plt.ion()
    plt.figure(num=None, figsize=(8, 7), dpi=80, facecolor='w', edgecolor='k')

    # Initialise with existing reading
    df = read_existing_data(countfile)

    # loop while true
    while True:
        # read the next frame from the webcam
        (grabbed, frame) = cam.read()  # type: (object, object)
        if not grabbed:
            break

        start = time.time()

        # Feed frame to network
        layerOutputs = execute_network(frame, net, ln)
        # Obtain detected objects, including cof levels and bounding boxes
        (idxs, classIDs, boxes, confidences) = get_detected_items(layerOutputs, nw_confidence, nw_threshold)
        npeople = count_people(idxs, classIDs)
        print("[INFO] People in frame : {}".format(npeople))
        save_count(countfile, npeople)

        # Add row to panda frame
        new_row = pd.DataFrame([[npeople]], columns = ["value"], index=[pd.to_datetime(datetime.datetime.now())])
        df = pd.concat([df, pd.DataFrame(new_row)], ignore_index=False)

        # Update frame with recognised objects
        frame = update_frame(frame, idxs, classIDs, boxes, confidences, COLORS, LABELS, npeople, showpeopleboxes,
                             blurpeople, showallboxes)

        # Show plot
        show_plots(df)
        # Show frame with bounding boxes on screen
        cv2.imshow('Video', frame)

        end = time.time()
        # write the output frame to disk, repeat (time taken * 30 fps) in order to get a video at real speed
        if save_video:
            writer.write(frame)
            if webcam and realspeed:
                for x in range(1, int((end-start)*fps)):
                    writer.write(frame)

        # Check for exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # release the file pointers
    print("[INFO] cleaning up...")
    writer.release()
    cam.release()
