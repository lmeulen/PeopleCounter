""""
Detect people (and other objects) in a videostream. Videostream may be a
specified webcam or videofile.
It can show graphs with the history of detected people. People count is
always saved to file.

Usage: python detector.py --config config.ini

It uses a pre-trained YOLO v3 network for object detection, trained on the COCO dataset
Yolo v3: https://arxiv.org/abs/1804.02767
"""
import argparse
import time
import os
import sys
import configparser
import csv
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import cv2

def define_args():
    """
    Specify the arguments of the application
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True, help="Configuration file")
    return vars(ap.parse_args())


def download_if_not_present(url, file_name):
    """
    Check if file is present, if not, download from url
    :param url: Full URL to download location
    :param file_name: filename for string the file, may include paths
    :return:
    """
    if not os.path.exists(file_name):
        with open(file_name, "wb") as f:
            response = requests.get(url, stream=True)
            total_length = response.headers.get('content-length')
            if total_length is None:
                # no content length header
                f.write(response.content)
            else:
                downloaded = 0
                total_length = int(total_length)
                for data in response.iter_content(chunk_size=4096):
                    downloaded += len(data)
                    f.write(data)
                    percentage = int(100 * downloaded / total_length)
                    progress = int(50 * downloaded / total_length)
                    sys.stdout.write("\rDownloading {} [{} {}] {}%".format(file_name, '=' * progress,
                                                                           ' ' * (50-progress), percentage))
                    sys.stdout.flush()
                sys.stdout.write("\n")
                sys.stdout.flush()


def read_config(filename):
    """
    Read the configuration file
    :param filename: Filename of the configuration file
    :return: configuration object
    """
    print("[INFO] Reading config: {}".format(filename))
    if not os.path.isfile(filename):
        print("[ERROR] Config file \"{}\" not found.".format(filename))
        exit()
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
    f = open(filename, "a")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H-%M-%S")
    line = "{} , {}\n".format(timestamp, n)
    f.write(line)
    f.close()


def read_existing_data(filename):
    """
    Read existing data from file. If ifle not found, an empty
    initialized frame is returned
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
    dataframe['value'] = pd.Series(dtype=np.int32)
    dataframe['timestamp'] = times
    dataframe['value'] = values
    dataframe.set_index('timestamp', inplace=True)
    return dataframe


def blur_area(image, top_x, top_y, w, h):
    """
     Blur the specified area of the frame.
     Blurred area = <x,y> - <x+w, y+h>
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


def load_network(network_folder):
    """
    Load the Yolo network from disk.
    https://pjreddie.com/media/files/yolov3.weights
    https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg

    :param network_folder: folder where network files are stored
    """
    # Derive file paths and check existance
    labelspath = os.path.sep.join([network_folder, "coco.names"])
    if not os.path.isfile(labelspath):
        print("[ERROR] Network: Labels file \"{}\" not found.".format(labelspath))
        exit()

    weightspath = os.path.sep.join([network_folder, "yolov3.weights"])
    download_if_not_present("https://pjreddie.com/media/files/yolov3.weights", weightspath)
    if not os.path.isfile(weightspath):
        print("[ERROR] Network: Weights file \"{}\" not found.".format(weightspath))
        exit()

    configpath = os.path.sep.join([network_folder, "yolov3.cfg"])
    download_if_not_present("https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg", configpath)
    if not os.path.isfile(configpath):
        print("[ERROR] Network: Configuration file \"{}\" not found.".format(configpath))
        exit()

    # load YOLO object detector trained on COCO dataset (80 classes)
    # and determine only the *output* layer names that we need from YOLO
    # Network storend in Darknet format
    print("[INFO] loading YOLO from disk...")
    labels = open(labelspath).read().strip().split("\n")
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
                (center_x, center_y, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top left corner of the bounding box
                top_x = int(center_x - (width / 2))
                top_y = int(center_y - (height / 2))

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


def save_frame(video_writer, new_frame, count=1):
    """
    Save frame <count> times to file.
    :param video_writer: writer for target file
    :param new_frame: frame to write
    :param count: number of times to write the frame
    :return:
    """
    for _ in range(0, count):
        video_writer.write(new_frame)


def get_webcamesource(webcam_id, width=640, height=480):
    """
    Create a reader for the input video. Input can be a webcam
    or a videofile
    """
    print("[INFO] initialising video source...")
    video_device = cv2.VideoCapture(webcam_id)
    video_device.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    video_device.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    (success, videoframe) = video_device.read()
    if not success:
        print("[ERROR] Could not read from webcam id {}".format(webcam_id))
    (height, width) = videoframe.shape[:2]
    print("[INFO] Frame W x H: {} x {}".format(width, height))
    return video_device, width, height


def get_filesource(filename):
    """
    Create a reader for the input video
    """
    print("[INFO] initialising video source : {}".format(filename))
    video_device = cv2.VideoCapture(filename)
    width = int(video_device.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_device.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("[INFO] Frame W x H: {} x {}".format(width, height))
    return video_device, width, height


def update_frame(frame, idxs, class_ids, boxes, confidences, colors, labels,
                 show_boxes, blur, box_all_objects):
    """
    Add bounding boxes and counted number of people to the frame
    Return frame and number of people
    """
    # ensure at least one detection exists
    count_people = 0
    if len(idxs) >= 1:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y, w, h) = (boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3])

            if classIDs[i] == 0:
                count_people += 1
                # Blur, if required, people in the image
                if blur:
                    frame = blur_area(frame, max(x, 0), max(y, 0), w, h)

            # draw a bounding box rectangle and label on the frame
            if (show_boxes and classIDs[i] == 0) or box_all_objects:
                color = [int(c) for c in colors[class_ids[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(labels[classIDs[i]], confidences[i])
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # write number of people in bottom corner
    text = "Persons: {}".format(count_people)
    cv2.putText(frame, text, (10, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return frame, count_people


def show_plots(data):
    """
    Show the graphs with historical data
    :param data: dataframe
    :return:
    """
    # Awful code to create new dataframes each time the graph is shown
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

    # determine if we need to show the enclosing boxes, etc
    showpeopleboxes = (config['OUTPUT']['ShowPeopleBoxes'] == "yes")
    showallboxes = (config['OUTPUT']['ShowAllBoxes'] == "yes")
    blurpeople = (config['OUTPUT']['BlurPeople'] == "yes")
    realspeed = (config['OUTPUT']['RealSpeed'] == "yes")
    nw_confidence = float(config['NETWORK']['Confidence'])
    nw_threshold = float(config['NETWORK']['Threshold'])
    countfile = config['OUTPUT']['Countfile']
    save_video = (config['OUTPUT']['SaveVideo'] == "yes")
    show_graphs = (config['OUTPUT']['ShowGraphs'] == "yes")
    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

    # Initialise video ouptut writer
    if save_video:
        (writer, fps) = get_videowriter(config['OUTPUT']['Filename'], W, H, int(config['OUTPUT']['FPS']))
    else:
        (writer, fps) = (None, 0)

    # Create output windows
    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Video', 600, 600)
    cv2.moveWindow('Video', 0, 0)
    # Create plot
    if show_graphs:
        plt.ion()
        plt.figure(num=None, figsize=(8, 7), dpi=80, facecolor='w', edgecolor='k')
        df = read_existing_data(countfile)
    else:
        df = None

    # loop while true
    while True:
        start = time.time()
        # read the next frame from the webcam
        (grabbed, frame) = cam.read()  # type: (object, object)
        if not grabbed:
            break
        # Feed frame to network
        layerOutputs = execute_network(frame, net, ln)
        # Obtain detected objects, including cof levels and bounding boxes
        (idxs, classIDs, boxes, confidences) = get_detected_items(layerOutputs, nw_confidence, nw_threshold)

        # Update frame with recognised objects
        frame, npeople = update_frame(frame, idxs, classIDs, boxes, confidences, COLORS, LABELS, showpeopleboxes,
                                      blurpeople, showallboxes)
        save_count(countfile, npeople)

        if show_graphs:
            # Add row to panda frame
            new_row = pd.DataFrame([[npeople]], columns=["value"], index=[pd.to_datetime(datetime.datetime.now())])
            df = pd.concat([df, pd.DataFrame(new_row)], ignore_index=False)
            show_plots(df)

        # Show frame with bounding boxes on screen
        cv2.imshow('Video', frame)
        # write the output frame to disk, repeat (time taken * 30 fps) in order to get a video at real speed
        if save_video:
            frame_cnt = int((time.time()-start)*fps) if webcam and realspeed else 1
            save_frame(writer, frame, frame_cnt)

        end = time.time()
        print("[INFO] Total handling  : %2.1f sec" % (end - start))
        print("[INFO] People in frame : {}".format(npeople))

        # Check for exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # release the file pointers
    print("[INFO] cleaning up...")
    if save_video:
        writer.release()
    cam.release()
