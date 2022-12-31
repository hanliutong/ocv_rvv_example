#include <fstream>
#include <sstream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/filesystem.hpp>

using namespace cv;
using namespace dnn;

std::vector<std::string> classes;

int main(int argc, char** argv)
{
    Scalar mean = Scalar(104, 117, 123);
    bool swapRB = true;
    bool crop = false;
    int inpWidth = 224;
    int inpHeight = 224;
    
    String model = cv::String("bvlc_googlenet.caffemodel");
    String config = cv::String("bvlc_googlenet.prototxt");
    String framework = "";
    int backendId = 0;
    int targetId = 0;


    printf("Open file with classes names.\n"); 
    std::string file = "classification_classes_ILSVRC2012.txt";
    std::ifstream ifs(file.c_str());
    if (!ifs.is_open())
        CV_Error(Error::StsError, "File " + file + " not found");
    std::string line;
    while (std::getline(ifs, line))
    {
        classes.push_back(line);
    }
    CV_Assert(!model.empty());

    printf("Read and initialize network\n");
    Net net = readNet(model, config, framework);
    net.setPreferableBackend(backendId);
    net.setPreferableTarget(targetId);


    printf("Process frames.\n");// Process frames.
    std::string input = "ILSVRC2011_val_00000001.JPEG";
    Mat frame = imread(input);
    printf("Create a 4D blob from a frame\n");// Create a 4D blob from a frame
    Mat blob;
    blobFromImage(frame, blob, 1, Size(inpWidth, inpHeight), mean, swapRB, crop);
    printf("Set input blob\n");// Set input blob
    net.setInput(blob);

    printf("Make forward pass\n");// Make forward pass
    Mat prob = net.forward();

    printf("Get a class with a highest score\n");// Get a class with a highest score
    Point classIdPoint;
    double confidence;
    minMaxLoc(prob.reshape(1, 1), 0, &confidence, 0, &classIdPoint);
    int classId = classIdPoint.x;

    
    std::vector<double> layersTimes;
    double freq = getTickFrequency() / 1000;
    double t = net.getPerfProfile(layersTimes) / freq;
    
    printf("Put efficiency information.\n");// Put efficiency information.
    std::string label = format("Inference time: %.2f ms", t);
    putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));

    printf("Print predicted class.\n");// Print predicted class.
    label = format("%s: %.4f", (classes.empty() ? format("Class #%d", classId).c_str() :
                                                    classes[classId].c_str()),
                                confidence);
    printf("%s\ttime:%.2f\t %.1f%% %s\n", input.c_str(), t, confidence*100, classes[classId].c_str());
    putText(frame, label, Point(0, 40), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));

    imwrite("output.jpg",frame);
    return 0;
}
