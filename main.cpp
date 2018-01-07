#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>


#include "helper.hpp"
#include "eigenfaces.hpp"

using namespace std;
using namespace cv;
Eigenfaces * eigenfaces;

void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels) {
	std::ifstream file(filename.c_str(), ifstream::in);
	if(!file)
		throw std::exception();
	std::string line, path, classlabel;
	// for each line
	while (std::getline(file, line)) {
		// get current line
		std::stringstream liness(line);
		// split line
		std::getline(liness, path, ';');
		std::getline(liness, classlabel);
		// push pack the data
		images.push_back(imread(path,0));
		labels.push_back(atoi(classlabel.c_str()));
	}
}

int main(int argc, char *argv[]) {
	VideoCapture capture;
	Mat frame;
	Mat compareFile;

	vector<Mat> images;
	vector<int> labels;
	// check for command line arguments
	if(argc < 2) {
		cout << "usage: " << argv[0] << " <csv.ext>" << endl;
		exit(1);
	}

	// path to your CSV
	string fn_csv = string(argv[1]);
	// read in the images
	try {
		read_csv(fn_csv, images, labels);
	} catch(exception& e) {
		cerr << "Error opening file \"" << fn_csv << "\"." << endl;
		exit(1);
	}

	if(argc == 3) {
		compareFile = images[atoi(argv[2])];
	}

	// get width and height
	int width = images[0].cols;
	int height = images[0].rows;
	// get test instances

	// num_components eigenfaces
	int num_components = 80;
	// compute the eigenfaces
	eigenfaces = new Eigenfaces(images, labels, num_components);


	capture.open(0);

	if (!capture.isOpened()) {
		printf("--(!)Error opening video capture\n");
		return -1;
	}

	while(capture.read(frame)) {
		if(frame.empty()) {
			printf(" --(!) No captured frame -- Break!");
			break;
		}

		Mat frameCompare = frame;
		cvtColor(frameCompare, frameCompare, CV_RGB2GRAY);

		frameCompare(Rect(300, 0, 302, 403)).copyTo(frameCompare);

		if(argc == 3) {
			compareFile.copyTo(frameCompare);
		}

		flip(frameCompare, frameCompare, 1);


		imshow("frame", frameCompare);

		int predicted = eigenfaces->predict(frameCompare);

		cout << "predicted: " << predicted << endl;

		std::vector<int>::iterator it = std::find (labels.begin(), labels.end(), predicted);
		auto index = std::distance(labels.begin(), it);


		Mat p = eigenfaces->project(frameCompare.reshape(1,1));
		Mat r = eigenfaces->reconstruct(p);

		imshow("prediction", images[index]);
		imshow("reconstruction", toGrayscale(r.reshape(1, height)));




		char c = (char)waitKey(10);
		if(c == 27) // escape
			break;
	}






	/*// get a prediction
	int predicted = eigenfaces->predict(testSample);


	// see the reconstruction with num_components
	Mat p = eigenfaces->project(images[0].reshape(1,1));
	Mat r = eigenfaces->reconstruct(p);
	imshow("original", images[0]);
	imshow("reconstruction", toGrayscale(r.reshape(1, height)));
	// get the eigenvectors
	Mat W = eigenfaces->eigenvectors();
	// show first 10 eigenfaces
	for(int i = 0; i < min(10,W.cols); i++) {
		Mat ev = W.col(i).clone();
		imshow(format("%d", i), toGrayscale(ev.reshape(1, height)));
	}
	waitKey(0);*/
	return 0;
}

