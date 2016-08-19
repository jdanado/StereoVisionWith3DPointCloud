/**
 * @file SBM_Sample
 * @brief Get a disparity map of two images
 * @author A. Huaman
 */

#include <iostream>
#include "opencv2/calib3d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "GL\freeglut.h"

using namespace cv;
using namespace std;

void readme(void);
void reshape(int w, int h);
void renderScene(void);
void init(void);
void centerOnScreen(void);
void createDisparityMap(char** argv);
void create3DCloud(void);
void initializeOpenGL(int argc, char** argv);
void printImage(void);

//  define the window position on screen
int window_x;
int window_y;

Mat disparityMap;
Mat pointCloud;
Mat image;

void reshape(int w, int h)
{
	glViewport(0, 0, (GLsizei)w, (GLsizei)h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60, (GLfloat)w / (GLfloat)h, 1.0, 500.0);
	glMatrixMode(GL_MODELVIEW);
}

//-------------------------------------------------------------------------
//  This function is passed to glutDisplayFunc in order to display 
//  OpenGL contents on the window.
//-------------------------------------------------------------------------
void renderScene(void)
{
	glClear(GL_COLOR_BUFFER_BIT);

	glBegin(GL_POINTS);
	for (int y = 0; y < pointCloud.rows; y++)
	{
		for (int x = 0; x < pointCloud.cols; x++)
		{
			Vec3f point = pointCloud.at<Vec3f>(y, x);
			Vec3b color = image.at<Vec3b>(y, x);
			glColor3d(color[0] / 255, color[1] / 255, color[2] / 255);
			glVertex3f(point[0], point[1], point[2]);
		}
	}
	glEnd();
	glFlush();
	//  Swap contents of backward and forward frame buffers
	glutSwapBuffers();
}

void createDisparityMap(char** argv) {
	char* windowDisparity = "Disparity Map";
	//-- 1. Read the images
	image = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	Mat imgLeft = imread(argv[1], IMREAD_GRAYSCALE);
	Mat imgRight = imread(argv[2], IMREAD_GRAYSCALE);
	//-- And create the image in which we will save our disparities
	Mat imgDisparity16S = Mat(imgLeft.rows, imgLeft.cols, CV_32S);
	Mat imgDisparity8U = Mat(imgLeft.rows, imgLeft.cols, CV_8UC1);
	disparityMap = Mat(imgLeft.rows, imgLeft.cols, CV_32F);

	if (imgLeft.empty() || imgRight.empty())
	{
		std::cout << " --(!) Error reading images " << std::endl; return;
	}

	//-- 2. Call the constructor for StereoBM
	int ndisparities = 16 * 16;   /**< Range of disparity */
	int SADWindowSize = 21; /**< Size of the block window. Must be odd */

	Ptr<StereoBM> sbm = StereoBM::create(ndisparities, SADWindowSize);

	//-- 3. Calculate the disparity image
	sbm->compute(imgLeft, imgRight, imgDisparity16S);

	//-- Check its extreme values
	double minVal; double maxVal;

	minMaxLoc(imgDisparity16S, &minVal, &maxVal);

	printf("Min disp: %f Max value: %f \n", minVal, maxVal);

	//-- 4. Encoded Disparity
	imgDisparity16S.convertTo(disparityMap, CV_32F, (1.0 / 64.0f));
	//-- 5. Display it as a CV_8UC1 image
	normalize(imgDisparity16S, imgDisparity8U, 0, 255, CV_MINMAX, CV_8UC1);

	minMaxLoc(disparityMap, &minVal, &maxVal);

	printf("Min disp: %f Max value: %f \n", minVal, maxVal);

	namedWindow(windowDisparity, WINDOW_NORMAL);
	imshow(windowDisparity, imgDisparity8U);

	//-- 6. Save the image
	imwrite("SBM_disparity_map.png", imgDisparity8U);
	imgDisparity16S.release();
	imgDisparity8U.release();

	return;
}

void create3DCloud() {
	//-- 6. Create the Q Matrix
	Mat Q = Mat(4, 4, CV_64F);
	pointCloud = Mat(disparityMap.rows, disparityMap.cols, CV_32F);
	Q.at<double>(0, 0) = 1.0;
	Q.at<double>(0, 1) = 0.0;
	Q.at<double>(0, 2) = 0.0;
	Q.at<double>(0, 3) = -(disparityMap.rows>>1); //cx
	Q.at<double>(1, 0) = 0.0;
	Q.at<double>(1, 1) = 1.0;
	Q.at<double>(1, 2) = 0.0;
	Q.at<double>(1, 3) = -(disparityMap.cols>>1);  //cy
	Q.at<double>(2, 0) = 0.0;
	Q.at<double>(2, 1) = 0.0;
	Q.at<double>(2, 2) = 0.0;
	Q.at<double>(2, 3) = 100;  //Focal
	Q.at<double>(3, 0) = 0.0;
	Q.at<double>(3, 1) = 0.0;
	Q.at<double>(3, 2) = 1.0 / 10;    //1.0/BaseLine
	Q.at<double>(3, 3) = -17.0;    //cx - cx'
	reprojectImageTo3D(disparityMap, pointCloud, Q, false, CV_32F);
	//-- 7. release the Q Matrix
	Q.release();
}

void printImage() {
	float min_x = 10000, max_x = -1, min_y=10000, max_y = -1, min_z=10000, max_z = -100;
	std::cout << " -- Print Image " << pointCloud.size() << "->" << image.size() << " -- " << std::endl;
	for (int i = 0; i < pointCloud.rows; i++) {
		for (int j = 0; j < pointCloud.cols >> 4; j++) {
			if (pointCloud.at<Vec3f>(i, j)[2] < 10) {
				if (min_x > pointCloud.at<Vec3f>(i, j)[0]) {
					min_x = pointCloud.at<Vec3f>(i, j)[0];
				}
				if (max_x < pointCloud.at<Vec3f>(i, j)[0]) {
					max_x = pointCloud.at<Vec3f>(i, j)[0];
				}
				if (min_y > pointCloud.at<Vec3f>(i, j)[1]) {
					min_y = pointCloud.at<Vec3f>(i, j)[1];
				}
				if (max_y < pointCloud.at<Vec3f>(i, j)[1]) {
					max_y = pointCloud.at<Vec3f>(i, j)[1];
				}
				if (min_z > pointCloud.at<Vec3f>(i, j)[2]) {
					min_z = pointCloud.at<Vec3f>(i, j)[2];
				}
				if (max_z < pointCloud.at<Vec3f>(i, j)[2]) {
					max_z = pointCloud.at<Vec3f>(i, j)[2];
				}
			}
		}
	}
	std::cout << min_x << "," << max_x << "->" << min_y << "," << max_y << "->" << min_z << "," << max_z << std::endl;
}

void initializeOpenGL(int argc, char** argv) {
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGB);
	centerOnScreen();
	glutInitWindowPosition(window_x, window_y);
	glutInitWindowSize(disparityMap.rows, disparityMap.cols);
	glutCreateWindow("3D disparity image");

	init();
	glutDisplayFunc(renderScene);
	//glutReshapeFunc(reshape);
	//glutIdleFunc(renderScene);
	glutMainLoop();
}

//-------------------------------------------------------------------------
//  Set OpenGL program initial state.
//-------------------------------------------------------------------------
void init()
{
	//  Set the frame buffer clear color to black. 
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(45, (float)image.rows / (float)image.cols, 0.1f, 100.0f);
	gluLookAt(0, 0, 0, 40, 50, -5.8, 0, 0, 0);
	//glPointSize(1.0);
}

//-------------------------------------------------------------------------
//  This function sets the window x and y coordinates
//  such that the window becomes centered
//-------------------------------------------------------------------------
void centerOnScreen()
{
	window_x = (glutGet(GLUT_SCREEN_WIDTH) - disparityMap.rows) / 2;
	window_y = (glutGet(GLUT_SCREEN_HEIGHT) - disparityMap.cols) / 2;
}
/**
* @function readme
*/
void readme()
{
	std::cout << " Usage: ./SBMSample <imgLeft> <imgRight>" << std::endl;
}
/**
 * @function main
 * @brief Main function
 */
int main(int argc, char** argv)
{
	if (argc != 3)
	{
		readme(); return -1;
	}

	createDisparityMap(argv);
	create3DCloud();
	//printImage();
	initializeOpenGL(argc, argv);
	waitKey(0);

	return 0;
}