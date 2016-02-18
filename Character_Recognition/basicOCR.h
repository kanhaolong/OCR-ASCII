/*
 *  basicOCR.h
 *  
 *
 *  Created by damiles on 18/11/08.
 *  Copyright 2008 __MyCompanyName__. All rights reserved.
 *
 */
#ifdef _CH_
#pragma package <opencv>
#endif

#ifndef _EiC
#include <cv.h>
#include <highgui.h>
#include <ml.h>
#include <stdio.h>
#include <ctype.h>
#include <string>
#endif

class basicOCR
{
	public:
		float classify(IplImage* img,int showResult);
		basicOCR ();
		void test();
		std::string basicOCR::readText(std::string filesrc);
	private:
		char file_path[255];
		int train_samples;
		int classes;
		CvMat* trainData;
		CvMat* trainClasses;
		int size;
		static const int K=5;//����ھӸ���
		CvKNearest *knn;
		void getData();
		void train();
};
