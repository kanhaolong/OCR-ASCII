/*
 *  basicOCR.c
 *  
 *
 *  Created by damiles on 18/11/08.
 *  Copyright 2008 Damiles. GPL License
 *
 */
#ifdef _CH_
#pragma package <opencv>
#endif
#ifndef _EiC
#include "cv.h"
#include "highgui.h"
#include "ml.h"
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include "cxcore.h"
#include <list>  
#include <algorithm> 
#endif

#include "preprocessing.h"
#include "basicOCR.h"

typedef std::list<CvRect> LISTRECT; 
typedef std::list<CvRect>::iterator ILR;

basicOCR::basicOCR()//构造函数
{

	//initial
	sprintf(file_path , "OCR/");
	train_samples = 36;//训练样本，总共100个，50个训练，50个测试
	classes= 127;//暂时识别十个数字

	size=128;//


	trainData = cvCreateMat(train_samples*classes, size*size, CV_32FC1);//训练数据的矩阵
	trainClasses = cvCreateMat(train_samples*classes, 1, CV_32FC1);

	//Get data (get images and process it)
	getData();

	//train	
	train();
	//Test	
	test();

	printf(" ------------------------------------------------------------------------\n");
	printf("|\t识别结果\t|\t 测试精度\t|\t  准确率\t|\n");
	printf(" ------------------------------------------------------------------------\n");


}



void basicOCR::getData()
{
	IplImage* src_image;
	IplImage prs_image;
	CvMat row,data;
	char file[255];
	int i,j;
	for(i =32; i<classes; i++)
	{
		for( j = 1; j< train_samples; j++)
		{
			
			//加载pbm格式图像，作为训练
			if(j<10)
				sprintf(file,"%s%d/0%d.pbm",file_path, i, j);
			else
				sprintf(file,"%s%d/%d.pbm",file_path, i , j);
			src_image = cvLoadImage(file,0);
			if(!src_image)
			{
				printf("Error: Cant load image %s\n", file);
				//exit(-1);
			}
			//process file
			prs_image = preprocessing(src_image, size, size);
			
			//Set class label
			cvGetRow(trainClasses, &row, i*train_samples + j);
			cvSet(&row, cvRealScalar(i));
			//Set data 
			cvGetRow(trainData, &row, i*train_samples + j);

			IplImage* img = cvCreateImage( cvSize( size, size ), IPL_DEPTH_32F, 1 );
			//convert 8 bits image to 32 float image
			cvConvertScale(&prs_image, img, 0.0039215, 0);

			cvGetSubRect(img, &data, cvRect(0,0, size,size));
			
			CvMat row_header, *row1;
			//convert data matrix sizexsize to vecor
			row1 = cvReshape( &data, &row_header, 0, 1 );
			cvCopy(row1, &row, NULL);
		}
	}
}

void basicOCR::train()
{
	knn=new CvKNearest( trainData, trainClasses, 0, false, K );
}

float basicOCR::classify(IplImage* img, int showResult)//第二个参数主要用来控制是测试训练样本还是手写识别
{
	IplImage prs_image;
	CvMat data;
	CvMat* nearest=cvCreateMat(1,K,CV_32FC1);
	float result;
	//处理输入的图像
	prs_image = preprocessing(img, size, size);
	
	//Set data 
	IplImage* img32 = cvCreateImage( cvSize( size, size ), IPL_DEPTH_32F, 1 );
	cvConvertScale(&prs_image, img32, 0.0039215, 0);
	cvGetSubRect(img32, &data, cvRect(0,0, size,size));
	CvMat row_header, *row1;
	row1 = cvReshape( &data, &row_header, 0, 1 );

	result=knn->find_nearest(row1,K,0,0,nearest,0);
	
	int accuracy=0;
	for(int i=0;i<K;i++)
	{
		if( (nearest->data.fl[i]) == result)
                    accuracy++;
	}
	float pre=100*((float)accuracy/(float)K);
	char out =  result;
	if(showResult==1)
	{
		printf("|\t    %c    \t| \t    %.2f%%  \t| \t %d of %d \t| \n",out,pre,accuracy,K);
		printf(" ------------------------------------------------------------------------\n");
	}

	return result;

}

void basicOCR::test()
{
	IplImage* src_image;
	IplImage prs_image;
	CvMat row,data;
	char file[255];
	int i,j;
	int error=0;
	int testCount=0;
	for(i =32; i<classes; i++)
	{
		for( j = 35; j< train_samples; j++)//五十个测试样本，计算一下错误率
		{
			
			if(j<10)
				sprintf(file,"%s%d/0%d.pbm",file_path, i, j);
			else
				sprintf(file,"%s%d/%d.pbm",file_path, i , j);
			src_image = cvLoadImage(file,0);
			if(!src_image)
			{
				printf("Error: Cant load image %s\n", file);
				exit(-1);
			}
			//process file
			prs_image = preprocessing(src_image, size, size);
			float r=classify(&prs_image,0);
			if((int)r!=i)
				error++;
			
			testCount++;
		}
	}
	float totalerror=100*(float)error/(float)testCount;
	printf("测试系统误识率: %.2f%%\n", totalerror);
	
}

std::string basicOCR::readText(std::string filesrc)
{

	IplImage* imgSrc = cvLoadImage(filesrc.data(),CV_LOAD_IMAGE_COLOR);
	IplImage* img_gray = cvCreateImage(cvGetSize(imgSrc), IPL_DEPTH_8U, 1);
	IplImage* img_check = cvCreateImage(cvGetSize(imgSrc), IPL_DEPTH_8U, 1);
	//cvSmooth(img_gray,img_gray,CV_GAUSSIAN,5,5);
	//cvCopyImage(imgSrc,img_gray);
	cvCvtColor(imgSrc, img_gray, CV_BGR2GRAY);
	cvCvtColor(imgSrc, img_check, CV_BGR2GRAY);
	cvThreshold(img_check, img_check,160, 255,CV_THRESH_BINARY);
	cvThreshold(img_gray, img_gray,160, 255,CV_THRESH_BINARY_INV);// CV_THRESH_BINARY_INV使得背景为黑色，字符为白色，这样找到的最外层才是字符的最外层
	//cvShowImage("ThresholdImg",img_gray);
	CvSeq* contours = NULL;
	CvMemStorage* storage = cvCreateMemStorage(0); 
	int count = cvFindContours(img_gray, storage, &contours,sizeof(CvContour),CV_RETR_EXTERNAL);
	int idx = 0;
	char szName[56] = {0};
	int tempCount=0;
	LISTRECT allrect;
	LISTRECT line;
	double countH = 0;
	double countW = 0;
	//取出所有字符边界，根据X轴排序
	int buu = 0;
	std::string output = "";
	for (CvSeq* c = contours; c != NULL; c = c->h_next)
	{
		bool isInster = false;
		CvRect rc =cvBoundingRect(c,0);
		countH += rc.height;
		countW += rc.width;
		for (ILR i = allrect.begin();i!= allrect.end();++i)
		{

			if(rc.x < i->x)
			{
				allrect.insert(i,rc);
				isInster= true;
				break;
			}
		}
		if (isInster == false)
		{
			allrect.push_back(rc);
		}
	}

	double avgh = countH/allrect.size();
	double avgw = countW/allrect.size();
	for (line.clear();allrect.size() != 0;line.clear())
	{
		//find the highest char
		ILR i = allrect.begin();
		int miny = i->y;
		for (++i;i != allrect.end(); ++i)
		{
			if (miny > i->y)
			{
				miny = i->y;
			}
		}
		//find first char of line
		for (i = allrect.begin();i->y > (miny+avgh)*1.2 ;++i);
		//cvDrawRect(imgSrc, cvPoint(i->x, i->y), cvPoint(i->x + i->width, i->y + i->height), CV_RGB(255, 0, 0));
		double countY = i->y + avgh;
		int lastXb = i->x;
		int lastXe = i->x+i->width;
		int lastY = i->y + i->height;
		int countX = 0;
		int countSpace = 0;
		//put first char to line list
		line.push_back(*i);
		i = allrect.erase(i);

		for (;i != allrect.end();)
		{
			//find next char
			if(i->y < lastY || i->y < (countY))
			{
				//cvDrawRect(imgSrc, cvPoint(i->x, i->y), cvPoint(i->x + i->width, i->y + i->height), CV_RGB(255, 0, 0));
				countX += i->x - lastXb;
				countSpace += i->x - lastXe;
				//countY += i->y + i->height;
				lastY = i->y + i->height;
				lastXb = i->x;
				lastXe = i->x+i->width;
				line.push_back(*i);
				i = allrect.erase(i);

			}
			else
			{
				++i;
			}
		}

		for (ILR li = line.begin();li != line.end();)
		{
			ILR lasti = li;
			li++;
			if (li == line.end())
			{
				break;
			}
			//cvDrawRect(imgSrc, cvPoint(li->x, li->y), cvPoint(li->x + li->width, li->y + li->height), CV_RGB(255, 0, 0));

			if (((li->height < avgh/2) || (lasti->height < avgh/2))
				&& ((li->x - lasti->x) < (countX/(line.size()-1)/2) 
				|| (li->x+li->width > lasti->x && li->x+li->width < (lasti->x+lasti->width)) 
				|| (li->x > lasti->x && li->x < (lasti->x+lasti->width))))
			{				
				int x = std::min(li->x,lasti->x);
				int y = std::min(li->y, lasti->y);
				int height = (std::max(li->y+li->height, lasti->y+lasti->height) - y);
				int width = (std::max(li->x+li->width, lasti->x+lasti->width) - x);
				CvRect add = {x,y,width,height};
				*lasti = add;
				li = line.erase(li);
				li--;
				//line.insert(li,add);
			}
		}

		for (ILR ci  = line.begin();ci != line.end();ci++)
		{
			int rate = ((double)ci->width/(double)ci->height) /((double)avgw/(double)avgh*2);
			rate++;
			if (rate > 1)
			{
				int x = ci->x;
				int y = ci->y;
				int h = ci->height;
				int w = ci->width;
				ci = line.erase(ci);
				for(int a = rate;a > 0 ;a--)
				{
					CvRect add = {x+w/rate*(rate-a),y,w/rate,h};
					if (ci == line.end())
					{
						line.push_back(add);
						ci = line.end();
					}
					else
					{
						line.insert(ci,add);
					}
				}
				ci--;
			}
		}
		int c = 0;
		i = line.begin();
		IplImage* imgNo = cvCreateImage(cvSize(i->width, i->height), IPL_DEPTH_8U, 1); 
		cvSetImageROI(img_check, *i);
		cvCopyImage(img_check, imgNo);
		cvResetImageROI(img_check); 
		char temp;
		temp = classify(imgNo,0);
		printf("%c",temp);
		c = c*10 + classify(imgNo,0);
		output += temp;
		//cvDrawRect(imgSrc, cvPoint(i->x, i->y), cvPoint(i->x + i->width, i->y + i->height), CV_RGB(255, 0, 0));
		int lastX = i->x+i->width;
		lastY = i->y;

		for (i++;i != line.end(); ++i)
		{
			buu++;
			if (i->x - lastX > (countSpace / (line.size() - 1)))
			{
							/*
				//cvDrawRect(imgSrc, cvPoint(lastX, lastY), cvPoint(lastX + avgw, lastY + avgh), CV_RGB(255, 0, 0));
				CvRect space = {lastX, lastY, avgw, avgh};
				imgNo = cvCreateImage(cvSize(avgw, avgh), IPL_DEPTH_8U, 1); 
				cvSetImageROI(img_check, space);
				cvCopyImage(img_check, imgNo);
				cvResetImageROI(img_check);
				temp = classify(imgNo,0);
				c = c*10 + classify(imgNo,0);
				output += temp;
				imgNo = cvCreateImage(cvSize(i->width, i->height), IPL_DEPTH_8U, 1); 
				cvSetImageROI(img_check, *i);
				cvCopyImage(img_check, imgNo);
				cvResetImageROI(img_check); 
				temp = classify(imgNo,0);
				*/
				printf(" ",temp);
			}
			lastX = i->x+i->width;
			lastY = i->y; 
			imgNo = cvCreateImage(cvSize(i->width, i->height), IPL_DEPTH_8U, 1); 
			cvSetImageROI(img_check, *i);
			cvCopyImage(img_check, imgNo);
			cvResetImageROI(img_check); 
			temp = classify(imgNo,0);
			printf("%c",temp);
			c = c*10 + classify(imgNo,0);
			output += temp;
			//char szName[56] = {0};
			//sprintf(szName, "%d", idx++); 
			//cvNamedWindow(szName); 
			//cvShowImage(szName, imgNo); 
			//cvDrawRect(imgSrc, cvPoint(i->x, i->y), cvPoint(i->x + i->width, i->y + i->height), CV_RGB(255, 0, 0));
		}
		output += "\n";
		printf("\n");
		//printf("%d\n",c);
	}
	printf("轮廓个数：%d",++buu);
	/*
	cvNamedWindow("src"); 
	cvShowImage("src", imgSrc);
	cvWaitKey(0); 
	cvReleaseMemStorage(&storage); 
	cvReleaseImage(&imgSrc); 
	cvReleaseImage(&img_gray); 
	cvDestroyAllWindows(); 
	*/
	return output;                    
}

