//* Projet Azzedine Dergaoui Paris 8
 #include "opencv2/objdetect/objdetect.hpp"
 #include "opencv2/highgui/highgui.hpp"
 #include "opencv2/imgproc/imgproc.hpp"
 #include "opencv2/core/core.hpp"
 #include "opencv2/contrib/contrib.hpp"
 #include <fstream>
 #include <sstream>
 #include <iostream>
 #include <stdio.h>
//g++ visage1.cpp -o visage1 `pkg-config --cflags opencv` `pkg-config --libs opencv`
 using namespace std;
 using namespace cv;

 /** Fonction Principale */
 void detectAndDisplay( Mat frame );

 /** Variables Globales */
 String face_cascade_name = "/home/dergazz/opencv-2.4.11/data/haarcascades/haarcascade_frontalface_alt.xml";
 String eyes_cascade_name = "/home/dergazz/opencv-2.4.11/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml";
 String mouth_cascade_name ="/home/dergazz/opencv-2.4.11/data/haarcascades/haarcascade_mcs_mouth.xml";
 String nose_cascade_name ="/home/dergazz/opencv-2.4.11/data/haarcascades/haarcascade_mcs_nose.xml";

 CascadeClassifier face_cascade;
 CascadeClassifier eyes_cascade;
 CascadeClassifier mouth_cascade;
 CascadeClassifier nose_cascade;

 string window_name1 = "Capture - Face detection";
 string window_name2 = "Capture - Eyes detection";
 string window_name3 = "Capture - Mouth detection";
 string window_name4 = "Capture - nose detection";
 string window_name0 = "Capture - Flou detection";


 RNG rng(12345);

 /** @function main */
 int main( int argc, const char** argv )
 {
   CvCapture* capture= 0x0;
   Mat frame;

   //-- 1. Chargement des cascades
   
   if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
   if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
   if( !mouth_cascade.load( mouth_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
   if( !nose_cascade.load( nose_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };


   //-- 2. Lire la video stream
   capture = cvCaptureFromCAM( -1 );
   if( capture )
   {
     while( true )
     {
   frame = cvQueryFrame( capture );

   //-- 3. Appliquer le classificateur au frame "cadre"
       if( !frame.empty() )
       { detectAndDisplay( frame ); }
       else
       { printf(" --(!) No captured frame -- Break!"); break; }

       int c = waitKey(10);
       if( (char)c == 'c' ) { break; }
      }
   }
   return 0;
 }

/** @fonction de détection et affichage  */
void detectAndDisplay( Mat frame )
{
  std::vector<Rect> faces;
  Mat frame_gray;
  cvtColor( frame, frame_gray, CV_BGR2GRAY );
  equalizeHist( frame_gray, frame_gray );
  
/**@Afficher la vidéo en flou */
Mat framedest(frame.size(),frame.type());
blur(frame,framedest,Size(1,50),Point(-0,-1));
  equalizeHist( frame_gray, frame_gray );
  imshow(window_name0, framedest);

  //-- Detect faces
  face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(10, 10) );

  for( size_t i = 0; i < faces.size(); i++ )
  {
    Point pt1(faces[i].x+faces[i].width, faces[i].y+faces[i].height);
    Point pt2(faces[i].x,faces[i].y);


  Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );

  ellipse( frame, center, Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar( 2, 0, 25 ), 4, 8, 0 );
  rectangle(frame, pt1, pt2, cvScalar(0,255,0), 4, 8, 0);
  

               std::cout<<"PT1 "<<pt1.x<<"   "<<pt1.y<<std::endl;
               std::cout<<"PT2 "<<pt2.x<<"   "<<pt2.y<<std::endl;

int pt1_x = std::max(faces[i].tl().x , 0);
int pt1_y = std::max(faces[i].tl().y , 0);
        string box_text = format("Pt1= %d %d",pt1_x, pt1_y);
                putText(frame, box_text, Point(pt1_x, pt1_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);

int pt2_x = std::max(faces[i].tl().x + faces[i].width  - 10, 10);
int pt2_y = std::max(faces[i].tl().y + faces[i].height - 10, 10);
        string box_text2 = format("Pt2= %d %d",pt2_x, pt2_y);
                putText(frame, box_text2, Point(pt2_x, pt2_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(250,0,0), 2.0);

// Afficher un Message sur la vidéo

string text = "Azzedine Dergaoui LP3 OCI";
int fontFace = FONT_HERSHEY_PLAIN;
double fontScale = 2;
int thickness = 2;  
cv::Point textOrg(50, 100);
cv::putText(frame, text, textOrg, fontFace, fontScale, cvScalar(10,15,20), thickness,2);

imshow( window_name1, frame );

//-----------------------------------face, detect eyes----------------------------------------------
    //déclaration 
    Mat faceROI = frame_gray( faces[i] );
    std::vector<Rect> eyes; 
    eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(30, 30) );

    for( size_t j = 0; j < eyes.size(); j++ )
     {
     Point center( faces[i].x + eyes[j].x + eyes[j].width*0.5, faces[i].y + eyes[j].y + eyes[j].height*0.5 );
     int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
	circle( frame, center, radius, Scalar( 12,118,55 ), 4, 2, 0 );
	//rectangle(frame, pt1, pt2, cvScalar(10,255,90), 4, 8, 0);

imshow( window_name2, frame );
     }
/*.................................. ***** FIN detect eyes ***......................................
..................................................................................................*/


//-------------------------------*** face, detect mouth ***-----------------------------------------
vector<Rect> mouth;
   
   mouth_cascade.detectMultiScale( faceROI, mouth, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(30, 30) );
for( size_t u = 0; u < mouth.size(); u++ )
     {
       Point center( faces[i].x + mouth[u].x + mouth[u].width*0.5, faces[i].y + mouth[u].y + mouth[u].height*0.5 );
       int radius = cvRound( (mouth[u].width + mouth[u].height)*0.25 );
       circle( frame, center, radius, Scalar(139,117,0), 4, 8, 0 );
imshow( window_name3, frame );
     }
/*.................................. ***** FIN detect mouth ***.....................................
..................................................................................................*/


//-------------------------------*** face, detect nose ***-----------------------------------------
 vector<Rect> nose;
   
   nose_cascade.detectMultiScale( faceROI, nose, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(30, 30) );
for( size_t e = 0; e < nose.size(); e++ )
     {
       Point center( faces[i].x + nose[e].x + nose[e].width*0.5, faces[i].y + nose[e].y + nose[e].height*0.5 );
       int radius = cvRound( (nose[e].width + nose[e].height)*0.2 );
	circle( frame, center, radius, Scalar(0,0,250), -1, 8, 0 );
        
	
imshow( window_name4, frame );
     }
/*.................................. ***** FIN detect nose ***.....................................
..................................................................................................*/
  }
  
 }
