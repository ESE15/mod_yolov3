#ifndef _MY_STREAMER_H_
#define _MY_STREAMER_H_
#endif

#define STREAM_FPS 30
#define BUFFSIZEOFRTP 30000000
#define STREAM_PIX_FMT  AV_PIX_FMT_YUV420P

#include "image.h"
#include "stdlib.h"
#ifdef __cplusplus
extern "C" {
#endif
// ffmpeg
////extern "C"{
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavformat/avio.h>
#include <libswscale/swscale.h>
#include <libavutil/time.h>
#include <libavdevice/avdevice.h>
////}
#ifdef __cplusplus
}
#endif

#define CODEC_FLAG_GLOBAL_HEADER (1 << 22)

#ifdef __cplusplus

#include <string>
#include <ctime>
#include <mutex>
#include <thread>
//opencv
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp> // Basic OpenCV structures (cv::Mat)
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;
enum MyClientError {
	CANT_ALLOC_FORMAT_CONTEXT = 10,
	CANT_FIND_STREAM_INFO = 11,
	CANT_OPEN_CODEC = 12,
	CANT_DECODE_IMAGE = 13,
	NO_FFMPEG_ERROR = 100
};

class MyClient
{
public:
	MyClient();
	~MyClient();

private:
	MyClientError last_error;
	bool is_initialized;
	// ffmpeg members
	std::string ip;
	int port;

	enum AVCodecID codec_id;
	SwsContext* img_convert_ctx;
	AVFormatContext* context;
	AVCodecContext* ccontext;
	AVPacket packet;
	int video_stream_index;
	AVFrame *src_picture, *dst_picture;
	uint8_t *src_buf, *dst_buf;


	bool is_receiving;
	// receive stream from remote server
	std::mutex mtx_lock;
	std::thread* receiver;
	// stream receiver
	void ReceiveStream();

	// event occur when receive image successfully
	 void(*receiveEvent)( cv::Mat& cv_img);
	//void(*receiveEvent)( void* cv_img);

public:
	bool Initialize(std::string ip = "127.0.0.1", int port = 5004);
	void Deinitialize();
	bool ReceiveImage(cv::Mat& cv_img);
	int GetLastError();
	bool IsInitialized();

	bool StartReceive();
	void EndReceive();
	void SetReceiveEvent(void(*receiveEvent)( cv::Mat& cv_img));
	//void SetReceiveEvent(void(*receiveEvent)( void* cv_img));
};
#else

void* MyClient_getInstance() ;
 
int MyClient_Initialize(void* instance) ;

void MyClient_startReceive(void* instance);
void MyClient_setCallback(void* instance);
//image MyClient_ReceiveStreamForYolo(void* instance);
#endif
