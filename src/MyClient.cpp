#include "MyClient.h"
#include <stdio.h>
#define F (char*)__FILE__
#define L __LINE__



MyClient::MyClient()
	: is_initialized(false), ip("127.0.0.1"), port(5004), codec_id(AV_CODEC_ID_MPEG4),
	img_convert_ctx(NULL), context(NULL), ccontext(NULL),
	src_buf(NULL), src_picture(NULL), dst_buf(NULL), dst_picture(NULL)
{}

MyClient::~MyClient()
{
	if (this->is_initialized)
		Deinitialize();
}

bool MyClient::Initialize(std::string ip, int port)
{
	int ret;
	printf("Initializing Start\n");
	context = avformat_alloc_context();
	ccontext = avcodec_alloc_context3(NULL);
	av_init_packet(&packet);

	/* Initialize libavcodec, and register all codecs and formats. */
	av_register_all();
	avformat_network_init();

	std::string tempUrl("");
	tempUrl.append("rtp://");
	tempUrl.append(ip + ":");
	tempUrl.append(std::to_string(port));
	//tempUrl.append("/kstream");

	/* allocate the media context */
	if (avformat_open_input(&context, tempUrl.c_str(), NULL, NULL) != 0) {
		this->last_error = MyClientError::CANT_ALLOC_FORMAT_CONTEXT;
		return false;
	}
	if (avformat_find_stream_info(context, NULL) < 0) {
		this->last_error = MyClientError::CANT_FIND_STREAM_INFO;
		return false;
	}

	//search video stream
	for (int i = 0; i < context->nb_streams; i++) {
		if (context->streams[i]->codec->codec_type == AVMEDIA_TYPE_VIDEO)
			video_stream_index = i;
	}

	// set codec
	AVCodec *codec = NULL;
	codec = avcodec_find_decoder(context->streams[video_stream_index]->codec->codec_id);
	if (!codec) {
		return false;
	}

	avcodec_get_context_defaults3(ccontext, codec);
	avcodec_copy_context(ccontext, context->streams[video_stream_index]->codec);


	if (avcodec_open2(ccontext, codec, NULL) < 0) {
		this->last_error = MyClientError::CANT_OPEN_CODEC;
		return false;
	}
	//ccontext->width = 640;
	//ccontext->height = 480;
	img_convert_ctx = sws_getContext(ccontext->width, ccontext->height, ccontext->pix_fmt,
		ccontext->width, ccontext->height, AV_PIX_FMT_BGR24,
		SWS_BICUBIC, NULL, NULL, NULL);

	int size = avpicture_get_size(ccontext->pix_fmt, ccontext->width, ccontext->height);
	src_buf = (uint8_t*)(av_malloc(size));
	src_picture = av_frame_alloc();
	int size2 = avpicture_get_size(AV_PIX_FMT_BGR24, ccontext->width, ccontext->height);
	dst_buf = (uint8_t*)(av_malloc(size2));
	dst_picture = av_frame_alloc();
	avpicture_fill((AVPicture *)src_picture, src_buf, ccontext->pix_fmt, ccontext->width, ccontext->height);
	avpicture_fill((AVPicture *)dst_picture, dst_buf, AV_PIX_FMT_BGR24, ccontext->width, ccontext->height);

	this->is_initialized = true;
	printf("Initializing Done\n");
	return true;
}

void MyClient::Deinitialize()
{
	av_free_packet(&packet);
	if (src_buf != NULL)
	{
		av_free(src_buf);
		src_buf = NULL;
	}
	if (dst_buf != NULL)
	{
		av_free(dst_buf);
		dst_buf = NULL;
	}
	if (dst_picture != NULL)
	{
		av_free(dst_picture);
		dst_picture = NULL;
	}
	if (src_picture != NULL)
	{
		av_free(src_picture);
		src_picture = NULL;
	}
	avformat_free_context(context);
	avcodec_free_context(&ccontext);
	av_read_pause(context);

	this->is_initialized = false;
}

bool MyClient::ReceiveImage(cv::Mat& cv_img)
{
	static AVFrame* frame;

	if (av_read_frame(context, &packet) >= 0)
	{
		//printf("RECEIVE :: read frame successfull\n");
		int check = 0;
		int result = avcodec_decode_video2(ccontext, src_picture, &check, &packet);

		if (!check)
		{
			//printf("RECEIVE :: check failed\n");
			this->last_error = MyClientError::CANT_DECODE_IMAGE;
			return false;
		}

		sws_scale(img_convert_ctx, src_picture->data, src_picture->linesize, 0, ccontext->height, dst_picture->data, dst_picture->linesize);
		cv::Mat mat(ccontext->height, ccontext->width, CV_8UC3, dst_picture->data[0], dst_picture->linesize[0]);
		cv_img = mat.clone();
		// printf("RECEIVE :: cloned\n");
		// resize(cv_img, cv_img, Size(640, 480), 0, 0, CV_INTER_LINEAR);
		// imshow("playing", cv_img);
		
		return true;
	}
	else
		return false;
}

int MyClient::GetLastError()
{
	return this->last_error;
}

bool MyClient::IsInitialized()
{
	return this->is_initialized;
}


bool MyClient::StartReceive()
{
	//EndReceive();
	//printf("Receiving Start\n");
	mtx_lock.lock();
	this->is_receiving = true;
	mtx_lock.unlock();
	// this->receiver = new std::thread(&MyClient::ReceiveStream, this);
	//printf("Thread created\n");
	// if (!this->receiver)
	// {
	// 	printf("Thread creation failed\n");
	// 	//this->last_error = 0;
	// 	return false;
	// }
	// this->receiver->join();
	// printf("Thread joined\n");
	MyClient::ReceiveStream();
	return true;
}

void MyClient::EndReceive()
{
	if (this->receiver)
	{
		mtx_lock.lock();
		this->is_receiving = false;
		mtx_lock.unlock();
		// wait until finish
		this->receiver->join();
		delete this->receiver;
	}

	this->receiver = NULL;
}

void MyClient::SetReceiveEvent(void(*receiveEvent)( cv::Mat& cv_img))
{
	printf("Callback has been set\n");
	this->receiveEvent = receiveEvent;
}
// void MyClient::SetReceiveEvent(void(*receiveEvent)( void* cv_img))
// {
// 	printf("Callback has been set\n");
// 	this->receiveEvent = receiveEvent;
// }

void MyClient::ReceiveStream()
{
	cv::Mat cv_img;
	cv::Mat frame_pool[STREAM_FPS];
	int key_flag=0;
	int key;
	int frame_pool_index = 0;
	while (true)
	{
		mtx_lock.lock();
		bool thread_end = this->is_receiving;
		mtx_lock.unlock();
		if(key_flag==0){
		// user finish
		if (!thread_end)
		{
			break;
		}
		// read frame
		if (!this->ReceiveImage(cv_img))
		{
			printf("THREAD:: RECEIVE FAILED\n");
			
			//this->last_error = KStreamR640eceiverError::FFMPEG_ERROR;
			continue;
		}
		// time stamp
		// std::time_t rawtime;
		// struct std::tm* timeinfo;
		// char timebuf[80];

		/*SYSTEMTIME time;
		GetLocalTime(&time);
		sprintf(timebuf, "%04d:%02d:%02d-%02d:%02d:%02d:%03d",
			time.wYear, time.wMonth, time.wDay,
			time.wHour, time.wMinute, time.wSecond, time.wMilliseconds);
		std::string timestr(timebuf);

		std::time(&rawtime);
		timeinfo = std::localtime(&rawtime);
		std::strftime(timebuf, sizeof(timebuf), "%d-%m-%Y %I:%M:%S", timeinfo);
		std::string timestr(timebuf);*/

		//cv::putText(cv_img, timestr, cv::Point(20, 50), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar::all(255), 2);
		// event occur
		if (this->receiveEvent)
		{
			frame_pool[frame_pool_index] = cv_img.clone();
			this->receiveEvent(frame_pool[frame_pool_index]);
			frame_pool_index = (frame_pool_index + 1) % STREAM_FPS;
			//waitKey((1000 / STREAM_FPS)-30 );
			

		}
		}
		key=waitKey(10);
		// if(key==27){
		// 	printf("Done!!\n");
		// 	break;
		// }
		// else if(key==255){}
		// else{
		// 	printf("Flag changed\n");
		// 	key_flag=(key_flag==1)?0:1;
		// }
		if(key>0){
			if(key==27){
				printf("Done!!\n");
				break;
			}
			printf("Flag changed\n");
			key_flag=(key_flag==1)?0:1;
		}
	}
}


extern "C"
{
 void imgCallback(cv::Mat& cv_img) {
	//resize(cv_img, cv_img, Size(640, 480), 0, 0, CV_INTER_LINEAR);
	imshow("playing", cv_img);
}
void* MyClient_getInstance()
{
        MyClient* myClient= new MyClient() ;
        return (void*)myClient ;
}
 
int MyClient_Initialize(void* instance)
{
        MyClient * myClient = (MyClient*)instance ;
        myClient->Initialize("127.0.0.1",5004) ;
 
        return 1 ;
}

void MyClient_startReceive(void* instance){
		MyClient * myClient = (MyClient*)instance ;
		myClient->StartReceive();
}
void MyClient_setCallback(void* instance ){
		MyClient * myClient = (MyClient*)instance ;
		myClient->SetReceiveEvent(imgCallback);
}

// image MyClient_ReceiveStreamForYolo(void* instance)
// {
// 	MyClient * myClient = (MyClient*)instance ;
// 	cv::Mat cv_img;
// 	cv::Mat m;
// 	// read frame
// 	if (!myClient->ReceiveImage(cv_img))
// 	{
// 		printf("THREAD:: RECEIVE FAILED\n");
// 		return make_empty_image(0,0,0);
// 	}
// 	m=cv_img.clone();
// 	if(m.empty()) return make_empty_image(0,0,0);
//     return mat_to_image(m);
// 	// if(m.empty()) return NULL;
// 	// return m;
// }

}