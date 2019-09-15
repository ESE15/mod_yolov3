#include "MyClient.h"
#include "network.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "image.h"
#include "demo.h"
#include <sys/time.h>

////JH EDIT
#include <unistd.h>
#include <string.h>
#include <time.h>
#include <sys/types.h>
#include "sys/socket.h"
#include "netinet/in.h"
#include <stdbool.h>
#define BUF_LEN 512
#define STX 0x02
#define ETX 0x03
#define ACK 0x04
#define START 0x05
#define END 0x06
#define RECT 0x07
#define MSG 0x08
#define STREAMING 1

////

#define DEMO 1


void *myClient;
//////#ifdef OPENCV
void imgCallback2(void *cv_img)
{
    printf("hello?\n");
    //resize(cv_img, cv_img, Size(640, 480), 0, 0, CV_INTER_LINEAR);
    //imshow("playing", cv_img);
}
static char **demo_names;
static image **demo_alphabet;
static int demo_classes;

// static image backUpImage;
static network *net;
static image buff[3];
static image buff_letter[3];
static int buff_index = 0;
static void *cap;
static float fps = 0;
static float demo_thresh = 0;
static float demo_hier = .5;
static int running = 0;

static int demo_frame = 3;
static int demo_index = 0;
static float **predictions;
static float *avg;
static int demo_done = 0;
static int demo_total = 0;
double demo_time;

////JH EDIT
int parseMsg(char *buffer, int msg_size, char* message);
void sendMsg(int client_fd, char *msg,char type);
void sendChar(int client_fd, char msg);

char buffer[BUF_LEN];
char msg[BUF_LEN - 3];
struct sockaddr_in server_addr, client_addr;
char temp[20];
int server_fd, client_fd;
// each number of socket
int len, msg_size;
struct timeval tval;
bool reuseflag = true;
char rectInfo[1000];
char finalRectInfo[1000];
bool startFlag=false;


double startTime;
int usedFrame = 0;
int adaptFrame = 0;
int ccnt = 0;
int isFetching = 0;
int isGetting = 0;
image loopimage;
pthread_mutex_t mutex_lock;
////

detection *get_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, int *num);

int size_network(network *net)
{
    int i;
    int count = 0;
    for (i = 0; i < net->n; ++i)
    {
        layer l = net->layers[i];
        if (l.type == YOLO || l.type == REGION || l.type == DETECTION)
        {
            count += l.outputs;
        }
    }
    return count;
}

void remember_network(network *net)
{
    int i;
    int count = 0;
    for (i = 0; i < net->n; ++i)
    {
        layer l = net->layers[i];
        if (l.type == YOLO || l.type == REGION || l.type == DETECTION)
        {
            memcpy(predictions[demo_index] + count, net->layers[i].output, sizeof(float) * l.outputs);
            count += l.outputs;
        }
    }
}

detection *avg_predictions(network *net, int *nboxes)
{
    int i, j;
    int count = 0;
    fill_cpu(demo_total, 0, avg, 1);
    for (j = 0; j < demo_frame; ++j)
    {
        axpy_cpu(demo_total, 1. / demo_frame, predictions[j], 1, avg, 1);
    }
    for (i = 0; i < net->n; ++i)
    {
        layer l = net->layers[i];
        if (l.type == YOLO || l.type == REGION || l.type == DETECTION)
        {
            memcpy(l.output, avg + count, sizeof(float) * l.outputs);
            count += l.outputs;
        }
    }
    detection *dets = get_network_boxes(net, buff[0].w, buff[0].h, demo_thresh, demo_hier, 0, 1, nboxes);
    return dets;
}

void *detect_in_thread(void *ptr)
{
    running = 1;
    float nms = .4;

    layer l = net->layers[net->n - 1];
    float *X = buff_letter[(buff_index + 2) % 3].data;
    network_predict(net, X);

    /*
       if(l.type == DETECTION){
       get_detection_boxes(l, 1, 1, demo_thresh, probs, boxes, 0);
       } else */
    remember_network(net);
    detection *dets = 0;
    int nboxes = 0;
    dets = avg_predictions(net, &nboxes);

    /*
       int i,j;
       box zero = {0};
       int classes = l.classes;
       for(i = 0; i < demo_detections; ++i){
       avg[i].objectness = 0;
       avg[i].bbox = zero;
       memset(avg[i].prob, 0, classes*sizeof(float));
       for(j = 0; j < demo_frame; ++j){
       axpy_cpu(classes, 1./demo_frame, dets[j][i].prob, 1, avg[i].prob, 1);
       avg[i].objectness += dets[j][i].objectness * 1./demo_frame;
       avg[i].bbox.x += dets[j][i].bbox.x * 1./demo_frame;
       avg[i].bbox.y += dets[j][i].bbox.y * 1./demo_frame;
       avg[i].bbox.w += dets[j][i].bbox.w * 1./demo_frame;
       avg[i].bbox.h += dets[j][i].bbox.h * 1./demo_frame;
       }
    //copy_cpu(classes, dets[0][i].prob, 1, avg[i].prob, 1);
    //avg[i].objectness = dets[0][i].objectness;
    }
     */

    if (nms > 0)
        do_nms_obj(dets, nboxes, l.classes, nms);

    //printf("\033[2J");
    //printf("\033[1;1H");

    // printf("\nFPS:%.1f\n", fps);
    // printf("Elapsed Time: %.2f\n", (what_time_is_it_now() - startTime) * 1000);
    // printf("Elapsed Frame:%d\n", usedFrame);

    //printf("Objects:\n\n");
    image display = buff[(buff_index + 2) % 3];
    memset(finalRectInfo,0,sizeof(finalRectInfo)); 
    draw_detections(display, dets, nboxes, demo_thresh, demo_names, demo_alphabet, demo_classes,rectInfo);
    if(rectInfo[0]!=0){
        //printf("%s\n",rectInfo);
        pthread_mutex_lock(&mutex_lock);
        strcpy(finalRectInfo,rectInfo);
        pthread_mutex_unlock(&mutex_lock);
    }else{
        pthread_mutex_lock(&mutex_lock);
        finalRectInfo[0]=0x10;
        pthread_mutex_unlock(&mutex_lock);
    }
    free_detections(dets, nboxes);

    demo_index = (demo_index + 1) % demo_frame;
    running = 0;
    return 0;
}

/*void *fetch_in_thread(void *ptr)
{
    free_image(buff[buff_index]);
    ////buff[buff_index] = get_image_from_stream(cap);
    buff[buff_index] = get_image_from_stream2(cap,(what_time_is_it_now()-startTime)*1000);
    if(buff[buff_index].data == 0) {
        demo_done = 1;
        return 0;
    }
    letterbox_image_into(buff[buff_index], net->w, net->h, buff_letter[buff_index]);
    return 0;
}*/
void *fetch_in_thread(void *ptr)
{
    free_image(buff[buff_index]);
    //////pthread_mutex_lock(&mutex_lock);
    //buff[buff_index] = get_image_from_stream3(cap,&mutex_lock);
    //buff[buff_index] = MyClient_ReceiveStreamForYolo(myClient);
    buff[buff_index] = STREAMING ? MyClient_ReceiveStreamForYolo(myClient) : get_image_from_stream3(cap, &mutex_lock);
    //////pthread_mutex_unlock(&mutex_lock);
    //isGetting=1;
    ////buff[buff_index] = loopimage;//get_image_from_stream2(cap,(what_time_is_it_now()-startTime)*1000);

    if (buff[buff_index].data == 0)
    {
        demo_done = 1;
        return 0;
    }
    letterbox_image_into(buff[buff_index], net->w, net->h, buff_letter[buff_index]); //isGetting=0;
    return 0;
}
void *socket_in_thread(void *ptr){
    ///////// socket
    tval.tv_sec=3;
    tval.tv_usec=0;


    if((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == -1)
    {// 소켓 생성
        printf("Server : Can't open stream socket\n");
        exit(0);
    }
    memset(&server_addr, 0x00, sizeof(server_addr));
    //server_Addr 을 NULL로 초기화
 
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = htonl(INADDR_ANY);
    server_addr.sin_port = htons(6666);
    //server_addr 셋팅
 
    if(bind(server_fd, (struct sockaddr *)&server_addr, sizeof(server_addr)) <0)
    {//bind() 호출
        printf("Server : Can't bind local address.\n");
        exit(0);
    }
 
    if(listen(server_fd, 5) < 0)
    {//소켓을 수동 대기모드로 설정
        printf("Server : Can't listening connect.\n");
        exit(0);
    }
 
    

    printf("Server : wating connection request.\n");
    len = sizeof(client_addr);
    client_fd = accept(server_fd, (struct sockaddr *)&client_addr, &len);
    setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, (char*) & tval, sizeof(struct timeval));
    setsockopt(client_fd,SOL_SOCKET,SO_REUSEADDR,(char*)&reuseflag,sizeof(reuseflag));
    if(client_fd < 0)
    {
        printf("Server: accept failed.\n");
        exit(0);
    }
    inet_ntop(AF_INET, &client_addr.sin_addr.s_addr, temp, sizeof(temp));
    printf("Server : %s client connected.\n", temp);
    // pthread_mutex_lock(&mutex_lock);
    // startFlag=true;
    // pthread_mutex_unlock(&mutex_lock);
    while(1)
    {
        memset(buffer, 0x00, sizeof(buffer));
        memset(msg,0x00,sizeof(msg));
    
        msg_size = read(client_fd, buffer, BUF_LEN);
        if(msg_size<0){
            printf("아무것도 안옴\n");
        }
        else if(msg_size==0){
            // pthread_mutex_lock(&mutex_lock);
            // startFlag=false;
            // pthread_mutex_unlock(&mutex_lock);
            memset(rectInfo,0,sizeof(rectInfo));
            memset(finalRectInfo,0,sizeof(finalRectInfo));
            finalRectInfo[0]=0x10;
            printf("Server: Connection closed. retrying to connect\n");
            client_fd = accept(server_fd, (struct sockaddr *)&client_addr, &len);
            inet_ntop(AF_INET, &client_addr.sin_addr.s_addr, temp, sizeof(temp));
            printf("Server : %s client connected.\n", temp);
            pthread_mutex_lock(&mutex_lock);
            startFlag=true;
            pthread_mutex_unlock(&mutex_lock);
            setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, (char*) & tval, sizeof(struct timeval));
            setsockopt(client_fd,SOL_SOCKET,SO_REUSEADDR,(char*)&reuseflag,sizeof(reuseflag));
        }
        if(-1 ==parseMsg(buffer, msg_size, msg)){
            printf("제대로 못받음");
        }else{
            //printf(msg);printf("\n");
            if(msg[0]==START){
                sendChar(client_fd,ACK);
            }
            else if(msg[0]==END){
                sendChar(client_fd,ACK);
            }
            if(msg[0]==RECT){
                pthread_mutex_lock(&mutex_lock);
                sendMsg(client_fd,finalRectInfo,RECT);
                pthread_mutex_unlock(&mutex_lock);
            }
            //sendMsg(client_fd, msg);
        }
        
        //write(client_fd, buffer, msg_size);
        //close(client_fd);
    }
    close(server_fd);
    ///////////////////////////////////////////
}
void *fetch_loop_thread(void *ptr)
{
    double frate = getFPS(cap);
    while (1)
    {

        if (frate == 60)
        { // 60fps video
            //////pthread_mutex_lock(&mutex_lock);
            //usleep(12000);
            if ((what_time_is_it_now() - startTime) * 60 > usedFrame + adaptFrame)
            {
                adaptFrame++;
                free_image(loopimage);
                loopimage = get_image_from_stream3(cap, &mutex_lock);
            }
            //frameRateMod(cap,(what_time_is_it_now()-startTime)*1000,&mutex_lock);
            //////pthread_mutex_unlock(&mutex_lock);
        }
        else if (frate == 30)
        { //30fps webcam or video. probably webcam. not maybe
            free_image(loopimage);
            loopimage = get_image_from_stream3(cap, &mutex_lock);
        }
    }
}
void *display_in_thread(void *ptr)
{
    int c = show_image(buff[(buff_index + 1) % 3], "Demo", 1);
    if (c != -1)
        c = c % 256;
    if (c == 27)
    {
        demo_done = 1;
        return 0;
    }
    else if (c == 82)
    {
        demo_thresh += .02;
    }
    else if (c == 84)
    {
        demo_thresh -= .02;
        if (demo_thresh <= .02)
            demo_thresh = .02;
    }
    else if (c == 83)
    {
        demo_hier += .02;
    }
    else if (c == 81)
    {
        demo_hier -= .02;
        if (demo_hier <= .0)
            demo_hier = .0;
    }
    return 0;
}

void *display_loop(void *ptr)
{
    while (1)
    {
        display_in_thread(0);
    }
}

void *detect_loop(void *ptr)
{
    while (1)
    {
        detect_in_thread(0);
    }
}
void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int delay, char *prefix, int avg_frames, float hier, int w, int h, int frames, int fullscreen)
{
    //demo_frame = avg_frames;
    image **alphabet = load_alphabet();
    demo_names = names;
    demo_alphabet = alphabet;
    demo_classes = classes;
    demo_thresh = thresh;
    demo_hier = hier;
    printf("Demo\n");
    net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    pthread_t detect_thread;
    pthread_t fetch_thread;
    pthread_t socket_thread;
    //void* myClient;
    int resultForClient;
    srand(2222222);
    int i;

    
    memset(finalRectInfo,0x00,sizeof(finalRectInfo));
    finalRectInfo[0]=0x10;

    demo_total = size_network(net);
    predictions = (float **)calloc(demo_frame, sizeof(float *));
    for (i = 0; i < demo_frame; ++i)
    {
        predictions[i] = (float *)calloc(demo_total, sizeof(float));
    }
    avg = (float *)calloc(demo_total, sizeof(float));

    if (filename)
    {
        printf("video file: %s\n", filename);
        ////cap = open_video_stream(filename, 0, 0, 0, 0);
        cap = open_video_stream(filename, 0, 0, 0, 0);
    }
    else
    {
        cap = open_video_stream(0, cam_index, w, h, frames);
    }
    if (!cap)
        error("Couldn't connect to webcam.\n");
    pthread_mutex_init(&mutex_lock, NULL);
    buff[0] = get_image_from_stream(cap);
    buff[1] = copy_image(buff[0]);
    buff[2] = copy_image(buff[0]);
    buff_letter[0] = letterbox_image(buff[0], net->w, net->h);
    buff_letter[1] = letterbox_image(buff[0], net->w, net->h);
    buff_letter[2] = letterbox_image(buff[0], net->w, net->h);

    int count = 0;
    if (!prefix)
    {
        make_window("Demo", 640, 480, fullscreen);
    }
    if(pthread_create(&socket_thread, 0, socket_in_thread, 0)) error("Thread creation failed");
    if (STREAMING)
    {
        myClient = MyClient_getInstance();
        resultForClient = MyClient_Initialize(myClient);
        //MyClient_setCallback(myClient);
        //MyClient_startReceive(myClient);
    }
    demo_time = what_time_is_it_now();
    startTime = demo_time;
    //if(pthread_create(&fetch_looping, 0, fetch_loop_thread, 0)) error("Thread creation failed");
    
    while (!demo_done)
    {
        // pthread_mutex_lock(&mutex_lock);
        // if(startFlag==false) continue;
        // pthread_mutex_unlock(&mutex_lock);
        buff_index = (buff_index + 1) % 3;
        if (pthread_create(&fetch_thread, 0, fetch_in_thread, 0))
            error("Thread creation failed");
        if (pthread_create(&detect_thread, 0, detect_in_thread, 0))
            error("Thread creation failed");
        if (!prefix)
        {
            fps = 1. / (what_time_is_it_now() - demo_time);
            demo_time = what_time_is_it_now();
            display_in_thread(0);
        }
        else
        {
            char name[256];
            sprintf(name, "%s_%08d", prefix, count);
            save_image(buff[(buff_index + 1) % 3], name);
        }
        pthread_join(fetch_thread, 0);
        pthread_join(detect_thread, 0);
        ++count;
    }
    pthread_join(socket_thread,0);
}
int parseMsg(char *buffer, int msg_size, char* message){
    char i=0;
    if(buffer[0]==STX){
        
        for(;i<buffer[1];i++){
            message[i]=buffer[2+i];
        }
        if(buffer[2+i]==ETX){
            return 1;
        }else{
            return 0;
        }
    }else{
        return 0;
    }
    return 0;
}
void sendChar(int client_fd, char msg){
    char temp[BUF_LEN];
    memset(temp,0x00,sizeof(temp));
    temp[0]=STX;
    temp[1]=0x01;
    temp[2]=msg;
    temp[strlen(temp)]=ETX;
    write(client_fd, temp, strlen(temp)+1);
}
void sendMsg(int client_fd, char* msg,char type){
    char temp[BUF_LEN];
    memset(temp,0x00,sizeof(temp));
    temp[0]=STX;
    if(type==RECT){
        temp[1]=strlen(msg)+2;
        temp[2]=RECT;
        strcat(temp,msg);
        temp[strlen(temp)]=ETX;
    }else if (type==MSG){
        temp[1]=strlen(msg);
        strcat(temp,msg);
        temp[strlen(temp)]=ETX;
    }
    
    //write(client_fd, temp, sizeof(temp));
    write(client_fd, temp, strlen(temp)+1);
}
/*
   void demo_compare(char *cfg1, char *weight1, char *cfg2, char *weight2, float thresh, int cam_index, const char *filename, char **names, int classes, int delay, char *prefix, int avg_frames, float hier, int w, int h, int frames, int fullscreen)
   {
   demo_frame = avg_frames;
   predictions = calloc(demo_frame, sizeof(float*));
   image **alphabet = load_alphabet();
   demo_names = names;
   demo_alphabet = alphabet;
   demo_classes = classes;
   demo_thresh = thresh;
   demo_hier = hier;
   printf("Demo\n");
   net = load_network(cfg1, weight1, 0);
   set_batch_network(net, 1);
   pthread_t detect_thread;
   pthread_t fetch_thread;

   srand(2222222);

   if(filename){
   printf("video file: %s\n", filename);
   cap = cvCaptureFromFile(filename);
   }else{
   cap = cvCaptureFromCAM(cam_index);

   if(w){
   cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_WIDTH, w);
   }
   if(h){
   cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_HEIGHT, h);
   }
   if(frames){
   cvSetCaptureProperty(cap, CV_CAP_PROP_FPS, frames);
   }
   }

   if(!cap) error("Couldn't connect to webcam.\n");

   layer l = net->layers[net->n-1];
   demo_detections = l.n*l.w*l.h;
   int j;

   avg = (float *) calloc(l.outputs, sizeof(float));
   for(j = 0; j < demo_frame; ++j) predictions[j] = (float *) calloc(l.outputs, sizeof(float));

   boxes = (box *)calloc(l.w*l.h*l.n, sizeof(box));
   probs = (float **)calloc(l.w*l.h*l.n, sizeof(float *));
   for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float *)calloc(l.classes+1, sizeof(float));

   buff[0] = get_image_from_stream(cap);
   buff[1] = copy_image(buff[0]);
   buff[2] = copy_image(buff[0]);
   buff_letter[0] = letterbox_image(buff[0], net->w, net->h);
   buff_letter[1] = letterbox_image(buff[0], net->w, net->h);
   buff_letter[2] = letterbox_image(buff[0], net->w, net->h);
   ipl = cvCreateImage(cvSize(buff[0].w,buff[0].h), IPL_DEPTH_8U, buff[0].c);

   int count = 0;
   if(!prefix){
   cvNamedWindow("Demo", CV_WINDOW_NORMAL); 
   if(fullscreen){
   cvSetWindowProperty("Demo", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
   } else {
   cvMoveWindow("Demo", 0, 0);
   cvResizeWindow("Demo", 1352, 1013);
   }
   }

   demo_time = what_time_is_it_now();

   while(!demo_done){
buff_index = (buff_index + 1) %3;
if(pthread_create(&fetch_thread, 0, fetch_in_thread, 0)) error("Thread creation failed");
if(pthread_create(&detect_thread, 0, detect_in_thread, 0)) error("Thread creation failed");
if(!prefix){
    fps = 1./(what_time_is_it_now() - demo_time);
    demo_time = what_time_is_it_now();
    display_in_thread(0);
}else{
    char name[256];
    sprintf(name, "%s_%08d", prefix, count);
    save_image(buff[(buff_index + 1)%3], name);
}
pthread_join(fetch_thread, 0);
pthread_join(detect_thread, 0);
++count;
}
}
*/
//////#else
//////void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int delay, char *prefix, int avg, float hier, int w, int h, int frames, int fullscreen)
//////{
//////fprintf(stderr, "Demo needs OpenCV for webcam images.\n");
//////}
//////#endif