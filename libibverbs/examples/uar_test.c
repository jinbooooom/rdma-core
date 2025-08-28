/*
* BUILD COMMAND:
* gcc -Wall -I/usr/local/ofed/include -O2 -o RDMA_RC_example -L/usr/local/ofed/lib64 -L/usr/local/ofed/lib -
libverbs RDMA_RC_example.c
*
*/
/******************************************************************************
*
* RDMA Aware Networks Programming Example
*
* This code demonstrates how to perform the following operations using the * VPI Verbs API:
*
* Send
* Receive
* RDMA Read
* RDMA Write
*
*****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include <inttypes.h>
#include <endian.h>
#include <byteswap.h>
#include <getopt.h>
#include <pthread.h>
#include <dlfcn.h>
#include "log.h"

#include <sys/time.h>
#include <arpa/inet.h>
#include "../verbs.h"
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>

/* poll CQ timeout in millisec (2 seconds) */
#define MAX_POLL_CQ_TIMEOUT 2000
#define MAX_WR_COUNT 2
#define MSG "SEND operation "
#define RDMAMSGR "RDMA read operation "
#define RDMAMSGW "RDMA write operation"
#define MSG_SIZE 64 //(strlen(MSG) + 1)
#if __BYTE_ORDER == __LITTLE_ENDIAN
static inline uint64_t htonll(uint64_t x) { return bswap_64(x); }
static inline uint64_t ntohll(uint64_t x) { return bswap_64(x); }
#elif __BYTE_ORDER == __BIG_ENDIAN
static inline uint64_t htonll(uint64_t x) { return x; }
static inline uint64_t ntohll(uint64_t x) { return x; }
#else
#error __BYTE_ORDER is neither __LITTLE_ENDIAN nor __BIG_ENDIAN
#endif

#define MAX_DEVX_INFO_COUNT (MAX_WR_COUNT+1)
// request_idx 与 complete_idx 相等，则说明 devx_infos 为空
// 注意：不使用求余操作，直接递增，使用volatile确保多线程可见性
static volatile uint32_t request_idx = 0;
static volatile uint32_t complete_idx = 0;
struct ibv_devx_info devx_infos[MAX_DEVX_INFO_COUNT] = {0};
static volatile int thread_running = 1; // 控制线程运行状态

// GPU模式相关变量
static void *cuda_plugin_handle = NULL;
static int (*init_cuda_func)(void) = NULL;
static int (*cleanup_cuda_func)(void) = NULL;
static int (*convert_host_va_to_gpu_va_func)(void *, size_t, int, void **) = NULL;
static int (*trigger_doorbell_func)(void *, void *) = NULL;
static int (*unregister_host_va_func)(void *) = NULL;

// GPU端指针
static void *gpu_bf = NULL;
static void *gpu_ctrl = NULL;

/* structure of test parameters */
struct config_t
{
	const char *dev_name; /* IB device name */
	char *server_name;	/* server host name */
	u_int32_t tcp_port;   /* server TCP port */
	int ib_port;		  /* local IB port to work with */
	int gid_idx;		  /* gid index to use */
	int repeat;		  /* repeat count */
	int mode;		  /* trigger mode: 0=CPU, 1=GPU */
	const char *cuda_plugin_path; /* CUDA plugin path */
};
/* structure to exchange data which is needed to connect the QPs */
struct cm_con_data_t
{
	uint64_t addr;   /* Buffer address */
	uint32_t rkey;   /* Remote key */
	uint32_t qp_num; /* QP number */
	uint16_t lid;	/* LID of the IB port */
	uint8_t gid[16]; /* gid */
} __attribute__((packed));

/* structure of system resources */
struct resources
{
	struct ibv_device_attr
		device_attr;
	/* Device attributes */
	struct ibv_port_attr port_attr;	/* IB port attributes */
	struct cm_con_data_t remote_props; /* values to connect to remote side */
	struct ibv_context *ib_ctx;		   /* device handle */
	struct ibv_pd *pd;				   /* PD handle */
	struct ibv_cq *cq;				   /* CQ handle */
	struct ibv_qp *qp;				   /* QP handle */
	struct ibv_mr *mr;				   /* MR handle for buf */
	char *buf;						   /* memory buffer pointer, used for RDMA and send
ops */
	int sock;						   /* TCP socket file descriptor */
	char sync[2];
};
struct config_t config = {
	NULL,  /* dev_name */
	NULL,  /* server_name */
	19875, /* tcp_port */
	1,	 /* ib_port */
	-1, /* gid_idx */
	1, /* repeat */
	0,  /* mode: 0=CPU, 1=GPU */
	"./cudaPlugin.so"  /* cuda_plugin_path */
};

/******************************************************************************
Socket operations
For simplicity, the example program uses TCP sockets to exchange control
information. If a TCP/IP stack/connection is not available, connection manager
(CM) may be used to pass this information. Use of CM is beyond the scope of
this example
******************************************************************************/
/******************************************************************************
* Function: sock_connect
*
* Input
* servername URL of server to connect to (NULL for server mode)
* port port of service
*
* Output
* none
*
* Returns
* socket (fd) on success, negative error code on failure
*
* Description
* Connect a socket. If servername is specified a client connection will be
* initiated to the indicated server and port. Otherwise listen on the
* indicated port for an incoming connection.
*
******************************************************************************/
static int sock_connect(const char *servername, int port)
{
	struct addrinfo *resolved_addr = NULL;
	struct addrinfo *iterator;
	char service[6];
	int sockfd = -1;
	int listenfd = 0;
	int tmp;
	struct addrinfo hints =
		{
			.ai_flags = AI_PASSIVE,
			.ai_family = AF_INET,
			.ai_socktype = SOCK_STREAM};
	if (sprintf(service, "%d", port) < 0)
		goto sock_connect_exit;
	/* Resolve DNS address, use sockfd as temp storage */
	sockfd = getaddrinfo(servername, service, &hints, &resolved_addr);
	if (sockfd < 0)
	{
		fprintf(stderr, "%s for %s:%d\n", gai_strerror(sockfd), servername, port);
		goto sock_connect_exit;
	}
	/* Search through results and find the one we want */
	for (iterator = resolved_addr; iterator; iterator = iterator->ai_next)
	{
		sockfd = socket(iterator->ai_family, iterator->ai_socktype, iterator->ai_protocol);
		if (sockfd >= 0)
		{
			if (servername){
				/* Client mode. Initiate connection to remote */
				if ((tmp = connect(sockfd, iterator->ai_addr, iterator->ai_addrlen)))
				{
					fprintf(stdout, "failed connect \n");
					close(sockfd);
					sockfd = -1;
				}
            }
			else
			{
					/* Server mode. Set up listening socket an accept a connection */
					listenfd = sockfd;
					sockfd = -1;
					if (bind(listenfd, iterator->ai_addr, iterator->ai_addrlen))
						goto sock_connect_exit;
					listen(listenfd, 1);
					sockfd = accept(listenfd, NULL, 0);
			}
		}
	}
sock_connect_exit:
	if (listenfd)
		close(listenfd);
	if (resolved_addr)
		freeaddrinfo(resolved_addr);
	if (sockfd < 0)
	{
		if (servername)
			fprintf(stderr, "Couldn't connect to %s:%d\n", servername, port);
		else
		{
			perror("server accept");
			fprintf(stderr, "accept() failed\n");
		}
	}
	return sockfd;
}
/******************************************************************************
* Function: sock_sync_data
*
* Input
* sock socket to transfer data on
* xfer_size size of data to transfer
* local_data pointer to data to be sent to remote
*
* Output
* remote_data pointer to buffer to receive remote data
*
* Returns
* 0 on success, negative error code on failure
*
* Description
* Sync data across a socket. The indicated local data will be sent to the
* remote. It will then wait for the remote to send its data back. It is
* assumed that the two sides are in sync and call this function in the proper
* order. Chaos will ensue if they are not. :)
*
* Also note this is a blocking function and will wait for the full data to be
* received from the remote.
*
******************************************************************************/
int sock_sync_data(int sock, int xfer_size, char *local_data, char *remote_data)
{
	int rc;
	int read_bytes = 0;
	int total_read_bytes = 0;
	rc = write(sock, local_data, xfer_size);
	if (rc < xfer_size)
		fprintf(stderr, "Failed writing data during sock_sync_data\n");
	else
		rc = 0;
	while (!rc && total_read_bytes < xfer_size)
	{
		read_bytes = read(sock, remote_data, xfer_size);
		if (read_bytes > 0)
			total_read_bytes += read_bytes;
		else
			rc = read_bytes;
	}
	return rc;
}
/******************************************************************************
End of socket operations
******************************************************************************/
/* poll_completion */
/******************************************************************************
* Function: poll_completion
*
* Input
* res pointer to resources structure
*
* Output
* none
*
* Returns
* 0 on success, 1 on failure
*
* Description
* Poll the completion queue for a single event. This function will continue to
* poll the queue until MAX_POLL_CQ_TIMEOUT milliseconds have passed.
*
******************************************************************************/
static int poll_completion(struct resources *res)
{
	struct ibv_wc wc;
	unsigned long start_time_msec;
	unsigned long cur_time_msec;
	struct timeval cur_time;
	int poll_result;
	int rc = 0;
	/* poll the completion for a while before giving up of doing it .. */
	gettimeofday(&cur_time, NULL);
	start_time_msec = (cur_time.tv_sec * 1000) + (cur_time.tv_usec / 1000);
	do
	{
		poll_result = ibv_poll_cq(res->cq, 1, &wc);
		gettimeofday(&cur_time, NULL);
		cur_time_msec = (cur_time.tv_sec * 1000) + (cur_time.tv_usec / 1000);
	} while ((poll_result == 0) && ((cur_time_msec - start_time_msec) < MAX_POLL_CQ_TIMEOUT));
	if (poll_result < 0)
	{
		/* poll CQ failed */
		fprintf(stderr, "poll CQ failed\n");
		rc = 1;
	}
	else if (poll_result == 0)
	{ /* the CQ is empty */
		fprintf(stderr, "completion wasn't found in the CQ after timeout\n");
		rc = 1;
	}
	else
	{
		/* CQE found */
		logi("completion was found in CQ with status 0x%x", wc.status);
		/* check the completion status (here we don't care about the completion opcode */
		if (wc.status != IBV_WC_SUCCESS)
		{
			fprintf(stderr, "got bad completion with status: 0x%x, vendor syndrome: 0x%x\n", wc.status,
					wc.vendor_err);
			rc = 1;
		}
	}
	return rc;
}

/**
 * 加载CUDA插件
 */
static int load_cuda_plugin() {
    if (config.mode == 0) {
        	logi("CPU mode, skipping CUDA plugin loading");
        return 0;
    }
    
    	logi("Loading CUDA plugin for GPU mode...");
    
    // 加载CUDA插件
    cuda_plugin_handle = dlopen(config.cuda_plugin_path, RTLD_LAZY);
    if (!cuda_plugin_handle) {
        		loge("Failed to load %s: %s", config.cuda_plugin_path, dlerror());
        return -1;
    }
    
    // 获取函数指针
    init_cuda_func = (int (*)(void))dlsym(cuda_plugin_handle, "init_cuda");
    if (!init_cuda_func) {
        		loge("Failed to get init_cuda function: %s", dlerror());
        dlclose(cuda_plugin_handle);
        return -1;
    }
    
    cleanup_cuda_func = (int (*)(void))dlsym(cuda_plugin_handle, "cleanup_cuda");
    if (!cleanup_cuda_func) {
        		loge("Failed to get cleanup_cuda function: %s", dlerror());
        dlclose(cuda_plugin_handle);
        return -1;
    }
    
    convert_host_va_to_gpu_va_func = (int (*)(void *, size_t, int, void **))dlsym(cuda_plugin_handle, "ConvertHostVA2GpuVA");
    if (!convert_host_va_to_gpu_va_func) {
        		loge("Failed to get ConvertHostVA2GpuVA function: %s", dlerror());
        dlclose(cuda_plugin_handle);
        return -1;
    }
    
    trigger_doorbell_func = (int (*)(void *, void *))dlsym(cuda_plugin_handle, "TriggerDoorbell");
    if (!trigger_doorbell_func) {
        		loge("Failed to get TriggerDoorbell function: %s", dlerror());
        dlclose(cuda_plugin_handle);
        return -1;
    }
    
    unregister_host_va_func = (int (*)(void *))dlsym(cuda_plugin_handle, "UnregisterHostVA");
    if (!unregister_host_va_func) {
        		loge("Failed to get UnregisterHostVA function: %s", dlerror());
        dlclose(cuda_plugin_handle);
        return -1;
    }
    
    // 初始化CUDA环境
    if (init_cuda_func() != 0) {
        		loge("Failed to initialize CUDA environment");
        dlclose(cuda_plugin_handle);
        return -1;
    }
    
    	logi("CUDA plugin loaded successfully");
    return 0;
}

/**
 * 转换第一个devx_info的指针到GPU
 */
static int convert_first_devx_info_to_gpu(struct ibv_devx_info *devx_info) {
    if (!convert_host_va_to_gpu_va_func) {
        		loge("ConvertHostVA2GpuVA function not available");
        return -1;
    }
    
    	logi("Converting first devx_info to GPU: bf=%p, ctrl=%p", devx_info->bf, devx_info->ctrl);
    
    // 转换bf指针 (type=0)
    if (convert_host_va_to_gpu_va_func(devx_info->bf, sizeof(uint64_t), 0, &gpu_bf) != 0) {
        		loge("Failed to convert bf to GPU VA");
        return -1;
    }
    
    // 转换ctrl指针 (type=1)
    if (convert_host_va_to_gpu_va_func(devx_info->ctrl, sizeof(uint64_t), 1, &gpu_ctrl) != 0) {
        		loge("Failed to convert ctrl to GPU VA");
        return -1;
    }
    
    	logi("Successfully converted to GPU: bf=%p->%p, ctrl=%p->%p",
		   devx_info->bf, gpu_bf, devx_info->ctrl, gpu_ctrl);
    return 0;
}

/**
 * 卸载CUDA插件
 */
static void unload_cuda_plugin() {
    if (cuda_plugin_handle) {
        // 取消注册GPU指针
        if (unregister_host_va_func) {
            if (gpu_bf) {
                unregister_host_va_func(gpu_bf);
                gpu_bf = NULL;
            }
            if (gpu_ctrl) {
                unregister_host_va_func(gpu_ctrl);
                gpu_ctrl = NULL;
            }
        }
        
        if (cleanup_cuda_func) {
            cleanup_cuda_func();
        }
        dlclose(cuda_plugin_handle);
        cuda_plugin_handle = NULL;
        	logi("CUDA plugin unloaded");
    }
}
/******************************************************************************
* Function: post_send
*
* Input
* res pointer to resources structure
* opcode IBV_WR_SEND, IBV_WR_RDMA_READ or IBV_WR_RDMA_WRITE
*
* Output
* none
*
* Returns
* 0 on success, error code on failure
*
* Description
* This function will create and post a send work request
******************************************************************************/
static int post_send(struct resources *res, int opcode, struct ibv_devx_info *devx_info)
{
	struct ibv_send_wr sr;
	struct ibv_sge sge;
	struct ibv_send_wr *bad_wr = NULL;
	int rc;
	/* prepare the scatter/gather entry */
	memset(&sge, 0, sizeof(sge));
	sge.addr = (uintptr_t)res->buf;
	sge.length = MSG_SIZE;
	sge.lkey = res->mr->lkey;
	/* prepare the send work request */
	memset(&sr, 0, sizeof(sr));
	sr.next = NULL;
	sr.wr_id = 0;
	sr.sg_list = &sge;
	sr.num_sge = 1;
	sr.opcode = opcode;
	sr.send_flags = IBV_SEND_SIGNALED;
	if (opcode != IBV_WR_SEND)
	{
		sr.wr.rdma.remote_addr = res->remote_props.addr;
		sr.wr.rdma.rkey = res->remote_props.rkey;
	}
	/* there is a Receive Request in the responder side, so we won't get any into RNR flow */
	// rc = ibv_post_send(res->qp, &sr, &bad_wr);
	rc = ibv_devx_post_send(res->qp, &sr, &bad_wr, devx_info);
	if (rc)
		fprintf(stderr, "failed to post SR\n");
	else
	{
		switch (opcode)
		{
		case IBV_WR_SEND:
			logi("Send Request was posted");
			break;
		case IBV_WR_RDMA_READ:
			logi("RDMA Read Request was posted");
			break;
		case IBV_WR_RDMA_WRITE:
			logi("RDMA Write Request was posted");
			break;
		default:
			logi("Unknown Request was posted");
			break;
		}
	}
	return rc;
}
/******************************************************************************
* Function: post_receive
*
* Input
* res pointer to resources structure
*
* Output
* none
*
* Returns
* 0 on success, error code on failure
*
* Description
*
******************************************************************************/
__attribute__((__unused__)) static int post_receive(struct resources *res)
{
	struct ibv_recv_wr rr;
	struct ibv_sge sge;
	struct ibv_recv_wr *bad_wr;
	int rc;
	/* prepare the scatter/gather entry */
	memset(&sge, 0, sizeof(sge));
	sge.addr = (uintptr_t)res->buf;
	sge.length = MSG_SIZE;
	sge.lkey = res->mr->lkey;
	/* prepare the receive work request */
	memset(&rr, 0, sizeof(rr));
	rr.next = NULL;
	rr.wr_id = 0;
	rr.sg_list = &sge;
	rr.num_sge = 1;
	/* post the Receive Request to the RQ */
	rc = ibv_post_recv(res->qp, &rr, &bad_wr);
	if (rc)
		fprintf(stderr, "failed to post RR\n");
	else
		logi("Receive Request was posted");
	return rc;
}
/******************************************************************************
* Function: resources_init
*
* Input
* res pointer to resources structure
*
* Output
* res is initialized
*
* Returns
* none
*
* Description
* res is initialized to default values
******************************************************************************/
static void resources_init(struct resources *res)
{
	memset(res, 0, sizeof *res);
	res->sock = -1;
}
/******************************************************************************
* Function: resources_create
*
* Input
* res pointer to resources structure to be filled in
*
* Output
* res filled in with resources
*
* Returns
* 0 on success, 1 on failure
*
* Description
*
* This function creates and allocates all necessary system resources. These
* are stored in res.
*****************************************************************************/
static int resources_create(struct resources *res)
{
	struct ibv_device **dev_list = NULL;
	struct ibv_qp_init_attr qp_init_attr;
	struct ibv_device *ib_dev = NULL;
	size_t size;
	int i;
	int mr_flags = 0;
	int cq_size = 0;
	int num_devices;
	int rc = 0;
	/* if client side */
	if (config.server_name)
	{
		res->sock = sock_connect(config.server_name, config.tcp_port);
		if (res->sock < 0)
		{
			fprintf(stderr, "failed to establish TCP connection to server %s, port %d\n",
					config.server_name, config.tcp_port);
			rc = -1;
			goto resources_create_exit;
		}
	}
	else
	{
		logi("waiting on port %d for TCP connection", config.tcp_port);
		res->sock = sock_connect(NULL, config.tcp_port);
		if (res->sock < 0)
		{
			fprintf(stderr, "failed to establish TCP connection with client on port %d\n",
					config.tcp_port);
			rc = -1;
			goto resources_create_exit;
		}
	}
	logi("TCP connection was established");
	logi("searching for IB devices in host");
	/* get device names in the system */
	dev_list = ibv_get_device_list(&num_devices);
	if (!dev_list)
	{
		fprintf(stderr, "failed to get IB devices list\n");
		rc = 1;
		goto resources_create_exit;
	}
	/* if there isn't any IB device in host */
	if (!num_devices)
	{
		fprintf(stderr, "found %d device(s)\n", num_devices);
		rc = 1;
		goto resources_create_exit;
	}
	logi("found %d device(s)", num_devices);
	/* search for the specific device we want to work with */
	for (i = 0; i < num_devices; i++)
	{
		if (!config.dev_name)
		{
			config.dev_name = strdup(ibv_get_device_name(dev_list[i]));
			logi("device not specified, using first one found: %s", config.dev_name);
		}
		if (!strcmp(ibv_get_device_name(dev_list[i]), config.dev_name))
		{
			ib_dev = dev_list[i];
			break;
		}
	}
	/* if the device wasn't found in host */
	if (!ib_dev)
	{
		fprintf(stderr, "IB device %s wasn't found\n", config.dev_name);
		rc = 1;
		goto resources_create_exit;
	}
	/* get device handle */
	res->ib_ctx = ibv_open_device(ib_dev);
	if (!res->ib_ctx)
	{
		fprintf(stderr, "failed to open device %s\n", config.dev_name);
		rc = 1;
		goto resources_create_exit;
	}
	/* We are now done with device list, free it */
	ibv_free_device_list(dev_list);
	dev_list = NULL;
	ib_dev = NULL;
	/* query port properties */
	if (ibv_query_port(res->ib_ctx, config.ib_port, &res->port_attr))
	{
		fprintf(stderr, "ibv_query_port on port %u failed\n", config.ib_port);
		rc = 1;
		goto resources_create_exit;
	}
	/* allocate Protection Domain */
	res->pd = ibv_alloc_pd(res->ib_ctx);
	if (!res->pd)
	{
		fprintf(stderr, "ibv_alloc_pd failed\n");
		rc = 1;
		goto resources_create_exit;
	}
	/* each side will send only one WR, so Completion Queue with 1 entry is enough */
	cq_size = 1;
	res->cq = ibv_create_cq(res->ib_ctx, cq_size, NULL, NULL, 0);
	if (!res->cq)
	{
		fprintf(stderr, "failed to create CQ with %u entries\n", cq_size);
		rc = 1;
		goto resources_create_exit;
	}
	/* allocate the memory buffer that will hold the data */
	size = MSG_SIZE;
	res->buf = (char *)malloc(size);
	if (!res->buf)
	{
		fprintf(stderr, "failed to malloc %Zu bytes to memory buffer\n", size);
		rc = 1;
		goto resources_create_exit;
	}
	memset(res->buf, 0, size);
	/* only in the server side put the message in the memory buffer */
	if (!config.server_name)
	{
		strcpy(res->buf, MSG);
		logi("going to send the message: '%s'", res->buf);
	}
	else
		memset(res->buf, 0, size);
	/* register the memory buffer */
	mr_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
	res->mr = ibv_reg_mr(res->pd, res->buf, size, mr_flags);
	if (!res->mr)
	{
		fprintf(stderr, "ibv_reg_mr failed with mr_flags=0x%x\n", mr_flags);
		rc = 1;
		goto resources_create_exit;
	}
	logi("MR was registered with addr=%p, lkey=0x%x, rkey=0x%x, flags=0x%x",
			res->buf, res->mr->lkey, res->mr->rkey, mr_flags);
	/* create the Queue Pair */
	memset(&qp_init_attr, 0, sizeof(qp_init_attr));
	qp_init_attr.qp_type = IBV_QPT_RC;
	qp_init_attr.sq_sig_all = 1;
	qp_init_attr.send_cq = res->cq;
	qp_init_attr.recv_cq = res->cq;
	qp_init_attr.cap.max_send_wr = MAX_WR_COUNT;
	qp_init_attr.cap.max_recv_wr = MAX_WR_COUNT;
	qp_init_attr.cap.max_send_sge = 1;
	qp_init_attr.cap.max_recv_sge = 1;
	res->qp = ibv_create_qp(res->pd, &qp_init_attr);
	if (!res->qp)
	{
		fprintf(stderr, "failed to create QP\n");
		rc = 1;
		goto resources_create_exit;
	}
	logi("QP was created, QP number=0x%x", res->qp->qp_num);
resources_create_exit:
	if (rc)
	{
		/* Error encountered, cleanup */
		if (res->qp)
		{
			ibv_destroy_qp(res->qp);
			res->qp = NULL;
		}
		if (res->mr)
		{
			ibv_dereg_mr(res->mr);
			res->mr = NULL;
		}
		if (res->buf)
		{
			free(res->buf);
			res->buf = NULL;
		}
		if (res->cq)
		{
			ibv_destroy_cq(res->cq);
			res->cq = NULL;
		}
		if (res->pd)
		{
			ibv_dealloc_pd(res->pd);
			res->pd = NULL;
		}
		if (res->ib_ctx)
		{
			ibv_close_device(res->ib_ctx);
			res->ib_ctx = NULL;
		}
		if (dev_list)
		{
			ibv_free_device_list(dev_list);
			dev_list = NULL;
		}
		if (res->sock >= 0)
		{
			if (close(res->sock))
				fprintf(stderr, "failed to close socket\n");
			res->sock = -1;
		}
	}
	return rc;
}
/******************************************************************************
* Function: modify_qp_to_init
*
* Input
* qp QP to transition
*
* Output
* none
*
* Returns
* 0 on success, ibv_modify_qp failure code on failure
*
* Description
* Transition a QP from the RESET to INIT state
******************************************************************************/
static int modify_qp_to_init(struct ibv_qp *qp)
{
	struct ibv_qp_attr attr;
	int flags;
	int rc;
	memset(&attr, 0, sizeof(attr));
	attr.qp_state = IBV_QPS_INIT;
	attr.port_num = config.ib_port;
	attr.pkey_index = 0;
	attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
	flags = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
	rc = ibv_modify_qp(qp, &attr, flags);
	if (rc)
		fprintf(stderr, "failed to modify QP state to INIT\n");
	return rc;
}
/******************************************************************************
* Function: modify_qp_to_rtr
*
* Input
* qp QP to transition
* remote_qpn remote QP number
* dlid destination LID
* dgid destination GID (mandatory for RoCEE)
*
* Output
* none
*
* Returns
* 0 on success, ibv_modify_qp failure code on failure
*
* Description
* Transition a QP from the INIT to RTR state, using the specified QP number
******************************************************************************/
static int modify_qp_to_rtr(struct ibv_qp *qp, uint32_t remote_qpn, uint16_t dlid, uint8_t *dgid)
{
	struct ibv_qp_attr attr;
	int flags;
	int rc;
	memset(&attr, 0, sizeof(attr));
	attr.qp_state = IBV_QPS_RTR;
	attr.path_mtu = IBV_MTU_256;
	attr.dest_qp_num = remote_qpn;
	attr.rq_psn = 0;
	attr.max_dest_rd_atomic = 1;
	attr.min_rnr_timer = 0x12;
	attr.ah_attr.is_global = 0;
	attr.ah_attr.dlid = dlid;
	attr.ah_attr.sl = 0;
	attr.ah_attr.src_path_bits = 0;
	attr.ah_attr.port_num = config.ib_port;
	if (config.gid_idx >= 0)
	{
		attr.ah_attr.is_global = 1;
		attr.ah_attr.port_num = 1;
		memcpy(&attr.ah_attr.grh.dgid, dgid, 16);
		attr.ah_attr.grh.flow_label = 0;
		attr.ah_attr.grh.hop_limit = 1;
		attr.ah_attr.grh.sgid_index = config.gid_idx;
		attr.ah_attr.grh.traffic_class = 0;
	}
	flags = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
			IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;
	rc = ibv_modify_qp(qp, &attr, flags);
	if (rc)
		fprintf(stderr, "failed to modify QP state to RTR\n");
	return rc;
}
/******************************************************************************
* Function: modify_qp_to_rts
*
* Input
* qp QP to transition
*
* Output
* none
*
* Returns
* 0 on success, ibv_modify_qp failure code on failure
*
* Description
* Transition a QP from the RTR to RTS state
******************************************************************************/
static int modify_qp_to_rts(struct ibv_qp *qp)
{
	struct ibv_qp_attr attr;
	int flags;
	int rc;
	memset(&attr, 0, sizeof(attr));
	attr.qp_state = IBV_QPS_RTS;
	attr.timeout = 0x12;
	attr.retry_cnt = 6;
	attr.rnr_retry = 0;
	attr.sq_psn = 0;
	attr.max_rd_atomic = 1;
	flags = IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
			IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC;
	rc = ibv_modify_qp(qp, &attr, flags);
	if (rc)
		fprintf(stderr, "failed to modify QP state to RTS\n");
	return rc;
}
/******************************************************************************
* Function: connect_qp
*
* Input
* res pointer to resources structure
*
* Output
* none
*
* Returns
* 0 on success, error code on failure
*
* Description
* Connect the QP. Transition the server side to RTR, sender side to RTS
******************************************************************************/
static int connect_qp(struct resources *res)
{
	struct cm_con_data_t local_con_data;
	struct cm_con_data_t remote_con_data;
	struct cm_con_data_t tmp_con_data;
	int rc = 0;
	char temp_char;
	union ibv_gid my_gid;
	if (config.gid_idx >= 0)
	{
		rc = ibv_query_gid(res->ib_ctx, config.ib_port, config.gid_idx, &my_gid);
		if (rc)
		{
			fprintf(stderr, "could not get gid for port %d, index %d\n", config.ib_port, config.gid_idx);
			return rc;
		}
	}
	else
		memset(&my_gid, 0, sizeof my_gid);
	/* exchange using TCP sockets info required to connect QPs */
	local_con_data.addr = htonll((uintptr_t)res->buf);
	local_con_data.rkey = htonl(res->mr->rkey);
	local_con_data.qp_num = htonl(res->qp->qp_num);
	local_con_data.lid = htons(res->port_attr.lid);
	memcpy(local_con_data.gid, &my_gid, 16);
	fprintf(stdout, "\nLocal LID = 0x%x\n", res->port_attr.lid);
	if (sock_sync_data(res->sock, sizeof(struct cm_con_data_t), (char *)&local_con_data, (char *)&tmp_con_data) < 0)
	{
		fprintf(stderr, "failed to exchange connection data between sides\n");
		rc = 1;
		goto connect_qp_exit;
	}
	remote_con_data.addr = ntohll(tmp_con_data.addr);
	remote_con_data.rkey = ntohl(tmp_con_data.rkey);
	remote_con_data.qp_num = ntohl(tmp_con_data.qp_num);
	remote_con_data.lid = ntohs(tmp_con_data.lid);
	memcpy(remote_con_data.gid, tmp_con_data.gid, 16);
	/* save the remote side attributes, we will need it for the post SR */
	res->remote_props = remote_con_data;
	fprintf(stdout, "Remote address = 0x%" PRIx64 "\n", remote_con_data.addr);
	fprintf(stdout, "Remote rkey = 0x%x\n", remote_con_data.rkey);
	fprintf(stdout, "Remote QP number = 0x%x\n", remote_con_data.qp_num);
	fprintf(stdout, "Remote LID = 0x%x\n", remote_con_data.lid);
	if (config.gid_idx >= 0)
	{
		uint8_t *p = remote_con_data.gid;
		fprintf(stdout, "Remote GID =%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x\n ",p[0],
				  p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11], p[12], p[13], p[14], p[15]);
	}
	/* modify the QP to init */
	rc = modify_qp_to_init(res->qp);
	if (rc)
	{
		fprintf(stderr, "change QP state to INIT failed\n");
		goto connect_qp_exit;
	}
	/* let the client post RR to be prepared for incoming messages */
	// if (config.server_name)
	// {
	// 	rc = post_receive(res);
	// 	if (rc)
	// 	{
	// 		fprintf(stderr, "failed to post RR\n");
	// 		goto connect_qp_exit;
	// 	}
	// }
	/* modify the QP to RTR */
	rc = modify_qp_to_rtr(res->qp, remote_con_data.qp_num, remote_con_data.lid, remote_con_data.gid);
	if (rc)
	{
		fprintf(stderr, "failed to modify QP state to RTR\n");
		goto connect_qp_exit;
	}
	rc = modify_qp_to_rts(res->qp);
	if (rc)
	{
		fprintf(stderr, "failed to modify QP state to RTR\n");
		goto connect_qp_exit;
	}
	fprintf(stdout, "QP state was change to RTS\n");
	/* sync to make sure that both sides are in states that they can connect to prevent packet loose */
	if (sock_sync_data(res->sock, 1, res->sync, &temp_char)) /* just send a dummy char back and forth */
	{
		fprintf(stderr, "sync error after QPs are were moved to RTS\n");
		rc = 1;
	}
connect_qp_exit:
	return rc;
}
/******************************************************************************
* Function: resources_destroy
*
* Input
* res pointer to resources structure
*
* Output
* none
*
* Returns
* 0 on success, 1 on failure
*
* Description
* Cleanup and deallocate all resources used
******************************************************************************/
static int resources_destroy(struct resources *res)
{
	int rc = 0;
	if (res->qp)
		if (ibv_destroy_qp(res->qp))
		{
			fprintf(stderr, "failed to destroy QP\n");
			rc = 1;
		}
	if (res->mr)
		if (ibv_dereg_mr(res->mr))
		{
			fprintf(stderr, "failed to deregister MR\n");
			rc = 1;
		}
	if (res->buf)
		free(res->buf);
	if (res->cq)
		if (ibv_destroy_cq(res->cq))
		{
			fprintf(stderr, "failed to destroy CQ\n");
			rc = 1;
		}
	if (res->pd)
		if (ibv_dealloc_pd(res->pd))
		{
			fprintf(stderr, "failed to deallocate PD\n");
			rc = 1;
		}
	if (res->ib_ctx)
		if (ibv_close_device(res->ib_ctx))
		{
			fprintf(stderr, "failed to close device context\n");
			rc = 1;
		}
	if (res->sock >= 0)
		if (close(res->sock))
		{
			fprintf(stderr, "failed to close socket\n");
			rc = 1;
		}
	return rc;
}
/******************************************************************************
* Function: print_config
*
* Input
* none
*
* Output
* none
*
* Returns
* none
*
* Description
* Print out config information
******************************************************************************/
static void print_config(void)
{
	fprintf(stdout, " ------------------------------------------------\n");
	fprintf(stdout, " Device name : \"%s\"\n", config.dev_name);
	fprintf(stdout, " IB port : %u\n", config.ib_port);
	if (config.server_name)
		fprintf(stdout, " IP : %s\n", config.server_name);
	fprintf(stdout, " TCP port : %u\n", config.tcp_port);
	if (config.gid_idx >= 0)
		fprintf(stdout, " GID index : %u\n", config.gid_idx);
	fprintf(stdout, " ------------------------------------------------\n\n");
}

/******************************************************************************
* Function: usage
*
* Input
* argv0 command line arguments
*
* Output
* none
*
* Returns
* none
*
* Description
* print a description of command line syntax
******************************************************************************/
static void usage(const char *argv0)
{
	fprintf(stdout, "Usage:\n");
	fprintf(stdout, " %s start a server and wait for connection\n", argv0);
	fprintf(stdout, " %s <host> connect to server at <host>\n", argv0);
	fprintf(stdout, "\n");
	fprintf(stdout, "Options:\n");
	fprintf(stdout, " -p, --port <port> listen on/connect to port <port> (default 18515)\n");
	fprintf(stdout, " -d, --ib-dev <dev> use IB device <dev> (default first device found)\n");
	fprintf(stdout, " -i, --ib-port <port> use port <port> of IB device (default 1)\n");
	fprintf(stdout, " -g, --gid_idx <git index> gid index to be used in GRH (default not used)\n");
	fprintf(stdout, " -r, --repeat <repeat> repeat count (default 1)\n");
	fprintf(stdout, " -m, --mode <mode> trigger mode: 0=CPU, 1=GPU (default 0)\n");
	fprintf(stdout, " -c, --cuda-plugin <path> CUDA plugin path (default ./cudaPlugin.so)\n");
}
/******************************************************************************
* Function: main
*
* Input
* argc number of items in argv
* argv command line parameters
*
* Output
* none
*
* Returns
* 0 on success, 1 on failure
*
* Description
* Main program code
******************************************************************************/

/******************************************************************************
* Function: producer_thread
*
* Input
* arg pointer to resources structure
*
* Output
* none
*
* Returns
* NULL
*
* Description
* 生产者线程：获取devx_info并放入环形队列
******************************************************************************/
static void* producer_thread(void *arg)
{
	struct resources *res = (struct resources *)arg;
	
	for (int i = 0; i < config.repeat && thread_running; ++i) {
		if (config.server_name) { // client
			uint8_t value = i % 9;
			memset(res->buf, value, MSG_SIZE);
			struct ibv_devx_info devx_info;
			
					// 轮询等待队列有空间，设置5秒超时
			unsigned long start_time_msec;
			unsigned long cur_time_msec;
			struct timeval cur_time;
			
			gettimeofday(&cur_time, NULL);
			start_time_msec = (cur_time.tv_sec * 1000) + (cur_time.tv_usec / 1000);
			
			while ((request_idx - complete_idx) >= MAX_WR_COUNT && thread_running) {
				// 检查是否超时（5秒 = 5000毫秒）
				gettimeofday(&cur_time, NULL);
				cur_time_msec = (cur_time.tv_sec * 1000) + (cur_time.tv_usec / 1000);
				if ((cur_time_msec - start_time_msec) >= 5000) {
					fprintf(stderr, "Producer timeout: queue full for 5 seconds\n");
					break;
				}
				// 短暂休眠避免过度占用CPU
				usleep(1000); // 1毫秒
			}
				
			if (!thread_running) {
				break;
			}

			if (post_send(res, IBV_WR_RDMA_WRITE, &devx_info)) {
				fprintf(stderr, "failed to post SR in producer thread\n");
				break;
			}
			
			// 将devx_info放入队列
			devx_infos[request_idx % MAX_DEVX_INFO_COUNT] = devx_info;
			
			
			logi("Producer: WR idx = %d, bf = %p, ctrl = %p, queue_idx = %d", 
					devx_info.idx, devx_info.bf, devx_info.ctrl, request_idx);

				// 如果是GPU模式，转换第一个devx_info的指针到GPU
			if (config.mode == 1) {
				logi("Converting first devx_info to GPU...");
				if (convert_first_devx_info_to_gpu(&devx_infos[0]) != 0) {
					loge("Failed to convert devx_info to GPU");
					return NULL;
				}
				logi("Successfully converted first devx_info to GPU");
			}

			request_idx++;
		}
	}
	
	logi("Producer thread finished");
	return NULL;
}

/******************************************************************************
* Function: consumer_thread
*
* Input
* arg pointer to resources structure
*
* Output
* none
*
* Returns
* NULL
*
* Description
* 消费者线程：从环形队列取出devx_info并触发门铃
******************************************************************************/
static void* consumer_thread(void *arg)
{
	struct resources *res = (struct resources *)arg;
	int completed_count = 0;
	
	while (completed_count < config.repeat && thread_running) {
		struct ibv_devx_info devx_info;
		
		// 轮询等待队列有数据，设置5秒超时
		unsigned long start_time_msec;
		unsigned long cur_time_msec;
		struct timeval cur_time;
		
		gettimeofday(&cur_time, NULL);
		start_time_msec = (cur_time.tv_sec * 1000) + (cur_time.tv_usec / 1000);
		while (complete_idx >= request_idx && thread_running) {
			// 检查是否超时（5秒 = 5000毫秒）
			gettimeofday(&cur_time, NULL);
			cur_time_msec = (cur_time.tv_sec * 1000) + (cur_time.tv_usec / 1000);
			if ((cur_time_msec - start_time_msec) >= 5000) {
				fprintf(stderr, "Consumer timeout: queue empty for 5 seconds\n");
				break;
			}
			// 短暂休眠避免过度占用CPU
			usleep(1000); // 1毫秒
		}
		
		if (!thread_running) {
			break;
		}
		
		// 从队列取出devx_info
		devx_info = devx_infos[complete_idx % MAX_DEVX_INFO_COUNT];
		
		// 根据模式选择触发方式
		if (config.mode == 1 && trigger_doorbell_func && gpu_bf && gpu_ctrl) {
			// GPU模式：使用GPU kernel触发门铃
			if (trigger_doorbell_func(gpu_bf, gpu_ctrl) != 0) {
				logw("GPU doorbell trigger failed, falling back to CPU mode");
				return NULL;
			}
		} else {
			// CPU模式：直接触发门铃
				logi("Consumer CPU mode: WR idx = %d, bf = %p, ctrl = %p, queue_idx = %d",
		   devx_info.idx, devx_info.bf, devx_info.ctrl, complete_idx);
			*((volatile uint64_t *)devx_info.bf) = *(uint64_t *)devx_info.ctrl;
		}
		
		// 等待完成
		if (poll_completion(res)) {
			fprintf(stderr, "poll completion failed in consumer thread\n");
			break;
		}
		
		complete_idx++;
		completed_count++;
	}
	
	logi("Consumer thread finished, completed %d requests", completed_count);
	return NULL;
}

int main(int argc, char *argv[])
{
	struct resources res;
	int rc = 1;
	char temp_char;
	/* parse the command line parameters */
	while (1)
	{
		int c;
		static struct option long_options[] = {
			{.name = "port", .has_arg = 1, .val = 'p'},
			{.name = "ib-dev", .has_arg = 1, .val = 'd'},
			{.name = "ib-port", .has_arg = 1, .val = 'i'},
			{.name = "gid-idx", .has_arg = 1, .val = 'g'},
			{.name = "repeat", .has_arg = 1, .val = 'r'},
			{.name = "mode", .has_arg = 1, .val = 'm'},
			{.name = "cuda-plugin", .has_arg = 1, .val = 'c'},
			{.name = NULL, .has_arg = 0, .val = '\0'}
        };
		c = getopt_long(argc, argv, "p:d:i:g:r:m:c:", long_options, NULL);
		if (c == -1)
			break;
		switch (c)
		{
		case 'p':
			config.tcp_port = strtoul(optarg, NULL, 0);
			break;
		case 'd':
			config.dev_name = strdup(optarg);
			break;
		case 'i':
			config.ib_port = strtoul(optarg, NULL, 0);
			if (config.ib_port < 0)
			{
				usage(argv[0]);
				return 1;
			}
			break;
		case 'g':
			config.gid_idx = strtoul(optarg, NULL, 0);
			if (config.gid_idx < 0)
			{
				usage(argv[0]);
				return 1;
			}
			break;
		case 'r':
			config.repeat = strtoul(optarg, NULL, 0);
			if (config.repeat <= 0)
			{
				usage(argv[0]);
				return 1;
			}
			break;
		case 'm':
			config.mode = strtoul(optarg, NULL, 0);
			if (config.mode < 0 || config.mode > 1)
			{
				loge("Invalid mode: %d. Use 0 for CPU mode or 1 for GPU mode", config.mode);
				usage(argv[0]);
				return 1;
			}
			break;
		case 'c':
			config.cuda_plugin_path = strdup(optarg);
			break;
		default:
			usage(argv[0]);
			return 1;
		}
	}
	/* parse the last parameter (if exists) as the server name */
	if (optind == argc - 1)
		config.server_name = argv[optind];
    if(config.server_name){
        	logi("servername=%s", config.server_name);
    }
	else if (optind < argc)
	{
		usage(argv[0]);
		return 1;
	}
	/* print the used parameters for info*/
	print_config();
	
	// 加载CUDA插件（如果需要）
	if (config.mode == 1) {
		if (load_cuda_plugin() != 0) {
			fprintf(stderr, "Failed to load CUDA plugin\n");
			rc = 1;
			goto main_exit;
		}
	}
	
	/* init all of the resources, so cleanup will be easy */
	resources_init(&res);
	/* create resources before using them */
	if (resources_create(&res))
	{
		fprintf(stderr, "failed to create resources\n");
		goto main_exit;
	}
	
	/* connect the QPs */
	if (connect_qp(&res))
	{
		fprintf(stderr, "failed to connect QPs\n");
		goto main_exit;
	}
	
	/* Sync so we are sure server side has data ready before client tries to read it */
	if (sock_sync_data(res.sock, 1, res.sync, &temp_char)) /* just send a dummy char back and forth */
	{
		fprintf(stderr, "sync error before RDMA ops\n");
		rc = 1;
		goto main_exit;
	}

	if (config.server_name) { // client
		// 启动生产者线程和消费者线程
		pthread_t producer_tid, consumer_tid;
		
		logi("Starting producer and consumer threads...");
		
		if (pthread_create(&producer_tid, NULL, producer_thread, &res) != 0) {
			fprintf(stderr, "failed to create producer thread\n");
			rc = 1;
			goto main_exit;
		}
		
		if (pthread_create(&consumer_tid, NULL, consumer_thread, &res) != 0) {
			fprintf(stderr, "failed to create consumer thread\n");
			thread_running = 0;
			pthread_join(producer_tid, NULL);
			rc = 1;
			goto main_exit;
		}
		
		// 等待线程完成
		pthread_join(producer_tid, NULL);
		pthread_join(consumer_tid, NULL);
		
		logi("All threads completed");
	}

	/* Sync so server will know that client is done mucking with its memory */
	if (sock_sync_data(res.sock, 1, res.sync, &temp_char)) /* just send a dummy char back and forth */
	{
		fprintf(stderr, "sync error after RDMA ops\n");
		rc = 1;
		goto main_exit;
	}

	// 打印最后一次的值
	if (!config.server_name) // server
	{
		printf("server got data: ");
		for (int n = 0; n < MSG_SIZE; ++n)
		{
			printf("%d, ", (uint8_t)res.buf[n]);
		}
		printf("\n");
	}

	rc = 0;
main_exit:
	// 停止线程运行
	thread_running = 0;
	
	// 卸载CUDA插件
	unload_cuda_plugin();
	
	if (resources_destroy(&res))
	{
		fprintf(stderr, "failed to destroy resources\n");
		rc = 1;
	}
	if (config.dev_name)
		free((char *)config.dev_name);
	if (config.cuda_plugin_path && config.cuda_plugin_path != "./cudaPlugin.so")
		free((char *)config.cuda_plugin_path);
	fprintf(stdout, "\ntest result is %d\n", rc);
	return rc;
}
