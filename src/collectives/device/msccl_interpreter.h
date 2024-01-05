/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "devcomm.h"
#include "primitives.h"
#include "collectives.h"
#include "npkit/npkit.h"

#include <time.h>

#define MSCCL_MAX_ITER 65536

// flags are a 3-tuple of (workindex, gridoffset_iter, step) and it follows a lexicographical order. a threadblock is ahead of another iff its flag is ahead
#define COMPUTE_FLAG(__WORKINDEX__,__GRIDOFFSET_ITER__,__STEP__) \
  MSCCL_MAX_ITER*MSCCL_MAX_NUM_STEPS*(uint64_t)__WORKINDEX__ + ((uint64_t)__GRIDOFFSET_ITER__ * MSCCL_MAX_NUM_STEPS + (uint64_t)__STEP__)

#define GET_WORKINDEX_FROM_FLAG(__FLAG__) \
  (__FLAG__) / (MSCCL_MAX_ITER*MSCCL_MAX_NUM_STEPS)

namespace {
  // a copy of the volatile load/store from prims_ll
  template<typename U>
  __device__ static U load(U *src) {
    union {
      U elt;
      uint16_t u2;
      uint32_t u4;
      uint64_t u8;
    };
    if(sizeof(U) == 1)
      asm("ld.volatile.global.b8 %0,[%1];" : "=r"(u4) : "l"(src));
    else if(sizeof(U) == 2)
      asm("ld.volatile.global.b16 %0,[%1];" : "=h"(u2) : "l"(src));
    else if(sizeof(U) == 4)
      asm("ld.volatile.global.b32 %0,[%1];" : "=r"(u4) : "l"(src));
    else
      asm("ld.volatile.global.b64 %0,[%1];" : "=l"(u8) : "l"(src));
    return elt;
  }

  template<typename U>
  __device__ static void store(U *dst, U val) {
    union {
      U elt;
      uint16_t u2;
      uint32_t u4;
      uint64_t u8;
    };
    elt = val;
    if(sizeof(U) == 1)
      asm("st.volatile.global.b8 [%0],%1;" :: "l"(dst), "r"(u4));
    else if(sizeof(U) == 2)
      asm("st.volatile.global.b16 [%0],%1;" :: "l"(dst), "h"(u2));
    else if(sizeof(U) == 4)
      asm("st.volatile.global.b32 [%0],%1;" :: "l"(dst), "r"(u4));
    else
      asm("st.volatile.global.b64 [%0],%1;" :: "l"(dst), "l"(u8));
  }

  inline __device__ void barrier(int nthreads) {
    asm volatile ("bar.sync %1, %0;" :: "r"(nthreads), "r"(15));
  }

  template<typename T, typename RedOp, typename Proto>
  __device__ __forceinline__ void runInterpreter(ncclWorkElem *args, int sizeMultiplier) {
    const int tid = threadIdx.x;
    const int nthreads = args->header.nWarps*WARP_SIZE;
    const int bid = blockIdx.x;
    struct mscclThreadBlock* mscclTB = &ncclShmem.mscclShmem.mscclTB;

    if (tid==0){
      printf("DEVICE | runInterpreter | nthreads is: %d\n", nthreads);
    }

    // User pointers for primitives
    T* thisInput = (T*)args->sendbuff;
    T* thisOutput = (T*)args->recvbuff;
    T* thisScratch = (T*)ncclShmem.mscclShmem.scratchBuffer;
    int recvPeer = mscclTB->recvpeer;
    int sendPeer = mscclTB->sendpeer;

    const ssize_t chunkSize = int(Proto::calcBytePerStep()/sizeof(T) * (Proto::Id == NCCL_PROTO_SIMPLE ? MSCCL_CHUNKSTEPS : 1));
    int minChunkSize;
    if (Proto::Id == NCCL_PROTO_LL)
      minChunkSize = nthreads*(Proto::calcBytePerGrain()/sizeof(T));
    if (Proto::Id == NCCL_PROTO_LL128) {
      // We should not need the final /2 but it makes performance much, much smoother. Might be a bug somewhere.
      minChunkSize = nthreads*(Proto::calcBytePerGrain()/sizeof(T))/2;
    }

    NPKIT_GPU_SYNC_TIME(bid, tid);

    RedOp redFn(args->redOpArg);
    Primitives<T, RedOp, FanAsymmetric<1,1>, 1, Proto, 0> prims
      (tid, nthreads, &recvPeer, &sendPeer, thisInput, thisOutput, args->redOpArg);

    NPKIT_GPU_SET_CTX_ID(prims);

    const ssize_t size = args->count;
    const ssize_t sizePerMscclChunk = (size*sizeMultiplier)/ncclShmem.mscclShmem.nchunksPerLoop;
    uint16_t mscclMaxAllowedCount = args->mscclWork.mscclMaxAllowedCount;
    int8_t needsFence = ncclShmem.mscclShmem.needsFence;

    // msccl flags all start out with 0. this is used as a part of the flag to make sure different work items deal with different synchronization flags
    // this still needs more work. when we make a way around the queue, the flag might have been set to undesired values. will be fixed in subsequent versions.
    const int64_t workIndex = ncclShmem.mscclShmem.workIndex;
    volatile struct mscclFlag* mscclFlags = ncclShmem.mscclShmem.flags;

    volatile unsigned long long lyd_msccl_chunk0_time = clock64();
    volatile unsigned long long lyd_msccl_chunk1_time = clock64();
    volatile int chunk_msccl_iteration_lyd; 

    // profile by lyd
    // if (tid == 0 && bid == 0) {
    //   printf("total size in run interpreter() is: %ld\n", size); 
    //   printf("for loop size (the MSCCL chunk size) in run interpreter() is: %ld\n", sizePerMscclChunk); 
    //   printf("nthreads is: %d\n", nthreads);
    // }
    for (ssize_t gridOffset = 0, iter = 0; gridOffset < sizePerMscclChunk; gridOffset += chunkSize, iter++) {
      ssize_t realChunkSize;
      if (Proto::Id == NCCL_PROTO_SIMPLE) {
        realChunkSize = min(chunkSize, sizePerMscclChunk-gridOffset);
        realChunkSize = roundUp(realChunkSize, (nthreads-WARP_SIZE)*sizeof(uint64_t)/sizeof(T));
      }
      else
        realChunkSize = min(chunkSize, divUp(sizePerMscclChunk-gridOffset, minChunkSize)*minChunkSize);
      realChunkSize = int(realChunkSize);
      int nelem = min(realChunkSize, sizePerMscclChunk-gridOffset);
       
      if (gridOffset == 0) lyd_msccl_chunk0_time == clock64();
      
      ssize_t srcoffset, dstoffset;
      T* srcPointer, * dstPointer;
      int step = 0;
      for (int i = 0; i < mscclTB->nsteps; i++){
        struct mscclTransfer* msccltran = &mscclTB->transfers[i];
        // first wait if there is a dependence
        int16_t numDependences = msccltran->numDependences;
        if (numDependences > 0){
          NPKIT_GPU_ENTER_EVENT(NPKIT_EVENT_DEP_CHECK_ENTRY, msccltran->numDependences);
          
          if (tid < numDependences){
            int16_t dependentPointer = msccltran->depencePointer;
            int8_t dependentBid = mscclTB->dependentBid[dependentPointer+tid];
            int16_t dependentStep = mscclTB->dependentStep[dependentPointer+tid];
            printf("DEVICE | %s:%d | runInterpreter() | step in TB is %d, bid is %d, tid is %d, numDependences is %d, dependentPointer is %d, dependentBid is %d, dependentStep is %d\n",  __FILE__, __LINE__, i, bid, tid, numDependences, dependentPointer, dependentBid, dependentStep);
            uint64_t goalFlag = COMPUTE_FLAG(workIndex, iter, dependentStep);
            while (true){
              uint64_t curFlag = (mscclFlags + dependentBid)->flag;
              if (curFlag >= goalFlag && GET_WORKINDEX_FROM_FLAG(curFlag) == workIndex) break;
            }
          }
          step += numDependences-1;
          barrier(nthreads);

          NPKIT_GPU_ENTER_EVENT(NPKIT_EVENT_DEP_CHECK_EXIT, msccltran->numDependences);
        }


        srcPointer = (msccltran->srcbuffer == MSCCL_INPUT_BUFFER) ? thisInput : ((msccltran->srcbuffer == MSCCL_OUTPUT_BUFFER) ? thisOutput : thisScratch);
        dstPointer = (msccltran->dstbuffer == MSCCL_INPUT_BUFFER) ? thisInput : ((msccltran->dstbuffer == MSCCL_OUTPUT_BUFFER) ? thisOutput : thisScratch);
        prims.setDataPtrs(srcPointer, dstPointer);
        int count = msccltran->count;
        for (int c = 0; c < count; c += mscclMaxAllowedCount) {
          srcoffset = gridOffset + (ssize_t) (msccltran->srcoffset+c) * sizePerMscclChunk;
          dstoffset = gridOffset + (ssize_t) (msccltran->dstoffset+c) * sizePerMscclChunk;
          int thisCount = min(mscclMaxAllowedCount, count-c);
          int thisNelem = nelem*thisCount;

          lyd_msccl_chunk1_time = clock64(); 
          chunk_msccl_iteration_lyd = gridOffset/chunkSize;
          double time_msccl_between_chunks = ((double)(lyd_msccl_chunk1_time - lyd_msccl_chunk0_time))/1.41e3; 
          
          
          
          if (msccltran->type == MSCCL_SEND){
            // profile by lyd
            // printf("[LYD INFO] T in MSCCL MSCCL_SEND (thisNelem=%d, tid=%d, bid=%d) of chunk %d, step %d, c %d is: %f us\n\n", thisNelem, threadIdx.x, blockIdx.x, chunk_msccl_iteration_lyd, i, c, time_msccl_between_chunks);
            prims.sendWithBarrier(srcoffset, thisNelem); // LL.send is the only situation where there is no barrier at the end.
          } else if (msccltran->type == MSCCL_RECV){
            // profile by lyd
            // printf("[LYD INFO] T in MSCCL MSCCL_RECV (thisNelem=%d, tid=%d, bid=%d) of chunk %d, step %d, c %d is: %f us\n\n", thisNelem, threadIdx.x, blockIdx.x, chunk_msccl_iteration_lyd, i, c, time_msccl_between_chunks);
            prims.recv(dstoffset, thisNelem);
          } else if (msccltran->type == MSCCL_REDUCE) {
            // profile by lyd
            // printf("[LYD INFO] T in MSCCL MSCCL_REDUCE (thisNelem=%d, tid=%d, bid=%d) of chunk %d, step %d, c %d is: %f us\n\n", thisNelem, threadIdx.x, blockIdx.x, chunk_msccl_iteration_lyd, i, c, time_msccl_between_chunks);
            int numReductions = msccltran->numReductions;
            if (thisNelem < nthreads){
              NPKIT_GPU_ENTER_EVENT(NPKIT_EVENT_REDUCE_ENTRY, thisNelem*sizeof(T));

              if (tid < thisNelem){
                dstoffset = gridOffset + (ssize_t) (msccltran->dstoffset+c) * sizePerMscclChunk;
                T* dst_index = dstPointer + dstoffset +tid;
                T o = load(dst_index);
                for (int r = 0; r < numReductions; r++){
                  srcoffset = gridOffset + (ssize_t) (mscclTB->reductionSrcOffsets[msccltran->reductionPointer+r]+c) * sizePerMscclChunk;
                  T t = load(srcPointer + srcoffset + tid);
                  o = redFn(t,o);
                }
                store(dst_index, o);
              }
              barrier(nthreads);

              NPKIT_GPU_ENTER_EVENT(NPKIT_EVENT_REDUCE_EXIT, thisNelem*sizeof(T));
            } else {
              T* srcs[MSCCL_MAX_REDUCE_FUSION+1]; // +1 is for SIMPLE protocol as dst is added in the list of srcs
              dstoffset = gridOffset + (ssize_t) (msccltran->dstoffset+c) * sizePerMscclChunk;
              T* dst = dstPointer + dstoffset;
              for (int r = 0; r < numReductions; r++) {
                srcoffset = gridOffset + (ssize_t) (mscclTB->reductionSrcOffsets[msccltran->reductionPointer+r]+c) * sizePerMscclChunk;
                srcs[r] = srcPointer + srcoffset;
              }
              prims.reduce(srcs, numReductions, &dst, 1, thisNelem);
            }
            if (c == 0) step += (numReductions-1); // only advance step once!
          } else if (msccltran->type == MSCCL_RECV_COPY_SEND){
            // profile by lyd
            // printf("[LYD INFO] T in MSCCL MSCCL_RECV_COPY_SEND (thisNelem=%d, tid=%d, bid=%d) of chunk %d, step %d, c %d is: %f us\n\n", thisNelem, threadIdx.x, blockIdx.x, chunk_msccl_iteration_lyd, i, c, time_msccl_between_chunks);
            prims.recvCopySend(dstoffset, thisNelem);
          } else if (msccltran->type == MSCCL_RECV_REDUCE_SEND){
            // profile by lyd
            // printf("[LYD INFO] T in MSCCL MSCCL_RECV_REDUCE_SEND (thisNelem=%d, tid=%d, bid=%d) of chunk %d, step %d, c %d is: %f us\n\n", thisNelem, threadIdx.x, blockIdx.x, chunk_msccl_iteration_lyd, i, c, time_msccl_between_chunks);
            prims.recvReduceSend(srcoffset, thisNelem);
          } else if (msccltran->type == MSCCL_RECV_REDUCE_COPY_SEND){
            // profile by lyd
            // printf("[LYD INFO] T in MSCCL MSCCL_RECV_REDUCE_COPY_SEND (thisNelem=%d, tid=%d, bid=%d) of chunk %d, step %d, c %d is: %f us\n\n", thisNelem, threadIdx.x, blockIdx.x, chunk_msccl_iteration_lyd, i, c, time_msccl_between_chunks);
            prims.recvReduceCopySend(srcoffset, dstoffset, thisNelem);
          } else if (msccltran->type == MSCCL_RECV_REDUCE_COPY){
            // profile by lyd
            // printf("[LYD INFO] T in MSCCL MSCCL_RECV_REDUCE_COPY (thisNelem=%d, tid=%d, bid=%d) of chunk %d, step %d, c %d is: %f us\n\n", thisNelem, threadIdx.x, blockIdx.x, chunk_msccl_iteration_lyd, i, c, time_msccl_between_chunks);
            prims.recvReduceCopy(srcoffset, dstoffset, thisNelem);
          } else if (msccltran->type == MSCCL_LOCAL_COPY){
            // profile by lyd
            // printf("[LYD INFO] T in MSCCL MSCCL_LOCAL_COPY (thisNelem=%d, tid=%d, bid=%d) of chunk %d, step %d, c %d is: %f us\n\n", thisNelem, threadIdx.x, blockIdx.x, chunk_msccl_iteration_lyd, i, c, time_msccl_between_chunks);
            prims.localCopy(srcPointer+srcoffset, dstPointer+dstoffset, thisNelem);
          } else
            return;
        }
        if (msccltran->has_dependence && tid == nthreads-1){
	        if (needsFence) __threadfence();
          mscclFlags[bid].flag = (uint64_t) COMPUTE_FLAG(workIndex, iter, step);
        }
        step++;
      }
    }
  }
}
