/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#define BLOCK_SIZE 512

__global__ void reduction(float *out, float *in, unsigned size)
{
    /********************************************************************
    Load a segment of the input vector into shared memory
    Traverse the reduction tree
    Write the computed sum to the output vector at the correct index
    ********************************************************************/

    // INSERT KERNEL CODE HERE
    __device__ __shared__ float partialsum[BLOCK_SIZE*2];
    unsigned int t = threadIdx.x;
    unsigned int start = blockIdx.x*blockDim.x*2;
    if(t+start<size)
    {
    partialsum[t]=in[start+t];
    } 
    else
    partialsum[t]=0.0;
    if((blockDim.x+t+start)<size)
    {
    partialsum[blockDim.x+t]=in[blockDim.x+t+start];
    }
    else
    partialsum[blockDim.x+t]=0.0;
    
    for(unsigned int stride=1;stride <=blockDim.x;stride=stride*2)
    {
    __syncthreads();
    if(t%stride==0){
    //__syncthreads();
    partialsum[2*t]+=partialsum[2*t+stride];
    } 
    __syncthreads();
    }
    out[blockIdx.x]=partialsum[0];
}
