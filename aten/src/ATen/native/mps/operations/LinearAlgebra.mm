//  Copyright © 2022 Apple Inc.

#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <ATen/Utils.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/mps/OperationUtils.h>
#include <torch/library.h>

#ifdef __OBJC__
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#endif


namespace at {
namespace native {

/*
 * Helper functions to be used for mm/addmm for detecting the Transpositions
 * when doing Batched GEMM operations.
 */

static Tensor prepare_batch_matrix_by_transposing(const Tensor& tensor,
                                       bool& transpose_tensor,
                                       int64_t& ld_tensor,
                                       bool transpose_result,
                                       int64_t m, int64_t n) {
  IntArrayRef tensor_strides = tensor.strides();
  Tensor tensor_;
  int fast_dim = transpose_result ? 2 : 1;
  int leading_dim = transpose_result ? 1 : 2;

  if (tensor_strides[fast_dim] == 1 &&
    (tensor_strides[leading_dim] >= std::max<int64_t>(1, m))) {
    transpose_tensor = false;
    tensor_ = tensor;
    ld_tensor = tensor_strides[leading_dim];
  } else if ((tensor_strides[leading_dim] == 1) &&
    (tensor_strides[fast_dim] >= std::max<int64_t>(1, n))) {
    transpose_tensor = true;
    tensor_ = tensor;
    ld_tensor = tensor_strides[fast_dim];
  } else {
    transpose_tensor = !transpose_result;
    // gemm call requires leading dimension and stride parameters to be non-zero
    bool is_stride_non_zero = tensor.stride(1) != 0 && tensor.stride(2) != 0;
    if (tensor.is_contiguous() && is_stride_non_zero) {
      tensor_ = tensor;
    } else {
      tensor_ = tensor.clone(at::MemoryFormat::Contiguous);
    }
    ld_tensor = tensor_.stride(1);
  }

  return tensor_;
}

/*
 * Helper functions to be used for mm/addmm for detecting the Transpositions
 * when doing GEMM operations.
 */
void prepare_matrices_for_broadcasting(
  const Tensor * bias,
  const Tensor & self,
  const Tensor & other,
  const Scalar * beta,
  bool * transpose_mat1_times_mat2,
  bool & transpose_mat1,
  bool & transpose_mat2) {
  TORCH_CHECK(self.dim() == 2 && other.dim() == 2, "tensors must be 2-D");
  if (bias && beta->toDouble() != 0.0f) {
    TORCH_CHECK(bias->dim() == 2, "tensors must be 2-D");
  }

  std::pair<int64_t, int64_t> mat1_sizes;
  std::pair<int64_t, int64_t> mat2_sizes;

  mat1_sizes = std::make_pair(self.sizes()[0], self.sizes()[1]);
  mat2_sizes = std::make_pair(other.sizes()[0], other.sizes()[1]);

  if (mat1_sizes == mat2_sizes) {
    transpose_mat2 = true;
    std::swap(mat2_sizes.first, mat2_sizes.second);
  }
  if (bias && beta && transpose_mat1_times_mat2) {
    if (beta->toDouble() != 0.0f && mat1_sizes.first == bias->sizes()[1] && mat2_sizes.second == bias->sizes()[0])
      *transpose_mat1_times_mat2 = true;
  }
}

enum LinearAlgebraOpType {
  ADDBMM_OP_TYPE,
  BADDBMM_OP_TYPE
};

Tensor& mm_out_mps_impl(
    const Tensor& self,
    const Tensor& other,
    Tensor& output) {
  using namespace mps;
  TORCH_CHECK(self.dim() == 2 && other.dim() == 2, "tensors must be 2-D");

  TensorArg args[]{{output, "out", 0}, {self, "mat1", 1}, {other, "mat2", 2}};
  checkAllSameGPU("mm", args);

  TORCH_CHECK(output.is_mps());

  // Transpose inputs if needed
  IntArrayRef output_sizes = output.sizes();
  if ((output_sizes[0] == 0) || (output_sizes[1] == 0)) {
    return output;
  }

  struct CachedGraph : public mps::MPSCachedGraph
  {
    CachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor *selfTensor_ = nil;
    MPSGraphTensor *otherTensor_ = nil;
    MPSGraphTensor *outputTensor_ = nil;
  };

  MPSStream* stream = getCurrentMPSStream();

  bool transpose_mat1            = false;
  bool transpose_mat2            = false;

  prepare_matrices_for_broadcasting(NULL, self, other, NULL, NULL, transpose_mat1, transpose_mat2);

  mps::MPSGraphCache *cache_ = mps::MPSGraphCache::getInstance();

  @autoreleasepool {

    string key = "mm_out_mps_impl" + getTensorsStringKey({self, other})
                                   + ":" + to_string(transpose_mat1) + ":" + to_string(transpose_mat2);

    CachedGraph* cachedGraph = static_cast<CachedGraph *>(cache_->LookUp(key));
    if(!cachedGraph) {

      mps::MPSCachedGraph *tmpCachedGraph = cache_->CreateCachedGraph(key, ^ mps::MPSCachedGraph * () {
        CachedGraph *newCachedGraph = nil;

        @autoreleasepool{
          MPSGraph *mpsGraph = mps::make_mps_graph();
          newCachedGraph = new CachedGraph(mpsGraph);

          MPSGraphTensor *selfTensor = mps::mpsGraphRankedPlaceHolder(mpsGraph, self);
          MPSGraphTensor *otherTensor =  mps::mpsGraphRankedPlaceHolder(mpsGraph, other);

          MPSGraphTensor* t1 = nil;
          MPSGraphTensor* t2 = nil;

          if(transpose_mat1)
            t1 = [mpsGraph transposeTensor:selfTensor
                                 dimension:-1
                             withDimension:-2
                                      name:nil];
          else
            t1 = selfTensor;

          if(transpose_mat2)
            t2 = [mpsGraph transposeTensor:otherTensor
                                 dimension:-1
                             withDimension:-2
                                      name:nil];
          else
            t2 = otherTensor;

          MPSGraphTensor* outputTensor = [mpsGraph matrixMultiplicationWithPrimaryTensor:t1
                                                                         secondaryTensor:t2
                                                                                    name:nil];

          newCachedGraph->selfTensor_ = selfTensor;
          newCachedGraph->otherTensor_ = otherTensor;
          newCachedGraph->outputTensor_ = outputTensor;
        }
        return newCachedGraph;
      });
      cachedGraph = static_cast<CachedGraph *>(tmpCachedGraph);
    }
    Placeholder selfPlaceholder = Placeholder(cachedGraph->selfTensor_, self);
    Placeholder otherPlaceholder = Placeholder(cachedGraph->otherTensor_, other);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output);

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
      selfPlaceholder.getMPSGraphTensor() : selfPlaceholder.getMPSGraphTensorData(),
      otherPlaceholder.getMPSGraphTensor() : otherPlaceholder.getMPSGraphTensorData()
    };

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
      outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()
    };

    mps::runMPSGraph(stream, cachedGraph->graph(), feeds, results);
  }

  return output;
}

Tensor& addmm_out_mps_impl(
    const Tensor& bias,
    const Tensor& self,  // input
    const Tensor& other, // weight
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& output) {
  using namespace mps;

  TORCH_CHECK(output.is_mps());
  TORCH_CHECK(self.dim() == 2 && other.dim() == 2, "tensors must be 2-D");

  TensorArg args[]{{output, "out", 0}, {bias, "self", 1}, {self, "mat1", 2}, {other, "mat2", 3}};
  checkAllSameGPU(__func__, args);

  IntArrayRef mat1_sizes = self.sizes();
  IntArrayRef mat2_sizes = other.sizes();
  IntArrayRef bias_sizes;
  c10::MaybeOwned<Tensor> bias_;
  if (&output != &bias) {
    bias_ = expand_size(bias, {mat1_sizes[0], mat2_sizes[1]}, "addmm");
    bias_sizes = bias_->sizes();
  } else {
    bias_ = c10::MaybeOwned<Tensor>::borrowed(bias);
    bias_sizes = bias_->sizes();
    TORCH_CHECK(output.dim() == 2, "tensors must be 2-D");
    TORCH_CHECK(bias_sizes[0] == mat1_sizes[0], "self_ dim 0 must match mat1 dim 0");
    TORCH_CHECK(bias_sizes[1] == mat2_sizes[1], "self_ dim 1 must match mat2 dim 1");
  }

  if (&output != &self) {
    output.resize_(bias_sizes);
    if (beta.toComplexDouble() != 0.0) {
      at::native::copy_(output, *bias_);
    }
  }
  IntArrayRef output_sizes = output.sizes();
  if ((output_sizes[0] == 0) || (output_sizes[1] == 0)) {
    return output;
  }

  MPSStream* stream = getCurrentMPSStream();

  MPSGraph* mpsGraph = make_mps_graph();

  bool transpose_mat1_times_mat2 = false;
  bool transpose_mat1            = false;
  bool transpose_mat2            = false;

  prepare_matrices_for_broadcasting(&bias, self, other, &beta, &transpose_mat1_times_mat2, transpose_mat1, transpose_mat2);

  struct CachedGraph : public mps::MPSCachedGraph
  {
    CachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor *selfTensor_ = nil;
    MPSGraphTensor *otherTensor_ = nil;
    MPSGraphTensor *biasTensor_ = nil;
    MPSGraphTensor *outputTensor_ = nil;
  };

  mps::MPSGraphCache *cache_ = mps::MPSGraphCache::getInstance();

  @autoreleasepool {
    string key = "addmm_out_mps_impl" + getTensorsStringKey({self, other, bias})
                                       + ":" + to_string(transpose_mat1) + ":" + to_string(transpose_mat2)
                                       + ":" + to_string(beta.toDouble())
                                       + ":" + to_string(alpha.toDouble());
    CachedGraph* cachedGraph = static_cast<CachedGraph *>(cache_->LookUp(key));
    if(!cachedGraph) {

      mps::MPSCachedGraph *tmpCachedGraph = cache_->CreateCachedGraph(key, ^ mps::MPSCachedGraph * () {
        CachedGraph *newCachedGraph = nil;

        @autoreleasepool{
          MPSGraph *mpsGraph = mps::make_mps_graph();
          newCachedGraph = new CachedGraph(mpsGraph);

          MPSGraphTensor *selfTensor = mps::mpsGraphRankedPlaceHolder(mpsGraph, self);
          MPSGraphTensor *otherTensor =  mps::mpsGraphRankedPlaceHolder(mpsGraph, other);
          MPSGraphTensor *biasTensor =  mps::mpsGraphRankedPlaceHolder(mpsGraph, bias);

          MPSGraphTensor* t1 = nil;
          MPSGraphTensor* t2 = nil;

          if(transpose_mat1)
            t1 = [mpsGraph transposeTensor:selfTensor
                                 dimension:-1
                             withDimension:-2
                                      name:nil];
          else
            t1 = selfTensor;

          if(transpose_mat2)
            t2 = [mpsGraph transposeTensor:otherTensor
                                 dimension:-1
                             withDimension:-2
                                      name:nil];
          else
            t2 = otherTensor;


          // TODO: Use alpha and beta here with fill_.Scalar and mul
          // Intermediate as placeholder
          MPSGraphTensor* productTensor = [mpsGraph matrixMultiplicationWithPrimaryTensor:t1
                                                                          secondaryTensor:t2
                                                                                     name:@"MM/(mat1@mat2)"];

          // Intermediates for beta and alpha
          MPSGraphTensor* betaTensor = [mpsGraph constantWithScalar:beta.toDouble()
                                                           dataType:getMPSScalarType(bias.scalar_type())];
          MPSGraphTensor* alphaTensor = [mpsGraph constantWithScalar:alpha.toDouble()
                                                           dataType:getMPSScalarType(self.scalar_type())];

          // Intermediates for multiplying by beta and alpha
          MPSGraphTensor* productTimesAlphaTensor = [mpsGraph multiplicationWithPrimaryTensor:productTensor
                                                                              secondaryTensor:alphaTensor
                                                                                         name:@"MM/alpha*(mat1@mat2)"];
          MPSGraphTensor* biasTimesBetaTensor = [mpsGraph multiplicationWithPrimaryTensor:biasTensor
                                                                          secondaryTensor:betaTensor
                                                                                     name:@"MM/beta*input"];

          if (transpose_mat1_times_mat2)
            biasTimesBetaTensor = [mpsGraph transposeTensor: biasTimesBetaTensor
                                                  dimension: -1
                                              withDimension: -2
                                                       name: nil];

          MPSGraphTensor* outputTensor = [mpsGraph additionWithPrimaryTensor:productTimesAlphaTensor
                                                             secondaryTensor:biasTimesBetaTensor
                                                                        name:@"MM/beta*input + alpha*(mat1@mat2)"];

          newCachedGraph->selfTensor_ = selfTensor;
          newCachedGraph->otherTensor_ = otherTensor;
          newCachedGraph->biasTensor_ = biasTensor;
          newCachedGraph->outputTensor_ = outputTensor;
        }
        return newCachedGraph;
      });
      cachedGraph = static_cast<CachedGraph *>(tmpCachedGraph);
    }

    Placeholder selfPlaceholder = Placeholder(cachedGraph->selfTensor_, self);
    Placeholder otherPlaceholder = Placeholder(cachedGraph->otherTensor_, other);
    Placeholder biasPlaceholder = Placeholder(cachedGraph->biasTensor_, bias);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output);

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
      selfPlaceholder.getMPSGraphTensor() : selfPlaceholder.getMPSGraphTensorData(),
      otherPlaceholder.getMPSGraphTensor() : otherPlaceholder.getMPSGraphTensorData(),
      biasPlaceholder.getMPSGraphTensor() : biasPlaceholder.getMPSGraphTensorData()
    };

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
      outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()
    };

    mps::runMPSGraph(stream, cachedGraph->graph(), feeds, results);
  }

  return output;
}

Tensor& ndmm_out_mps_impl(
  const Tensor & batch1,
  const Tensor & batch2,
  Tensor & result) {

  using namespace mps;
  if (batch1.numel() == 0 || batch2.numel() == 0) {
    return result;
  }
  MPSStream* stream = getCurrentMPSStream();

  struct CachedGraph : public mps::MPSCachedGraph
  {
    CachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor *batch1Tensor_ = nil;
    MPSGraphTensor *batch2Tensor_ = nil;
    MPSGraphTensor *outputTensor_ = nil;
  };

  mps::MPSGraphCache *cache_ = mps::MPSGraphCache::getInstance();

  bool expandBatch2 = (batch1.dim() == 4) && (batch2.dim() == 3);
  bool expandBatch1 = (batch2.dim() == 4) && (batch1.dim() == 3);

  @autoreleasepool {
    string key = "ndmm_out_mps_impl" + getTensorsStringKey({batch1, batch2});

    CachedGraph* cachedGraph = static_cast<CachedGraph *>(cache_->LookUp(key));
    if(!cachedGraph) {

      mps::MPSCachedGraph *tmpCachedGraph = cache_->CreateCachedGraph(key, ^ mps::MPSCachedGraph * () {
        CachedGraph *newCachedGraph = nil;

        @autoreleasepool{
          MPSGraph *mpsGraph = mps::make_mps_graph();
          newCachedGraph = new CachedGraph(mpsGraph);

          MPSGraphTensor *batch1Tensor = mps::mpsGraphRankedPlaceHolder(mpsGraph, batch1);
          MPSGraphTensor *batch2Tensor =  mps::mpsGraphRankedPlaceHolder(mpsGraph, batch2);

          // If left or right tensors are 1D vector, perform expand to make them 2D
          MPSGraphTensor *batch1InputTensor = batch1Tensor;
          if(batch1.dim() == 1 || expandBatch1) {
              batch1InputTensor = [mpsGraph expandDimsOfTensor:batch1Tensor axis:0 name:nil];
          }
          MPSGraphTensor *batch2InputTensor = batch2Tensor;
          if(batch2.dim() == 1) {
              batch2InputTensor = [mpsGraph expandDimsOfTensor:batch2Tensor axis:1 name:nil];
          }
          if(expandBatch2) {
              batch2InputTensor = [mpsGraph expandDimsOfTensor:batch2Tensor axis:0 name:nil];
          }
          
          MPSGraphTensor* productTensor = [mpsGraph matrixMultiplicationWithPrimaryTensor:batch1InputTensor
                                                                          secondaryTensor:batch2InputTensor
                                                                                     name:@"MM/(batch1@batch2)"];

          if(batch1.dim() == 1) {
              productTensor = [mpsGraph squeezeTensor:productTensor axis:-2 name:nil];
          }
          if(batch2.dim() == 1) {
              productTensor = [mpsGraph squeezeTensor:productTensor axis:-1 name:nil];
          }

          newCachedGraph->batch1Tensor_ = batch1Tensor;
          newCachedGraph->batch2Tensor_ = batch2Tensor;
          newCachedGraph->outputTensor_ = productTensor;
        }
        return newCachedGraph;
      });
      cachedGraph = static_cast<CachedGraph *>(tmpCachedGraph);
    }
    
    Placeholder batch1Placeholder = Placeholder(cachedGraph->batch1Tensor_, batch1);
    Placeholder batch2Placeholder = Placeholder(cachedGraph->batch2Tensor_, batch2);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, result);

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
      batch1Placeholder.getMPSGraphTensor() : batch1Placeholder.getMPSGraphTensorData(),
      batch2Placeholder.getMPSGraphTensor() : batch2Placeholder.getMPSGraphTensorData(),
    };

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
      outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()
    };
    
    mps::runMPSGraph(stream, cachedGraph->graph(), feeds, results);
  }

  return result;
}


Tensor& bmm_out_mps_impl(
  const Tensor & batch1,
  const Tensor & batch2,
  Tensor & result) {
  using namespace mps;

  if (batch1.numel() == 0 || batch2.numel() == 0) {
    return result;
  }

  MPSStream* stream = getCurrentMPSStream();

  struct CachedGraph : public mps::MPSCachedGraph
  {
    CachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor *batch1Tensor_ = nil;
    MPSGraphTensor *batch2Tensor_ = nil;
    MPSGraphTensor *outputTensor_ = nil;
  };

  mps::MPSGraphCache *cache_ = mps::MPSGraphCache::getInstance();

  @autoreleasepool {
    string key = "bmm_out_mps_impl" + getTensorsStringKey({batch1, batch2});

    CachedGraph* cachedGraph = static_cast<CachedGraph *>(cache_->LookUp(key));
    if(!cachedGraph) {

      mps::MPSCachedGraph *tmpCachedGraph = cache_->CreateCachedGraph(key, ^ mps::MPSCachedGraph * () {
        CachedGraph *newCachedGraph = nil;

        @autoreleasepool{
          MPSGraph *mpsGraph = mps::make_mps_graph();
          newCachedGraph = new CachedGraph(mpsGraph);

          MPSGraphTensor *batch1Tensor = mps::mpsGraphRankedPlaceHolder(mpsGraph, batch1);
          MPSGraphTensor *batch2Tensor =  mps::mpsGraphRankedPlaceHolder(mpsGraph, batch2);

          MPSGraphTensor* productTensor = [mpsGraph matrixMultiplicationWithPrimaryTensor:batch1Tensor
                                                                          secondaryTensor:batch2Tensor
                                                                                     name:@"MM/(batch1@batch2)"];

          newCachedGraph->batch1Tensor_ = batch1Tensor;
          newCachedGraph->batch2Tensor_ = batch2Tensor;
          newCachedGraph->outputTensor_ = productTensor;
        }
        return newCachedGraph;
      });
      cachedGraph = static_cast<CachedGraph *>(tmpCachedGraph);
    }
    Placeholder batch1Placeholder = Placeholder(cachedGraph->batch1Tensor_, batch1);
    Placeholder batch2Placeholder = Placeholder(cachedGraph->batch2Tensor_, batch2);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, result);

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
      batch1Placeholder.getMPSGraphTensor() : batch1Placeholder.getMPSGraphTensorData(),
      batch2Placeholder.getMPSGraphTensor() : batch2Placeholder.getMPSGraphTensorData(),
    };

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
      outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()
    };

    mps::runMPSGraph(stream, cachedGraph->graph(), feeds, results);
  }

  return result;
}

Tensor& addbmm_or_baddbmm_out_mps_impl(
  const Tensor       & input,
  const Tensor       & batch1,
  const Tensor       & batch2,
  const Scalar       & beta,
  const Scalar       & alpha,
  Tensor             & result,
  LinearAlgebraOpType  opType) {
  using namespace mps;

  TORCH_CHECK(input.is_mps());
  TORCH_CHECK(batch1.is_mps());
  TORCH_CHECK(batch2.is_mps());
  TORCH_CHECK(result.is_mps());

  TORCH_CHECK(batch1.dim() == 3, "batch1 must be a 3D tensor");
  TORCH_CHECK(batch2.dim() == 3, "batch2 must be a 3D tensor");
  TORCH_CHECK(batch1.size(0) == batch2.size(0),
      "batch1 and batch2 must have same number of batches, got ",
      batch1.size(0), " and ", batch2.size(0));
  TORCH_CHECK(batch1.size(2) == batch2.size(1),
      "Incompatible matrix sizes for bmm (",
      batch1.size(1), "x", batch1.size(2), " and ",
      batch2.size(1), "x", batch2.size(2), ")");

  if (opType == ADDBMM_OP_TYPE)
  {
    result.resize_as_(input);

    const int64_t num_batches = batch1.size(0);

    if (num_batches == 0) {
      result.zero_();
      return result;
    }
  }

  MPSStream* stream = getCurrentMPSStream();

  struct CachedGraph : public mps::MPSCachedGraph
  {
    CachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor *inputTensor_ = nil;
    MPSGraphTensor *batch1Tensor_ = nil;
    MPSGraphTensor *batch2Tensor_ = nil;
    MPSGraphTensor *outputTensor_ = nil;
  };

  mps::MPSGraphCache *cache_ = mps::MPSGraphCache::getInstance();

  @autoreleasepool {
    string key = (opType == ADDBMM_OP_TYPE) ? ("addbmm_out_mps_impl") : ("baddbmm_out_mps_impl");
    key += getTensorsStringKey({batch1, batch2, input})
               + ":" + to_string(beta.toDouble())
               + ":" + to_string(alpha.toDouble());

    CachedGraph* cachedGraph = static_cast<CachedGraph *>(cache_->LookUp(key));
    if(!cachedGraph) {

      mps::MPSCachedGraph *tmpCachedGraph = cache_->CreateCachedGraph(key, ^ mps::MPSCachedGraph * () {
        CachedGraph *newCachedGraph = nil;

        @autoreleasepool{
          MPSGraph *mpsGraph = mps::make_mps_graph();
          newCachedGraph = new CachedGraph(mpsGraph);

          MPSGraphTensor *inputTensor  = mps::mpsGraphRankedPlaceHolder(mpsGraph, input);
          MPSGraphTensor *batch1Tensor = mps::mpsGraphRankedPlaceHolder(mpsGraph, batch1);
          MPSGraphTensor *batch2Tensor =  mps::mpsGraphRankedPlaceHolder(mpsGraph, batch2);

          // Intermediates for beta and alpha
          MPSGraphTensor* betaTensor = [mpsGraph constantWithScalar: beta.toDouble()
                                                           dataType: getMPSScalarType(input.scalar_type())];
          MPSGraphTensor* alphaTensor = [mpsGraph constantWithScalar: alpha.toDouble()
                                                            dataType: getMPSScalarType(batch1.scalar_type())];

          MPSGraphTensor* productTensor = [mpsGraph matrixMultiplicationWithPrimaryTensor:batch1Tensor
                                                                          secondaryTensor:batch2Tensor
                                                                                     name:@"(batch1@batch2)"];

          MPSGraphTensor* reductionSumTensor = productTensor;
          if (opType == ADDBMM_OP_TYPE) {
            reductionSumTensor = [mpsGraph reductionSumWithTensor: productTensor
                                                             axis: 0
                                                             name: @"reductionSum(batch1@batch2)"];
          }

          // Intermediates for multiplying by beta and alpha
          MPSGraphTensor* reductionSumTimesAlphaTensor = [mpsGraph multiplicationWithPrimaryTensor: reductionSumTensor
                                                                              secondaryTensor: alphaTensor
                                                                                         name: @"alpha*(batch1@batch2)"];
          MPSGraphTensor* biasTimesBetaTensor = [mpsGraph multiplicationWithPrimaryTensor: inputTensor
                                                                          secondaryTensor: betaTensor
                                                                                     name: @"beta*input"];

          MPSGraphTensor* outputTensor = [mpsGraph additionWithPrimaryTensor:reductionSumTimesAlphaTensor
                                                             secondaryTensor:biasTimesBetaTensor
                                                                        name:@"beta*input + alpha*(batch1@batch2)"];

          newCachedGraph->inputTensor_  = inputTensor;
          newCachedGraph->batch1Tensor_ = batch1Tensor;
          newCachedGraph->batch2Tensor_ = batch2Tensor;
          newCachedGraph->outputTensor_ = outputTensor;
        }
        return newCachedGraph;
      });
      cachedGraph = static_cast<CachedGraph *>(tmpCachedGraph);
    }
    Placeholder inputPlaceholder  = Placeholder(cachedGraph->inputTensor_,  input);
    Placeholder batch1Placeholder = Placeholder(cachedGraph->batch1Tensor_, batch1);
    Placeholder batch2Placeholder = Placeholder(cachedGraph->batch2Tensor_, batch2);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, result);

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
      inputPlaceholder.getMPSGraphTensor()  : inputPlaceholder.getMPSGraphTensorData(),
      batch1Placeholder.getMPSGraphTensor() : batch1Placeholder.getMPSGraphTensorData(),
      batch2Placeholder.getMPSGraphTensor() : batch2Placeholder.getMPSGraphTensorData(),
    };

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
      outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()
    };

    mps::runMPSGraph(stream, cachedGraph->graph(), feeds, results);
  }

  return result;
}


bool ndmm_gradient_reduction(NSMutableArray<NSNumber *> * reduction_axes, MPSShape* input1_shape, MPSShape* input2_shape, MPSShape* grad_output_shape, bool is_grad_input1) {
    
  auto input1_dims = [input1_shape count];
  auto input2_dims = [input2_shape count];
  auto grad_output_dims = [grad_output_shape count];
  
  auto input_tensor_matmul_dims = is_grad_input1? input2_dims: input1_dims;
  auto output_tensor_matmul_dims = is_grad_input1? input1_dims: input2_dims;
  
  // Batch dimensions of the product
  auto product_tensor_batch_shape = (input_tensor_matmul_dims > grad_output_dims)? (is_grad_input1? input2_shape: input1_shape): grad_output_shape;
  // Batch dimensions of the final gradient
  auto output_tensor_batch_shape = is_grad_input1? input1_shape: input2_shape;
  // All batch dims will be present in product through bcast
  auto product_batch_dims = std::max(input_tensor_matmul_dims - 2, grad_output_dims - 2);
  // Number of batch dims in grad_input tensor
  auto final_batch_dims = output_tensor_matmul_dims - 2;
  
  // If there are fewer batch dims in final result than in product, we must reduce them
  if(product_batch_dims > final_batch_dims) {
      auto num_dims_to_reduce = product_batch_dims - final_batch_dims;
      for(int i = 0; i < num_dims_to_reduce; ++i)
          [reduction_axes addObject:[NSNumber numberWithInteger:i]];
  }
  // In the equal dims, check for bcast dims. They must be reduced
  auto product_batch_start = product_batch_dims - std::min(product_batch_dims, final_batch_dims);
  for(int i = 0; i < product_batch_dims; ++i) {
      auto product_batch_iter = product_batch_start + i;
      if(output_tensor_batch_shape[i].intValue == 1)
          [reduction_axes addObject:[NSNumber numberWithInteger:product_batch_iter]];
  }
  auto num_reduction_axes = [reduction_axes count];
  return (num_reduction_axes > 0);

}


Tensor ndmm_backward_out_mps(
  const Tensor       & tensor1,
  const Tensor       & tensor2,
  const Tensor       & grad_output,
  bool               is_grad_input1) {
  using namespace mps;

  TORCH_CHECK(tensor1.is_mps());
  TORCH_CHECK(tensor2.is_mps());
  TORCH_CHECK(grad_output.is_mps());
    
  MPSShape* input1_final_shape = nil;
  MPSShape* input2_final_shape = nil;
  MPSShape* grad_output_shape = getMPSShape(grad_output);
  MPSShape* grad_output_final_shape = nil;
  NSMutableArray<NSNumber *>* grad_output_mutable = [grad_output_shape mutableCopy];
  MPSShape* grad_input_final_shape = nil;
    
  if(tensor1.dim() == 1 || tensor2.dim() == 1) {
    NSUInteger indexToUpdate;
    // Is input1 a vector, convert it to a row matrix, and expand grad_output
    if(tensor1.dim() == 1) {
        input1_final_shape = @[[NSNumber numberWithInteger:1], [NSNumber numberWithInteger:tensor1.size(0)]];
        indexToUpdate = [grad_output_mutable count] - 1;
    }
    
    // Is input2 a vector, convert it to a column matrix, and expand grad_output
    if(tensor2.dim() == 1) {
        input2_final_shape = @[[NSNumber numberWithInteger:tensor2.size(0)], [NSNumber numberWithInteger:1]];
        indexToUpdate = [grad_output_mutable count];
    }
    
    [grad_output_mutable insertObject:[NSNumber numberWithInteger:1] atIndex:indexToUpdate];
    grad_output_final_shape = [NSArray arrayWithArray:grad_output_mutable];
  }
  if(tensor1.dim() == 3 && tensor2.dim() == 4) {
    input1_final_shape = @[[NSNumber numberWithInteger:1], [NSNumber numberWithInteger:tensor1.size(0)], [NSNumber numberWithInteger:tensor1.size(1)], [NSNumber  numberWithInteger:tensor1.size(2)]];
  }
  if(tensor2.dim() == 3 && tensor1.dim() == 4) {
    input2_final_shape = @[[NSNumber numberWithInteger:1], [NSNumber numberWithInteger:tensor2.size(0)], [NSNumber numberWithInteger:tensor2.size(1)], [NSNumber  numberWithInteger:tensor2.size(2)]];
  }
    
  NSMutableArray<NSNumber *> *reduction_axes = [[NSMutableArray alloc] init];
  MPSShape* tensor1_dims = input1_final_shape? input1_final_shape: getMPSShape(tensor1);
  MPSShape* tensor2_dims = input2_final_shape? input2_final_shape: getMPSShape(tensor2);
  MPSShape* grad_output_dims = grad_output_final_shape? grad_output_final_shape: getMPSShape(grad_output);
  bool needs_reduction = ndmm_gradient_reduction(reduction_axes, tensor1_dims, tensor2_dims, grad_output_dims, is_grad_input1);

  MPSStream* stream = getCurrentMPSStream();

  struct CachedGraph : public mps::MPSCachedGraph
  {
    CachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor *fwdInputTensor_ = nil;
    MPSGraphTensor *gradOutputTensor_ = nil;
    MPSGraphTensor *gradInputTensor_ = nil;
  };
    
  IntArrayRef grad_input_size = is_grad_input1? tensor1.sizes() : tensor2.sizes();
  Tensor grad_input = at::native::empty_mps(grad_input_size,
                                          grad_output.scalar_type(),
                                          c10::nullopt,
                                          kMPS,
                                          c10::nullopt,
                                          grad_output.suggest_memory_format());
  MPSShape* grad_input_shape = getMPSShape(grad_input);
  TORCH_CHECK(grad_input.is_mps());

  mps::MPSGraphCache *cache_ = mps::MPSGraphCache::getInstance();

  @autoreleasepool {
    std::string key = "ndmm_backward_out_mps" + getTensorsStringKey({tensor1, tensor2, grad_output}) + std::to_string(is_grad_input1);

    CachedGraph* cachedGraph = static_cast<CachedGraph *>(cache_->LookUp(key));
    if(!cachedGraph) {
      mps::MPSCachedGraph *tmpCachedGraph = cache_->CreateCachedGraph(key, ^ mps::MPSCachedGraph * () {
      CachedGraph *newCachedGraph = nil;

      /*
      Case 1: No bcast in batch dim
      Forward: A_pmk x B_pkn = C_pmn
      dA_pmk = dC_pmn x B_pkn^T
      dB_pkn = A_pmk^T x dC_pmn
      
      Case 2: With bcast in batch dim
      Forward: A_pmk x B_kn = C_pmn
      dA_pmk = dC_pmn x B_kn^T
      dB_kn = ReduceOverP(A_pmk^T x dC_pmn)
      */
      @autoreleasepool{
        MPSGraph *mpsGraph = mps::make_mps_graph();
        newCachedGraph = new CachedGraph(mpsGraph);
        MPSGraphTensor *fwdInputTensor = is_grad_input1? mps::mpsGraphRankedPlaceHolder(mpsGraph, tensor2): mps::mpsGraphRankedPlaceHolder(mpsGraph, tensor1);
        MPSGraphTensor *gradOutputTensor = mps::mpsGraphRankedPlaceHolder(mpsGraph, grad_output);
        MPSGraphTensor *gradInputTensor =  mps::mpsGraphRankedPlaceHolder(mpsGraph, grad_input);
        MPSGraphTensor* fwdInputTensorExpanded = fwdInputTensor;
        MPSGraphTensor* gradOutputTensorExpanded = gradOutputTensor;
            
        if(is_grad_input1) {
          if(input2_final_shape) {
            fwdInputTensorExpanded = [mpsGraph reshapeTensor:fwdInputTensorExpanded
                                                   withShape:input2_final_shape
                                                        name:nil];
          }
          if(grad_output_final_shape) {
            gradOutputTensorExpanded = [mpsGraph reshapeTensor:gradOutputTensorExpanded
                                                     withShape:grad_output_final_shape
                                                          name:nil];
          }
          MPSGraphTensor* input2Transpose = [mpsGraph transposeTensor: fwdInputTensorExpanded
                                                            dimension: -1
                                                        withDimension: -2
                                                                 name: nil];
          gradInputTensor = [mpsGraph matrixMultiplicationWithPrimaryTensor:gradOutputTensorExpanded
                                                            secondaryTensor:input2Transpose
                                                                       name:nil];
        }
        else {
            
          if(input1_final_shape) {
              fwdInputTensorExpanded = [mpsGraph reshapeTensor:fwdInputTensorExpanded
                                                     withShape:input1_final_shape
                                                          name:nil];
          }
          if(grad_output_final_shape) {
              gradOutputTensorExpanded = [mpsGraph reshapeTensor:gradOutputTensorExpanded
                                                       withShape:grad_output_final_shape
                                                            name:nil];
          }
          MPSGraphTensor* input1Transpose = [mpsGraph transposeTensor: fwdInputTensorExpanded
                                                            dimension: -1
                                                        withDimension: -2
                                                                 name: nil];
          gradInputTensor = [mpsGraph matrixMultiplicationWithPrimaryTensor:input1Transpose
                                                            secondaryTensor:gradOutputTensorExpanded
                                                                       name:nil];
        }
        if(needs_reduction) {
          gradInputTensor = [mpsGraph reductionSumWithTensor: gradInputTensor
                                                        axes: reduction_axes
                                                        name: nil];
        }
            
          gradInputTensor = [mpsGraph reshapeTensor:gradInputTensor
                                          withShape:grad_input_shape
                                               name:nil];

          newCachedGraph->fwdInputTensor_  = fwdInputTensor;
          newCachedGraph->gradOutputTensor_ = gradOutputTensor;
          newCachedGraph->gradInputTensor_ = gradInputTensor;
        }
        return newCachedGraph;
      });
      cachedGraph = static_cast<CachedGraph *>(tmpCachedGraph);
    }
    
    Tensor fwd_input = is_grad_input1? tensor2 : tensor1;
    Placeholder fwdInputPlaceholder  = Placeholder(cachedGraph->fwdInputTensor_,  fwd_input);
    Placeholder gradOutputPlaceholder = Placeholder(cachedGraph->gradOutputTensor_, grad_output);
    Placeholder gradInputPlaceholder = Placeholder(cachedGraph->gradInputTensor_, grad_input);

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
      fwdInputPlaceholder.getMPSGraphTensor()  : fwdInputPlaceholder.getMPSGraphTensorData(),
      gradOutputPlaceholder.getMPSGraphTensor() : gradOutputPlaceholder.getMPSGraphTensorData()
    };

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
      gradInputPlaceholder.getMPSGraphTensor() : gradInputPlaceholder.getMPSGraphTensorData(),
    };

    mps::runMPSGraph(stream, cachedGraph->graph(), feeds, results);
  }

  return grad_input;
}


TORCH_IMPL_FUNC(mm_out_mps)(const Tensor& self, const Tensor& mat2, const Tensor& result) {
  mm_out_mps_impl(self, mat2, const_cast<Tensor&>(result));
}

TORCH_IMPL_FUNC(addmm_out_mps)(const Tensor& self, const Tensor& mat1, const Tensor& mat2, const Scalar& beta, const Scalar& alpha, const Tensor& result) {
  addmm_out_mps_impl(self, mat1, mat2, beta, alpha, const_cast<Tensor&>(result));
}

TORCH_IMPL_FUNC(bmm_out_mps) (const Tensor & batch1, const Tensor & batch2, const Tensor & result) {
  bmm_out_mps_impl(batch1, batch2, const_cast<Tensor&>(result));
}

TORCH_IMPL_FUNC(baddbmm_out_mps) (const Tensor & self, const Tensor & batch1, const Tensor & batch2, const Scalar& beta, const Scalar& alpha, const Tensor& result) {
  addbmm_or_baddbmm_out_mps_impl(self, batch1, batch2, beta, alpha, const_cast<Tensor&>(result), BADDBMM_OP_TYPE);
}

TORCH_IMPL_FUNC(ndmm_out_mps) (const Tensor & batch1, const Tensor & batch2, const Tensor & result) {
  ndmm_out_mps_impl(batch1, batch2, const_cast<Tensor&>(result));
}

Tensor& addbmm_out_mps(const Tensor& self, const Tensor& batch1, const Tensor& batch2, const Scalar& beta, const Scalar& alpha, Tensor& result) {
  auto b_self = expand_size(self, {batch1.size(1), batch2.size(2)}, "addbmm_out");

  addbmm_or_baddbmm_out_mps_impl(*b_self, batch1, batch2, beta, alpha, result, ADDBMM_OP_TYPE);
  return result;
}

Tensor addbmm_mps(const Tensor& self, const Tensor& batch1, const Tensor& batch2, const Scalar& beta, const Scalar& alpha) {
  Tensor result = at::empty({0}, self.options());
  return addbmm_out_mps(self, batch1, batch2, beta, alpha, result);
}

Tensor &addbmm_mps_(Tensor& self, const Tensor& batch1, const Tensor& batch2, const Scalar& beta, const Scalar& alpha) {
  return addbmm_out_mps(self, batch1, batch2, beta, alpha, self);
}

std::tuple<Tensor, Tensor> ndmm_backward_mps(
    const Tensor& input1, const Tensor& input2,
    const Tensor& grad, std::array<bool,2> grad_mask) {
    Tensor grad_input1, grad_input2;
  if(grad_mask[0])
      grad_input1 = ndmm_backward_out_mps(input1, input2, grad, true);
  if(grad_mask[1])
      grad_input2 = ndmm_backward_out_mps(input1, input2, grad, false);
    return std::tuple<Tensor, Tensor>(grad_input1, grad_input2);
}


} // namespace native
} // namespace at
