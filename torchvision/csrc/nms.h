#pragma once
#include "cpu/vision.h"

#ifdef WITH_CUDA
  #include "cuda/vision.h"
#endif


at::Tensor nms(const at::Tensor& dets,
               const at::Tensor& scores,
               const float threshold) {

  if (dets.type().is_cuda()) {
#ifdef WITH_CUDA
    if (dets.numel() == 0)
      return at::empty({0}, dets.options().dtype(at::kFloat).device(at::kCUDA));
    return nms_cuda(dets, scores, threshold);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }

  at::Tensor result = nms_cpu(dets, scores, threshold);
  return result;
}
