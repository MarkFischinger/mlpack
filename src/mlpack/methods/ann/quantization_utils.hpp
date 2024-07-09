#ifndef MLPACK_METHODS_ANN_QUANTIZATION_QUANTIZATION_UTILS_HPP
#define MLPACK_METHODS_ANN_QUANTIZATION_QUANTIZATION_UTILS_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace ann {

/**
 * Find the quantization scale factor for converting weights to a target type.
 *
 * @tparam TargetType The target type for quantization (e.g., int8_t).
 * @tparam MatType The source matrix type.
 * @param weights The weights to be quantized.
 * @return The scale factor for quantization.
 */
template<typename TargetType, typename MatType>
double FindQuantizationScale(const MatType& weights)
{
  double maxAbs = arma::abs(weights).max();
  double targetTypeMax = std::numeric_limits<TargetType>::max();
  return targetTypeMax / maxAbs;
}

/**
 * Quantize weights from one type to another using the given scale factor.
 *
 * @tparam SourceMatType The source matrix type.
 * @tparam TargetMatType The target matrix type for quantization.
 * @param sourceWeights The original weights.
 * @param targetWeights The quantized weights (output).
 * @param scaleFactor The scale factor for quantization.
 */
template<typename SourceMatType, typename TargetMatType>
void QuantizeWeights(const SourceMatType& sourceWeights,
                     TargetMatType& targetWeights,
                     double scaleFactor)
{
  targetWeights = arma::conv_to<TargetMatType>::from(
      arma::clamp(sourceWeights * scaleFactor, 
                  std::numeric_limits<typename TargetMatType::elem_type>::min(),
                  std::numeric_limits<typename TargetMatType::elem_type>::max()));
}

/**
 * Dequantize weights from one type to another using the given scale factor.
 *
 * @tparam SourceMatType The source matrix type (quantized weights).
 * @tparam TargetMatType The target matrix type for dequantization.
 * @param sourceWeights The quantized weights.
 * @param targetWeights The dequantized weights (output).
 * @param scaleFactor The scale factor used for quantization.
 */
template<typename SourceMatType, typename TargetMatType>
void DequantizeWeights(const SourceMatType& sourceWeights,
                       TargetMatType& targetWeights,
                       double scaleFactor)
{
  targetWeights = arma::conv_to<TargetMatType>::from(sourceWeights) / scaleFactor;
}

} // namespace ann
} // namespace mlpack

#endif
