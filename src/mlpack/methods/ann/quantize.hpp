#ifndef MLPACK_METHODS_ANN_QUANTIZATION_QUANTIZE_HPP
#define MLPACK_METHODS_ANN_QUANTIZATION_QUANTIZE_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/methods/ann/ffn/ffn.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include "quantization_utils.hpp"

namespace mlpack {
namespace ann {

/**
 * Quantize a neural network from one matrix type to another.
 *
 * @tparam TargetType The target element type for quantization (e.g., int8_t).
 * @tparam NetworkType The type of the original network.
 * @param network The original network to be quantized.
 * @return A new network with weights quantized to the TargetType.
 */
template<typename TargetType, typename NetworkType>
auto Quantize(const NetworkType& network)
{
  using SourceMatType = typename NetworkType::MatType;
  using TargetMatType = typename SourceMatType::template changed_type<TargetType>::type;
  
  NetworkType quantizedNetwork;
  
  // Copy network parameters
  quantizedNetwork.InputDimensions() = network.InputDimensions();
  quantizedNetwork.OutputDimensions() = network.OutputDimensions();
  quantizedNetwork.Reset() = network.Reset();
  quantizedNetwork.NumFunctions() = network.NumFunctions();

  // Quantize each layer
  for (size_t i = 0; i < network.Network().size(); ++i)
  {
    auto quantizedLayer = network.Network()[i]->template As<TargetMatType>();
    
    // Quantize weights if the layer has them
    if (quantizedLayer->Parameters().n_elem > 0)
    {
      TargetMatType quantizedWeights;
      double scaleFactor = FindQuantizationScale<TargetType>(network.Network()[i]->Parameters());
      QuantizeWeights(network.Network()[i]->Parameters(), quantizedWeights, scaleFactor);
      quantizedLayer->Parameters() = std::move(quantizedWeights);
    }

    quantizedNetwork.Network().push_back(quantizedLayer);
  }

  // Reset the network to ensure correct weight aliasing
  quantizedNetwork.Reset();

  return quantizedNetwork;
}

} // namespace ann
} // namespace mlpack

#endif
