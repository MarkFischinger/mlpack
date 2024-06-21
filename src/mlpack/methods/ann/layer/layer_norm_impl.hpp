#ifndef MLPACK_METHODS_ANN_LAYER_LAYERNORM_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_LAYERNORM_IMPL_HPP

#include "layer_norm.hpp"
#include <omp.h>
#include <immintrin.h> 

namespace mlpack {

template <typename MatType>
LayerNormType<MatType>::LayerNormType(const double eps) :
    eps(eps)
{
    gammaTemp.set_size(size, 1);
    betaTemp.set_size(size, 1);
}

template<typename MatType>
void LayerNormType<MatType>::SetWeights(const MatType& weightsIn)
{
  MakeAlias(weights, weightsIn, 2 * size, 1);
  MakeAlias(gamma, weightsIn, size, 1);
  MakeAlias(beta, weightsIn, size, 1, gamma.n_elem);
}

template<typename MatType>
void LayerNormType<MatType>::CustomInitialize(
      MatType& W,
      const size_t elements)
{
  if (elements != 2 * size)
  {
    throw std::invalid_argument("LayerNormType::CustomInitialize(): wrong "
                                "elements size!");
  }
  
  MakeAlias(gammaTemp, W, size, 1);
  MakeAlias(betaTemp, W, size, 1, gammaTemp.n_elem);

  gammaTemp.fill(1.0);
  betaTemp.fill(0.0);
}

template<typename MatType>
void LayerNormType<MatType>::Forward(
    const MatType& input, MatType& output)
{
  #pragma omp parallel sections
  {
    #pragma omp section
    {
      mean = arma::mean(input, 0);
    }

    #pragma omp section
    {
      variance = arma::var(input, 1, 0);
    }
  }

  output = input.each_row() - mean;
  inputMean = output;

  size_t n_elem = output.n_elem;
  float* output_ptr = output.memptr();
  float* variance_ptr = variance.memptr();
  float sqrt_eps = sqrt(eps);

  #pragma omp parallel for
  for (size_t i = 0; i < n_elem; i += 8)
  {
    __m256 v = _mm256_loadu_ps(&variance_ptr[i]);
    __m256 sqrt_v = _mm256_sqrt_ps(_mm256_add_ps(v, _mm256_set1_ps(sqrt_eps)));
    __m256 out = _mm256_loadu_ps(&output_ptr[i]);
    _mm256_storeu_ps(&output_ptr[i], _mm256_div_ps(out, sqrt_v));
  }

  normalized = output;

  output.each_col() %= gamma;
  output.each_col() += beta;
}

template<typename MatType>
void LayerNormType<MatType>::Backward(
    const MatType& /* input */,
    const MatType& /* output */,
    const MatType& gy,
    MatType& g)
{
  const MatType stdInv = 1.0 / sqrt(variance + eps);
  const MatType norm = gy.each_col() % gamma;

  MatType var;
  #pragma omp parallel for
  for (size_t i = 0; i < var.n_elem; ++i)
  {
    var(i) = sum(norm % inputMean, 0) % pow(stdInv, 3.0) * -0.5;
  }

  #pragma omp parallel for
  for (size_t i = 0; i < g.n_elem; ++i)
  {
    g(i) = (norm.each_row() % stdInv) + (inputMean.each_row() %
      var * 2 / gy.n_rows);
  }

  g.each_row() += sum(norm.each_row() % -stdInv, 0) / gy.n_rows;
}

template<typename MatType>
void LayerNormType<MatType>::Gradient(
    const MatType& /* input */,
    const MatType& error,
    MatType& gradient)
{
  gradient.set_size(size + size, 1);

  #pragma omp parallel sections
  {
    #pragma omp section
    {
      gradient.submat(0, 0, gamma.n_elem - 1, 0) = sum(normalized % error, 1);
    }

    #pragma omp section
    {
      gradient.submat(gamma.n_elem, 0, gradient.n_elem - 1, 0) = sum(error, 1);
    }
  }
}

template<typename MatType>
template<typename Archive>
void LayerNormType<MatType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(cereal::base_class<Layer<MatType>>(this));
  ar(CEREAL_NVP(size));
  ar(CEREAL_NVP(eps));
}

} // namespace mlpack

#endif
