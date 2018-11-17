/**
 * @file print_doc_functions_impl.hpp
 * @author Ryan Curtin
 *
 * This file contains functions useful for printing documentation strings
 * related to Julia bindings.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_JULIA_PRINT_DOC_FUNCTIONS_IMPL_HPP
#define MLPACK_BINDINGS_JULIA_PRINT_DOC_FUNCTIONS_IMPL_HPP

#include <mlpack/core/util/hyphenate_string.hpp>

namespace mlpack {
namespace bindings {
namespace julia {

/**
 * Given a parameter type, print the corresponding value.
 */
template<typename T>
inline std::string PrintValue(const T& value, bool quotes)
{
  std::ostringstream oss;
  if (quotes)
    oss << "`";
  oss << value;
  if (quotes)
    oss << "`";
  return oss.str();
}

// Special overload for booleans.
template<>
inline std::string PrintValue(const bool& value, bool quotes)
{
  if (quotes && value)
    return "\"true\"";
  else if (quotes && !value)
    return "\"false\"";
  else if (!quotes && value)
    return "true";
  else
    return "false";
}

// Recursion base case.
std::string CreateInputArguments() { return ""; }

/**
 * This prints anything that is required to create an input value.  We only need
 * to create input values for matrices.
 */
template<typename T, typename... Args>
std::string CreateInputArguments(const std::string& paramName,
                                 const T& value,
                                 Args... args)
{
  // We only need to do anything if it is an input option.
  if (CLI::Parameters().count(paramName) > 0)
  {
    const util::ParamData& d = CLI::Parameters()[paramName];
    std::ostringstream oss;

    if (d.input)
    {
      if (d.cppType == "arma::mat")
      {
        oss << "julia> " << value << " = rand(100, 10)" << std::endl;
      }
      else if (d.cppType == "arma::Mat<size_t>")
      {
        oss << "julia> " << value << " = rand(UInt64, (100, 10))" << std::endl;
      }
    }

    oss << CreateInputArguments(args...);

    return oss.str();
  }
  else
  {
    // Unknown parameter!
    throw std::runtime_error("Unknown parameter '" + paramName + "' " +
        "encountered while assembling documentation!  Check PROGRAM_INFO() " +
        "declaration.");
  }
}

// Recursion base case.
std::string PrintInputOptions() { return ""; }

/**
 * This prints an argument, assuming that it is already known whether or not it
 * is required.
 */
template<typename T>
std::string PrintInputOption(const std::string& paramName,
                             const T& value,
                             const bool required)
{
  std::ostringstream oss;
  if (!required)
    oss << paramName << "=";
  oss << value;
  return oss.str();
}

// Base case: no modification needed.
void GetOptions(
    std::vector<std::tuple<std::string, std::string>>& /* results */,
    bool /* input */)
{
  // Nothing to do.
}

/**
 * Assemble a vector of string tuples indicating parameter names and what should
 * be printed for them.  (For output parameters, we just need to print the
 * value.)
 */
template<typename T, typename... Args>
void GetOptions(
    std::vector<std::tuple<std::string, std::string>>& results,
    bool input,
    const std::string& paramName,
    const T& value,
    Args... args)
{
  // Determine whether or not the value is required.
  if (CLI::Parameters().count(paramName) > 0)
  {
    const util::ParamData& d = CLI::Parameters()[paramName];

    if (d.input && input)
    {
      // Print and add to results.
      results.push_back(std::make_tuple(paramName,
          PrintInputOption(paramName, value, d.required)));
    }
    else
    {
      std::ostringstream oss;
      oss << value;
      results.push_back(std::make_tuple(paramName, oss.str()));
    }

    GetOptions(results, input, args...);
  }
  else
  {
    // Unknown parameter!
    throw std::runtime_error("Unknown parameter '" + paramName + "' " +
        "encountered while assembling documentation!  Check PROGRAM_INFO() " +
        "declaration.");
  }
}

/**
 * Print the input options for a program call.  For a parameter 'x' with value
 * '5', this will print something like x=5; however, the 'x=' will be omitted if
 * the parameter is required.
 */
template<typename... Args>
std::string PrintInputOptions(Args... args)
{
  // Gather list of required and non-required options.
  std::vector<std::string> inputOptions;
  for (auto it = CLI::Parameters().begin(); it != CLI::Parameters().end(); ++it)
  {
    const util::ParamData& d = it->second;
    if (d.input && d.required)
    {
      // Ignore some parameters.
      if (d.name != "help" && d.name != "info" &&
          d.name != "version")
        inputOptions.push_back(it->first);
    }
  }

  for (auto it = CLI::Parameters().begin(); it != CLI::Parameters().end(); ++it)
  {
    const util::ParamData& d = it->second;
    if (d.input && !d.required &&
        d.name != "help" && d.name != "info" &&
        d.name != "version")
      inputOptions.push_back(it->first);
  }

  // Now collect the way that we print all the parameters.
  std::vector<std::tuple<std::string, std::string>> printedParameters;
  GetOptions(printedParameters, true, args...);

  // Next, we need to match each option.  Note that required options will come
  // first.
  std::ostringstream oss;
  bool doneWithRequired = false;
  bool printedAny = false;
  for (size_t i = 0; i < inputOptions.size(); ++i)
  {
    const util::ParamData& d = CLI::Parameters()[inputOptions[i]];
    // Does this option exist?
    bool found = false;
    size_t index = printedParameters.size();
    for (size_t j = 0; j < printedParameters.size(); ++j)
    {
      if (inputOptions[i] == std::get<0>(printedParameters[j]))
      {
        found = true;
        index = j;
        break;
      }
    }

    if (found)
    {
      // Print this as an option.  We may need a preceding comma.
      if (printedAny)
      {
        if (!d.required && !doneWithRequired)
        {
          doneWithRequired = true;
          oss << "; ";
        }
        else
        {
          oss << ", ";
        }
      }
      else if (!d.required && !doneWithRequired)
      {
        // No required arguments for this binding.
        doneWithRequired = true;
      }

      // Print the parameter itself.
      printedAny = true;
      oss << std::get<1>(printedParameters[index]);
    }
    else if (d.required)
    {
      throw std::invalid_argument("Required parameter '" + inputOptions[i] +
          "' not passed in list of input arguments to PROGRAM_CALL()!");
    }
  }

  return oss.str();
}

// Recursion base case.
inline std::string PrintOutputOptions() { return ""; }

template<typename... Args>
std::string PrintOutputOptions(Args... args)
{
  // Get the list of output options for the binding.
  std::vector<std::string> outputOptions;
  for (auto it = CLI::Parameters().begin(); it != CLI::Parameters().end(); ++it)
  {
    const util::ParamData& d = it->second;
    if (!d.input)
      outputOptions.push_back(it->first);
  }

  // Now get the full list of output options that we have.
  std::vector<std::tuple<std::string, std::string>> passedOptions;
  GetOptions(passedOptions, false, args...);

  // Next, iterate over all the options.
  std::ostringstream oss;
  for (size_t i = 0; i < outputOptions.size(); ++i)
  {
    // Does this option exist?
    bool found = false;
    size_t index = passedOptions.size();
    for (size_t j = 0; j < passedOptions.size(); ++j)
    {
      if (outputOptions[i] == std::get<0>(passedOptions[j]))
      {
        found = true;
        index = j;
        break;
      }
    }

    if (found)
    {
      // We have received this option, so print it.
      if (i > 0)
        oss << ", ";
      oss << std::get<1>(passedOptions[index]);
    }
    else
    {
      // We don't care about this option.
      if (i > 0)
        oss << ", ";
      oss << "_";
    }
  }

  return oss.str();
}

/**
 * Given a name of a binding and a variable number of arguments (and their
 * contents), print the corresponding function call.
 */
template<typename... Args>
std::string ProgramCall(const std::string& programName, Args... args)
{
  std::ostringstream oss;
  oss << "```jldoctest" << std::endl;

  // Print any input argument definitions.
  oss << CreateInputArguments(args...);

  oss << "julia> ";

  // Find out if we have any output options first.
  std::ostringstream ossOutput;
  ossOutput << PrintOutputOptions(args...);
  if (ossOutput.str() != "")
    oss << ossOutput.str() << " = ";
  oss << programName << "(";

  // Now process each input option.
  oss << PrintInputOptions(args...);
  oss << ")" << std::endl;
  oss << "```";

  return util::HyphenateString(oss.str(), 0);
}

/**
 * Given the name of a model, print it.  Here we do not need to modify anything.
 */
inline std::string PrintModel(const std::string& modelName)
{
  return "`" + modelName + "`";
}

/**
 * Given the name of a matrix, print it.  Here we do not need to modify
 * anything.
 */
inline std::string PrintDataset(const std::string& datasetName)
{
  return "`" + datasetName + "`";
}

/**
 * Given the name of a binding, print its invocation.
 */
inline std::string ProgramCall(const std::string& programName)
{
  return "julia> " + programName + "(";
}

/**
 * Print any closing call to a program.  For a Python binding this is a closing
 * brace.
 */
inline std::string ProgramCallClose()
{
  return ")";
}

/**
 * Given the parameter name, determine what it would actually be when passed to
 * the command line.
 */
inline std::string ParamString(const std::string& paramName)
{
  // For a Julia binding we don't need to know the type.
  return "`" + paramName + "`";
}

/**
 * Given the parameter name and an argument, return what should be written as
 * documentation when referencing that argument.
 */
template<typename T>
inline std::string ParamString(const std::string& paramName, const T& value)
{
  std::ostringstream oss;
  oss << paramName << " = " << value;
  return oss.str();
}

inline bool IgnoreCheck(const std::string& paramName)
{
  return !CLI::Parameters()[paramName].input;
}

inline bool IgnoreCheck(const std::vector<std::string>& constraints)
{
  for (size_t i = 0; i < constraints.size(); ++i)
  {
    if (!CLI::Parameters()[constraints[i]].input)
      return true;
  }

  return false;
}

inline bool IgnoreCheck(
    const std::vector<std::pair<std::string, bool>>& constraints,
    const std::string& paramName)
{
  for (size_t i = 0; i < constraints.size(); ++i)
  {
    if (!CLI::Parameters()[constraints[i].first].input)
      return true;
  }

  return !CLI::Parameters()[paramName].input;
}

} // namespace julia
} // namespace bindings
} // namespace mlpack

#endif
