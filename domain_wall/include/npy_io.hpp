#pragma once

#include <cstdint>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct NpyArray
{
    std::vector<std::size_t> shape;
    std::vector<double> data;
};

inline void save_npy_float64(
    const std::string& filename,
    const std::vector<std::size_t>& shape,
    const std::vector<double>& data)
{
    std::size_t expected = 1;
    for (std::size_t n : shape)
        expected *= n;

    if (expected != data.size())
        throw std::runtime_error("NPY shape does not match data size");

    std::ostringstream shape_text;
    shape_text << "(";

    for (std::size_t i = 0; i < shape.size(); ++i)
    {
        shape_text << shape[i];

        if (shape.size() == 1 || i + 1 < shape.size())
            shape_text << ",";

        if (i + 1 < shape.size())
            shape_text << " ";
    }

    shape_text << ")";

    std::string header =
        "{'descr': '<f8', 'fortran_order': False, 'shape': "
        + shape_text.str() + ", }";

    const std::size_t preamble_size = 10;
    std::size_t padding =
        16 - ((preamble_size + header.size() + 1) % 16);

    if (padding == 16)
        padding = 0;

    header.append(padding, ' ');
    header.push_back('\n');

    std::ofstream out(filename, std::ios::binary);

    if (!out)
        throw std::runtime_error("Cannot create " + filename);

    out.write("\x93NUMPY", 6);

    const unsigned char version[2] = {1, 0};
    out.write(reinterpret_cast<const char*>(version), 2);

    const std::uint16_t header_length =
        static_cast<std::uint16_t>(header.size());

    out.write(
        reinterpret_cast<const char*>(&header_length),
        sizeof(header_length));

    out.write(
        header.data(),
        static_cast<std::streamsize>(header.size()));

    out.write(
        reinterpret_cast<const char*>(data.data()),
        static_cast<std::streamsize>(data.size() * sizeof(double)));
}

inline NpyArray load_npy_float64(const std::string& filename)
{
    std::ifstream in(filename, std::ios::binary);

    if (!in)
        throw std::runtime_error("Cannot open " + filename);

    char magic[6];
    in.read(magic, 6);

    if (std::string(magic, 6) != "\x93NUMPY")
        throw std::runtime_error("Not an NPY file");

    unsigned char major = 0;
    unsigned char minor = 0;
    in.read(reinterpret_cast<char*>(&major), 1);
    in.read(reinterpret_cast<char*>(&minor), 1);

    std::uint32_t header_length = 0;

    if (major == 1)
    {
        std::uint16_t length16 = 0;
        in.read(reinterpret_cast<char*>(&length16), 2);
        header_length = length16;
    }
    else
    {
        in.read(reinterpret_cast<char*>(&header_length), 4);
    }

    std::string header(header_length, '\0');
    in.read(header.data(), header_length);

    if (header.find("<f8") == std::string::npos)
        throw std::runtime_error("Expected little-endian float64 NPY data");

    std::smatch match;
    const std::regex shape_regex(
        R"('shape'\s*:\s*\(([^\)]*)\))");

    if (!std::regex_search(header, match, shape_regex))
        throw std::runtime_error("Could not parse NPY shape");

    std::vector<std::size_t> shape;
    const std::regex number_regex(R"((\d+))");
    const std::string shape_text = match[1].str();

    for (std::sregex_iterator it(
             shape_text.begin(), shape_text.end(), number_regex),
         end;
         it != end;
         ++it)
    {
        shape.push_back(
            static_cast<std::size_t>(
                std::stoull((*it)[1].str())));
    }

    std::size_t count = 1;
    for (std::size_t n : shape)
        count *= n;

    std::vector<double> data(count);

    in.read(
        reinterpret_cast<char*>(data.data()),
        static_cast<std::streamsize>(count * sizeof(double)));

    if (!in)
        throw std::runtime_error("Truncated NPY data");

    return {shape, std::move(data)};
}
