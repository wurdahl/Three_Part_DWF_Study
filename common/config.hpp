#pragma once

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

class TextConfig
{
public:
    explicit TextConfig(const std::string& filename)
    {
        std::ifstream input(filename);
        if (!input)
            throw std::runtime_error("Cannot open parameter file: " + filename);

        std::string line;
        int line_number = 0;
        while (std::getline(input, line))
        {
            ++line_number;
            const auto comment = line.find('#');
            if (comment != std::string::npos)
                line.erase(comment);
            trim(line);
            if (line.empty())
                continue;

            const auto equals = line.find('=');
            if (equals == std::string::npos)
                throw std::runtime_error(
                    filename + ":" + std::to_string(line_number)
                    + ": expected key = value");

            std::string key = line.substr(0, equals);
            std::string value = line.substr(equals + 1);
            trim(key);
            trim(value);
            if (key.empty() || value.empty())
                throw std::runtime_error(
                    filename + ":" + std::to_string(line_number)
                    + ": empty key or value");
            values_[key] = value;
        }
    }

    int integer_or(const std::string& key, int fallback) const
    {
        if (values_.find(key) == values_.end())
            return fallback;
        return integer(key);
    }

    int integer(const std::string& key) const
    {
        const std::string value = require(key);
        std::size_t used = 0;
        const int result = std::stoi(value, &used);
        ensure_consumed(key, value, used);
        return result;
    }

    unsigned long long unsigned_integer(const std::string& key) const
    {
        const std::string value = require(key);
        std::size_t used = 0;
        const auto result = std::stoull(value, &used);
        ensure_consumed(key, value, used);
        return result;
    }

    double real(const std::string& key) const
    {
        const std::string value = require(key);
        std::size_t used = 0;
        const double result = std::stod(value, &used);
        ensure_consumed(key, value, used);
        return result;
    }

    std::string text(const std::string& key) const { return require(key); }

private:
    std::unordered_map<std::string, std::string> values_;

    static void trim(std::string& value)
    {
        const auto not_space = [](unsigned char c) { return !std::isspace(c); };
        value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
        value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
    }

    std::string require(const std::string& key) const
    {
        const auto found = values_.find(key);
        if (found == values_.end())
            throw std::runtime_error("Missing required parameter: " + key);
        return found->second;
    }

    static void ensure_consumed(
        const std::string& key, const std::string& value, std::size_t used)
    {
        if (used != value.size())
            throw std::runtime_error("Invalid numeric value for " + key + ": " + value);
    }
};

inline const TextConfig& runtime_config()
{
    static const TextConfig config("parameters.txt");
    return config;
}
