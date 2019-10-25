#pragma once

#include <string>
#include <map>
#include <memory>

class OriReader
{

public:

    typedef std::map<std::string, int> FrequencyMap;

    OriReader();
    ~OriReader();

    static int PatternCount(const std::string& Text, const std::string& Pattern);
    static std::unique_ptr<FrequencyMap> GenerateFrequencyMap(const std::string& Text, const int& k);
};

