#include "OriReader.h"


OriReader::OriReader()
{
}


OriReader::~OriReader()
{
}

int OriReader::PatternCount(const std::string& Text, const std::string& Pattern)
{
    int count = 0;
    int patternSize = Pattern.length();
    for (int i = 0; i < Text.length() - patternSize + 1 ; ++i)
    {
        if (Text.substr(i, patternSize) == Pattern)
            count++;
    }
    return count;
}


std::unique_ptr<OriReader::FrequencyMap> OriReader::GenerateFrequencyMap(const std::string& Text, const int& k)
{
    std::unique_ptr<OriReader::FrequencyMap> frequencyMap(new OriReader::FrequencyMap);
    for (int i = 0; i < Text.length() - k + 1; ++i)
    {
        (*frequencyMap.get())[Text.substr(i, k)] += 1;
    }
    return frequencyMap;
}
