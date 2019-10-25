#pragma once

#include <string>
#include <map>
#include <unordered_map>

class OriReader
{

public:

	typedef std::unordered_map<std::string, int> FrequencyMap;
	OriReader();
    ~OriReader();

    int PatternCount(const std::string& Text, const std::string& Pattern);

    void GenerateFrequencyMap(const std::string& Text, const int& k);

	void PrintFrequencyMap(FrequencyMap* = nullptr);

	void FindMostFrequentWord();

private:

	FrequencyMap* m_FrequencyMap;
};

