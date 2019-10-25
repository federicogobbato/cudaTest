#include "OriReader.h"
#include <iostream>
#include <algorithm>
#include <vector>

typedef std::map<std::string, int> FrequencyMap;

OriReader::OriReader() : m_FrequencyMap(new OriReader::FrequencyMap())
{
}

OriReader::~OriReader()
{
	delete m_FrequencyMap;
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


void OriReader::GenerateFrequencyMap(const std::string& Text, const int& k)
{    
    for (int i = 0; i < Text.length() - k + 1; ++i)
    {
        (*m_FrequencyMap)[Text.substr(i, k)] += 1;
    }
}


void OriReader::PrintFrequencyMap(FrequencyMap* freqMap)
{
	if (freqMap == nullptr)
		freqMap = m_FrequencyMap;

	for (auto word = freqMap->begin(); word != freqMap->end(); ++word)
	{
		std::cout << word->first << "  " << word->second << std::endl;
	}
	std::cout << std::endl;
}


bool value_comparer(FrequencyMap::value_type &i1, FrequencyMap::value_type &i2)
{
	return i1.second < i2.second;
}

void OriReader::FindMostFrequentWord()
{
	FrequencyMap::iterator itor = std::max_element(m_FrequencyMap->begin(), m_FrequencyMap->end(), value_comparer);

	std::vector<std::string> mostFrequentWords;
	for (auto &word : *m_FrequencyMap) {
		if (word.second == itor->second) {
			mostFrequentWords.push_back(word.first);
		}
	}

	for (int i=0; i< mostFrequentWords.size(); i++)
	{
		std::cout << mostFrequentWords[i] << std::endl;
	}
}
