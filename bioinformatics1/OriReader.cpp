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

int OriReader::PatternCount(const std::string& text, const std::string& pattern)
{
    int count = 0;
    int patternSize = pattern.length();
    for (int i = 0; i < text.length() - patternSize + 1 ; ++i)
    {
        if (text.substr(i, patternSize) == pattern)
            count++;
    }
    return count;
}


std::vector<int> OriReader::PatternPositions(const std::string& text, const std::string& pattern) {

	std::vector<int> positions;
	int patternSize = pattern.length();

	for (int i = 0; i < text.length() - patternSize + 1; ++i)
	{
		if (text.substr(i, patternSize) == pattern)
			positions.push_back(i);
	}
	return positions;
}


bool value_comparer(FrequencyMap::value_type &i1, FrequencyMap::value_type &i2)
{
	return i1.second < i2.second;
}

void OriReader::FindMostFrequentWordInMap(const char* const text, const int& k)
{    
	m_FrequencyMap->clear();

	const int size = strlen(text);

	FMFW(text, k, size);
	std::string textString = text;
    for (int i = 0; i < size - k + 1; ++i)
    {
        (*m_FrequencyMap)[textString.substr(i, k)] += 1;
    }

	//for (auto word = m_FrequencyMap->begin(); word != m_FrequencyMap->end(); ++word)
	//{
	//	std::cout << word->first << "  " << word->second << std::endl;
	//}
	//std::cout << std::endl;

	FrequencyMap::iterator itor = std::max_element(m_FrequencyMap->begin(), m_FrequencyMap->end(), value_comparer);

	std::cout << itor->second << std::endl;

	std::vector<std::string> mostFrequentWords;
	for (auto &word : *m_FrequencyMap) {
		if (word.second == itor->second) {
			mostFrequentWords.push_back(word.first);
		}
	}

	for (int i = 0; i < mostFrequentWords.size(); i++)
	{
		std::cout << mostFrequentWords[i] << std::endl;
	}
}

void OriReader::FindMostFrequentWordInMapWithCuda(const char * const text, const int & k)
{
	const int size = strlen(text);

	FMFW(text, k, size);
}


std::unique_ptr<char> OriReader::ReverseComplement(const char* const text)
{
	const int size = strlen(text);
	std::unique_ptr<char> reverseText(new char[size + 1]);
	reverseText.get()[size] = '\0';

	reverseText.get()[0] = text[0];
	reverseText.get()[size-1] = text[size-1];
	for (int i = 1; i < size - 1; i++) {
		char next = text[size - i - 1];

		if (next == 'A') next = 'T';
		else if (next == 'T') next = 'A';
		else if (next == 'G') next = 'C';
		else if (next == 'C') next = 'G';

		reverseText.get()[i] = next;
	}
	return reverseText;
}
