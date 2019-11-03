#pragma once

#include <string>
#include <map>
#include <unordered_map>

void FMFW1(const char* const in, const int& sizeWord, unsigned int sizeIN);
void FMFW2(const char* const in, const int& sizeWord, unsigned int sizeIN);

class OriReader
{

public:

	typedef std::unordered_map<std::string, int> FrequencyMap;
	OriReader();
    ~OriReader();

    int PatternCount(const std::string& text, const std::string& pattern);

	std::vector<int> PatternPositions(const std::string & Text, const std::string & Pattern);

    void FindMostFrequentWordInMap(const char* const Text, const int& k);

	void FindMostFrequentWordInMapWithCuda(const char* const Text, const int& k);

	std::unique_ptr<char> ReverseComplement(const char* text);

private:

	FrequencyMap* m_FrequencyMap;
};

