#pragma once

#include <stdio.h>
#include <iostream>
#include <fstream>

#include <cuda_profiler_api.h>
#include "cuda_runtime.h"
#include "utils.h"

#include "OriReader.h"

using namespace std;

void FMFW1(const char* const in, const int& sizeWord, unsigned int sizeIN);
void FMFW2(const char* const in, const int& sizeWord, unsigned int sizeIN);

int main()
{
	const char* m_Ori = "ATCAATGATCAACGTAAGCTTCTAAGCATGATCAAGGTGCTCACACAGTTTATCCACAACCTGAGTGGATGACATCAAGATAGGTCGTT\
GTATCTCCTTCCTCTCGTACTCTCATGACCACGGAAAGATGATCAAGAGAGGATGATTTCTTGGCCATATCGCAATGAATACTTGTGACTTGTGCTTCCAATTGACATCTTC\
AGCGCCATATTGCGCTGGCCAAGGTGACGGAGCGGGATTACGAAAGCATGATCATGGCTGTTGTTCTGTTTATCTTGTTTTGACTGAGACTTGTTAGGATAGACGGTTTTTCA\
TCACTGACTAGCCAAAGCCTTACTCTGCCTGACATCGACCGTAAATTGATAATGAATTTACATGCTTCCGCGACGATTTACCTCTTGATCATCGATCCGATTGAAGATCTTCA\
ATTGTTAATTCTCTTGCCTCGACTCATAGCCATGATGAGCTCTTGATCATGTTTCCTTAACCCTCTATTTTTTACGGAAGAATGATCAAGCTGCTGCTCTTGATCATCGTTTC";

	const unsigned long GIGABYTE = 1073741824UL;
	const unsigned long long sizeGPUHeap = 4ULL * GIGABYTE;
	cudaDeviceSetLimit(cudaLimitMallocHeapSize, sizeGPUHeap);

	OriReader* m_Reader = new OriReader();
	char* m_Genome = nullptr;

	// Find the most frequent k-mers inside Ori region
	m_Reader->FindMostFrequentWordInMap(m_Ori, 9);

	const int sizeOri = strlen(m_Ori);
	FMFW2(m_Ori, 9, sizeOri);
	FMFW1(m_Ori, 9, sizeOri);

	ifstream oriFile("D:/GitProject/cudaTest/bio1/Vibrio_cholerae.txt", ios::in);	
	if (oriFile.is_open()) {

		//Get the size of the file content
		oriFile.seekg(0, oriFile.end);
		int sizeText = oriFile.tellg();
		oriFile.seekg(0, oriFile.beg);

		m_Genome = new char[sizeText];

		oriFile.read(m_Genome, sizeText);
		if (oriFile)
			cout << "all characters read successfully." << endl;
		else
			cout << "error: only " << oriFile.gcount() << " could be read" << endl;
		oriFile.close();

		// Find the most frequent k-mers inside the entirely Genome
		m_Reader->FindMostFrequentWordInMap(m_Genome, 9);

		////cudaProfilerStart();
		const int sizeGenome = strlen(m_Genome);
		FMFW2(m_Genome, 9, sizeGenome);
		////cudaProfilerStop();
		FMFW1(m_Genome, 9, sizeGenome);

		// Find how many times and the positions inside the arrray of k-mer inside the Genome
		////cout << m_Reader->PatternCount(m_Genome, "CTTGATCAT") << endl;
		////auto positions = m_Reader->PatternPositions(m_Genome, "CTTGATCAT");
		////for (int &pos : positions) {
		////	cout << pos << " ";
		////}
		////cout << endl;
	}

	////char text[] = "AAAACCCGGT";
	////unique_ptr<char> reverse(m_OriReader->ReverseComplement(text)) ;
	////cout << text << endl << reverse.get() << endl;

	delete m_Genome;
	delete m_Reader;
	////system("PAUSE");

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaError_t cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}