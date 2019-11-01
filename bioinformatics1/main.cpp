
#include <stdio.h>
#include <iostream>
#include <fstream>
#include "OriReader.h"

using namespace std;

int main()
{
	const char* m_Ori = "ATCAATGATCAACGTAAGCTTCTAAGCATGATCAAGGTGCTCACACAGTTTATCCACAACCTGAGTGGATGACATCAAGATAGGTCGTT\
GTATCTCCTTCCTCTCGTACTCTCATGACCACGGAAAGATGATCAAGAGAGGATGATTTCTTGGCCATATCGCAATGAATACTTGTGACTTGTGCTTCCAATTGACATCTTC\
AGCGCCATATTGCGCTGGCCAAGGTGACGGAGCGGGATTACGAAAGCATGATCATGGCTGTTGTTCTGTTTATCTTGTTTTGACTGAGACTTGTTAGGATAGACGGTTTTTCA\
TCACTGACTAGCCAAAGCCTTACTCTGCCTGACATCGACCGTAAATTGATAATGAATTTACATGCTTCCGCGACGATTTACCTCTTGATCATCGATCCGATTGAAGATCTTCA\
ATTGTTAATTCTCTTGCCTCGACTCATAGCCATGATGAGCTCTTGATCATGTTTCCTTAACCCTCTATTTTTTACGGAAGAATGATCAAGCTGCTGCTCTTGATCATCGTTTC";

	OriReader* m_Reader = new OriReader();
	char* m_Genome = nullptr;

	ifstream oriFile("Vibrio_cholerae.txt", ios::in);
	
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

		// Find the most frequent k-mers inside Ori region
		m_Reader->FindMostFrequentWordInMap(m_Ori, 9);
		m_Reader->FindMostFrequentWordInMapWithCuda(m_Ori, 9);

		//// Find the most frequent k-mers inside the entirely Genome
		//m_Reader->FindMostFrequentWordInMap(m_Genome, 9);
		//m_Reader->FindMostFrequentWordInMapWithCuda(m_Genome, 9);

		//// Find how many times and the positions inside the arrray of k-mer inside the Genome
		//cout << m_Reader->PatternCount(m_Genome, "CTTGATCAT") << endl;
		//auto positions = m_Reader->PatternPositions(m_Genome, "CTTGATCAT");
		//for (int &pos : positions) {
		//	cout << pos << " ";
		//}
		//cout << endl;
	}

	//char text[] = "AAAACCCGGT";
	//unique_ptr<char> reverse(m_OriReader->ReverseComplement(text)) ;
	//cout << text << endl << reverse.get() << endl;

	delete m_Genome;
	delete m_Reader;
	//system("PAUSE");
	return 0;
}